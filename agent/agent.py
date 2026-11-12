"""A delivery operations assistant: an LLM that answers questions using the delivery data
and the prediction model. Works with any provider that speaks the OpenAI chat format."""
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

import tools

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Choose the provider here: "groq", "gemini" or "ollama"
PROVIDER = "groq"
PROVIDERS = {
    #           base URL                                                    API key variable   model
    "groq":   ("https://api.groq.com/openai/v1",                            "GROQ_API_KEY",   "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/",  "GEMINI_API_KEY", "gemini-flash-latest"),
    "ollama": ("http://localhost:11434/v1",                                 None,             "qwen3:8b"),
}
BASE_URL, KEY_NAME, MODEL = PROVIDERS[PROVIDER]
client = OpenAI(base_url=BASE_URL, api_key=os.getenv(KEY_NAME) if KEY_NAME else "ollama")

LOG_FILE = ROOT / "logs" / "conversations.jsonl"
MAX_STEPS = 8   # stop after this many rounds, so a confused agent can't loop forever

SYSTEM_PROMPT = f"""You are an analytics assistant for a food delivery operations team.
You answer questions about past deliveries using two tools: run_sql and predict_delivery_time.

## The data
{tools.TABLE_DESCRIPTION}
Notes on the data:
- Courier pressure is measured by orders_per_dasher. Do not compare busy with on-shift couriers:
  busy can exceed on-shift for administrative reasons.
- The SQL dialect is DuckDB: use median(x) and quantile_cont(x, 0.9).

## How to analyse
- Get every number from a tool. Never estimate.
- Use medians for delivery times, because a few very late deliveries skew averages.
- For "per day" questions, count per order_date first, then average across days.
- When comparing groups, report how many orders each is based on, and ignore groups under 100 orders.
- For courier pressure, compare bands of orders_per_dasher (under 0.5, 0.5-1, 1-1.5, 1.5-2, 2+).

## How to answer
- Lead with the answer and its key number, then one or two sentences of explanation.
- For predictions, give the time, the typical error (about 11 minutes) and the main assumption.
- If the data cannot answer the question, say so and say what it does contain.
- If the question is not about this data, decline briefly and suggest an example question.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run one read-only SQL SELECT query (DuckDB syntax) on the `deliveries` "
                           "table. Returns at most 50 rows as CSV.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A single SELECT query."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_delivery_time",
            "description": "Predict the delivery time in minutes for one order. "
                           "Inputs that are not given are set to typical values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "market_id": {"type": "integer", "description": "Market 1-6."},
                    "store_category": {"type": "string", "description": "e.g. 'pizza', 'mexican'."},
                    "hour": {"type": "integer", "description": "Hour of the day, 0-23, US Pacific time."},
                    "weekday": {"type": "string", "description": "'Monday' ... 'Sunday'."},
                    "total_items": {"type": ["integer", "null"]},
                    "subtotal": {"type": ["number", "null"], "description": "Order value in dollars."},
                    "total_onshift_dashers": {"type": ["number", "null"], "description": "Leave empty to use typical values."},
                    "total_busy_dashers": {"type": ["number", "null"], "description": "Leave empty to use typical values."},
                    "total_outstanding_orders": {"type": ["number", "null"], "description": "Leave empty to use typical values."},
                },
                "required": ["market_id", "store_category", "hour", "weekday"],
            },
        },
    },
]

TOOL_FUNCTIONS = {"run_sql": tools.run_sql, "predict_delivery_time": tools.predict_delivery_time}
_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in TOOLS}


def _clean_args(name: str, raw: str) -> dict:
    """Parse the tool arguments and convert numbers sent as text ("2") to the right type."""
    args = json.loads(raw or "{}")
    out = {}
    for key, value in args.items():
        kind = _SCHEMAS[name].get(key, {}).get("type")
        if isinstance(kind, list):              # e.g. ["number", "null"]
            kind = kind[0]
        if value in (None, ""):
            continue
        if kind == "integer":
            value = int(float(value))
        elif kind == "number":
            value = float(value)
        out[key] = value
    return out


def _call_model(messages: list):
    """Call the model; on a rate limit (free tiers), wait and try again."""
    extra = {"reasoning_effort": "low"} if PROVIDER == "groq" else {}   # think briefly: fewer tokens
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOLS, temperature=0,
                max_tokens=1500, extra_body=extra,
            )
        except RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"  (rate limit reached, waiting {wait} seconds)")
            time.sleep(wait)
    raise RuntimeError("Rate limit: try again later.")


def ask(question: str, verbose: bool = True) -> dict:
    """Answer one question. Returns the answer plus every tool call, for logging and evaluation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]
    steps = []

    for _ in range(MAX_STEPS):
        message = _call_model(messages).choices[0].message

        if not message.tool_calls:
            answer = message.content or ""
            break

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments}}
                           for c in message.tool_calls],
        })
        for call in message.tool_calls:
            name = call.function.name
            try:
                args = _clean_args(name, call.function.arguments)
                output = TOOL_FUNCTIONS[name](**args)
            except Exception as e:      # pass errors back so the model can correct itself
                args, output = call.function.arguments, f"Tool error: {e}"
            steps.append({"tool": name, "input": args, "output": output})
            if verbose:
                print(f"  [{name}] {json.dumps(args, default=str)[:150]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
    else:
        answer = "I could not finish this question within the step limit."

    # Some models show their reasoning between <think> tags; keep only the answer
    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    record = {"time": datetime.now().isoformat(timespec="seconds"), "provider": PROVIDER,
              "model": MODEL, "question": question, "answer": answer, "steps": steps}
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return record


if __name__ == "__main__":
    print(f"Delivery operations assistant ({PROVIDER}: {MODEL}). Type a question, or 'quit' to stop.\n")
    while (q := input("You: ").strip()).lower() not in {"quit", "exit", ""}:
        print(f"\nAssistant: {ask(q)['answer']}\n")