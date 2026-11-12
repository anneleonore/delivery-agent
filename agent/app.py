"""Chat interface for the delivery assistant. Start with: streamlit run agent/app.py"""
from io import StringIO

import pandas as pd
import sqlparse
import streamlit as st

from agent import MODEL, PROVIDER, ask

MAX_QUESTIONS = 5          # per visitor, to protect the free API limit
ICON_NAME = "smart_toy"    # a Google Material icon name (fonts.google.com/icons)
AGENT_ICON = f":material/{ICON_NAME}:"   # used next to every answer

st.set_page_config(page_title="Delivery Ops Agent", page_icon="🛵", layout="centered")

TOOL_LABELS = {"run_sql": "Database query", "predict_delivery_time": "Prediction model"}


def show_step(step: dict) -> None:
    """Show one tool call in a readable way: formatted SQL and a table, or the prediction text."""
    st.markdown(f"**{TOOL_LABELS.get(step['tool'], step['tool'])}**")
    inp, out = step["input"], step["output"]
    if step["tool"] == "run_sql" and isinstance(inp, dict):
        st.code(sqlparse.format(inp["query"], reindent=True, keyword_case="upper"), language="sql")
        if out.startswith(("SQL error", "Error")):
            st.caption("This query failed; the agent read the error and tried again below.")
        else:
            try:
                table = pd.read_csv(StringIO(out.split("\n(showing")[0]))
                st.dataframe(table, hide_index=True, use_container_width=True)
            except Exception:          # unexpected output, show it as text
                st.text(out[:500])
    else:
        st.markdown(out.replace("$", "\\$"))   # escape $ so Streamlit doesn't read it as maths

# --- Styling --------------------------------------------------------------------
st.markdown("""
<style>
/* The card around everything */
.st-key-agent_card {
    border-radius: 20px !important;
    padding: 1.5rem 1.75rem !important;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
}
/* Icon above the title */
.agent-icon {
    width: 64px; height: 64px; margin: 0.3rem auto 0.6rem auto;
    border-radius: 50%; background-color: #F5F5F5;
    display: flex; align-items: center; justify-content: center; font-size: 2rem; color: #C62828;
}
.agent-icon .material-icon { font-family: "Material Symbols Rounded"; font-size: 36px; line-height: 1;
                             font-weight: normal; font-style: normal; }
/* Title and subtitle, centred */
.agent-title { text-align: center; font-size: 2.2rem; font-weight: 700; margin: 0.2rem 0 0.4rem 0; }
.agent-subtitle { text-align: center; font-size: 1.05rem; color: #555; line-height: 1.5; margin-bottom: 1.2rem; }
.agent-subtitle .small { font-size: 0.9rem; color: #888; }
/* The question box: white, light grey on hover and while typing */
div[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #DDDDDD !important;
    border-radius: 14px !important;
}
div[data-testid="stChatInput"]:hover,
div[data-testid="stChatInput"]:focus-within {
    background-color: #F5F5F5 !important;
    border-color: #BBBBBB !important;
}
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    border-color: transparent !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"] textarea { min-height: 70px !important; font-size: 1.05rem !important; }
/* The robot icon next to each answer: red, like the icon at the top */
[class*="st-key-agent_msg_"] [data-testid="stChatMessageAvatarCustom"] { color: #C62828 !important; }
[class*="st-key-agent_msg_"] [data-testid="stChatMessageAvatarCustom"] * { color: #C62828 !important; }
/* White space above the question box */
.st-key-agent_card div[data-testid="stChatInput"] { margin-top: 1.75rem; }
/* Heading above the example questions */
.examples-label { font-size: 0.85rem; font-weight: 600; color: #888; text-transform: uppercase;
                  letter-spacing: 0.05em; margin: 0.4rem 0 0.2rem 0; }
/* Example question buttons: light red on hover */
.st-key-agent_card div[data-testid="stButton"] button:hover {
    background-color: rgba(255, 48, 8, 0.08) !important;
    border-color: rgba(255, 48, 8, 0.45) !important;
    color: #C62828 !important;
}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

# --- Everything sits inside one rounded card ----------------------------------------
with st.container(border=True, key="agent_card"):
    st.markdown(f'<div class="agent-icon"><span class="material-icon">{ICON_NAME}</span></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="agent-title">Delivery Ops Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="agent-subtitle">Ask questions about 196,000 DoorDash deliveries from January–February 2015.<br>'
        f'<span class="small">Powered by {MODEL} via {PROVIDER.capitalize()}</span></div>',
        unsafe_allow_html=True,
    )

    # Example questions as buttons: short label -> full question
    st.markdown('<div class="examples-label">Try an example question</div>', unsafe_allow_html=True)
    examples = {
        "Slowest day of the week": "Which day of the week has the slowest deliveries?",
        "Pizza on Friday at 6 pm": "How long would a pizza order in market 2 take on Friday at 6 pm?",
        "Busy couriers": "Do deliveries take longer when couriers are busy?",
    }
    clicked = None
    for col, (label, example) in zip(st.columns(len(examples)), examples.items()):
        if col.button(label, use_container_width=True):
            clicked = example

    # The conversation so far
    for i, turn in enumerate(st.session_state.history):
        with st.chat_message("user", avatar=":material/person:"):
            st.write(turn["question"])
        with st.container(key=f"agent_msg_{i}"):
            with st.chat_message("assistant", avatar=AGENT_ICON):
                st.write(turn["answer"])
                if turn["steps"]:
                    with st.expander("How I got this answer"):
                        for step in turn["steps"]:
                            show_step(step)

    # New question (with a limit per visitor)
    asked = len(st.session_state.history)
    if asked >= MAX_QUESTIONS:
        st.info(f"You've reached the limit of {MAX_QUESTIONS} questions for this demo. "
                "Refresh the page to start again.")
    else:
        question = st.chat_input(
            f"Ask a question about the deliveries ({MAX_QUESTIONS - asked} left)") or clicked
        if question:
            with st.spinner("Thinking..."):
                result = ask(question, verbose=False)
            st.session_state.history.append(result)
            st.rerun()