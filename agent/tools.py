"""Tools the delivery agent can use: query the delivery data and predict delivery time."""
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "deliveries_clean.parquet"
MODEL_FILE = ROOT / "agent" / "model.joblib"
MAX_ROWS = 50            # most rows a query may return to the model
TYPICAL_ERROR_MIN = 11   # average error of the model on the test week (10.7 minutes)
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# --- 1. Load the data into one in-memory table: `deliveries` -------------------
# Only local (US Pacific) times are kept, so the model cannot mix them up with UTC.
_con = duckdb.connect()
_con.execute(f"""
    CREATE TABLE deliveries AS
    SELECT
        market_id,
        created_local,
        CAST(created_local AS DATE)   AS order_date,
        hour(created_local)           AS hour,
        dayname(created_local)        AS weekday,
        delivery_min,
        store_id,
        store_primary_category,
        order_protocol,
        total_items, num_distinct_items,
        subtotal, min_item_price, max_item_price,
        total_onshift_dashers, total_busy_dashers, total_outstanding_orders,
        total_outstanding_orders / NULLIF(total_onshift_dashers, 0) AS orders_per_dasher,
        estimated_order_place_duration,
        estimated_store_to_consumer_driving_duration
    FROM read_parquet('{DATA_FILE}')
    WHERE NOT long_delivery
""")
# Safety: from here on, queries cannot read or write any file on the computer
_con.execute("SET enable_external_access = false")
_con.execute("SET lock_configuration = true")


def _q(sql: str) -> pd.DataFrame:
    return _con.execute(sql).fetchdf()


# --- 2. Facts about the data, so the model knows what it is working with -------
_facts = _q("""
    SELECT count(*) AS orders, min(order_date) AS first_day, max(order_date) AS last_day,
           count(DISTINCT order_date) AS days, count(DISTINCT store_id) AS stores
    FROM deliveries
""").iloc[0]
_days_per_weekday = (
    _q("SELECT weekday, count(DISTINCT order_date) AS n FROM deliveries GROUP BY weekday")
    .set_index("weekday")["n"].reindex(WEEKDAYS).fillna(0).astype(int)
)
_categories = sorted(_q("SELECT DISTINCT store_primary_category AS c FROM deliveries")["c"])
_top_categories = _q("""
    SELECT store_primary_category AS c FROM deliveries
    GROUP BY 1 ORDER BY count(*) DESC LIMIT 15
""")["c"].tolist()

TABLE_DESCRIPTION = f"""
Table `deliveries`: one row per DoorDash order. {_facts.orders:,} orders from {_facts.stores:,} stores,
placed between {_facts.first_day:%d %B %Y} and {_facts.last_day:%d %B %Y} ({_facts.days} days).
Deliveries recorded as taking more than 5 hours are excluded as likely data errors.
Number of days per weekday in the data: {", ".join(f"{d} {n}" for d, n in _days_per_weekday.items())}.

Columns:
- market_id: anonymised region, 1 to 6 (all on the US West Coast; city names are not known)
- created_local: timestamp when the order was placed, in US Pacific time
- order_date (date), hour (0-23), weekday ('Monday' ... 'Sunday'): derived from created_local
- delivery_min: minutes from order to delivery (the outcome of interest)
- store_id; store_primary_category: cuisine, {len(_categories)} values; most common:
  {", ".join(_top_categories)}. 'unknown' means the category is missing.
- order_protocol: id of how the store receives orders; the meaning of each id is not known
- total_items, num_distinct_items: size of the order
- subtotal, min_item_price, max_item_price: in US dollars
- total_onshift_dashers, total_busy_dashers, total_outstanding_orders: couriers on shift, couriers
  busy, and open orders in the market when the order was placed (missing for about 8% of orders;
  busy can exceed on-shift, as couriers may still finish a delivery after their shift)
- orders_per_dasher: total_outstanding_orders / total_onshift_dashers (courier pressure)
- estimated_order_place_duration, estimated_store_to_consumer_driving_duration: DoorDash's own
  estimates, in seconds

Not in the data: weather, distance or addresses, customer ratings, tips, restaurant preparation time,
city names, and anything outside the period above.
"""


# --- 3. Tool: run a SQL query --------------------------------------------------
def run_sql(query: str) -> str:
    """Run one read-only SQL query on `deliveries`; return at most MAX_ROWS rows as CSV."""
    q = query.strip().rstrip(";")
    if not q.lower().startswith(("select", "with")) or ";" in q:
        return "Error: only a single SELECT query is allowed."
    try:
        result = _q(q)
    except Exception as e:  # return the error so the model can fix its query
        return f"SQL error: {e}"
    note = f"\n(showing the first {MAX_ROWS} of {len(result):,} rows)" if len(result) > MAX_ROWS else ""
    return result.head(MAX_ROWS).round(2).to_csv(index=False) + note


# --- 4. Tool: predict delivery time ---------------------------------------------
_bundle = joblib.load(MODEL_FILE)
_model, _features = _bundle["model"], _bundle["features"]

_ORDER_COLS = ["total_items", "num_distinct_items", "subtotal", "min_item_price", "max_item_price",
               "estimated_order_place_duration", "estimated_store_to_consumer_driving_duration"]
_MARKET_COLS = ["total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders"]


def _typical_values(market_id: int, weekday: str, hour: int) -> dict:
    """Median order and courier values; courier values for this market, weekday and hour."""
    medians = ", ".join(f"median({c}) AS {c}" for c in _ORDER_COLS)
    row = _q(f"SELECT {medians}, mode(order_protocol) AS order_protocol FROM deliveries").iloc[0].to_dict()

    market = ", ".join(f"median({c}) AS {c}" for c in _MARKET_COLS)
    for where in (f"market_id = {market_id} AND weekday = '{weekday}' AND hour = {hour}",
                  f"market_id = {market_id} AND hour = {hour}",
                  "TRUE"):                       # fall back if there are no orders at that moment
        local = _q(f"SELECT {market}, count(*) AS n FROM deliveries WHERE {where}").iloc[0]
        if local["n"] > 0 and not pd.isna(local["total_onshift_dashers"]):
            row.update({c: local[c] for c in _MARKET_COLS})
            break
    return row


def predict_delivery_time(market_id: int, store_category: str, hour: int, weekday: str,
                          total_items: int | None = None, subtotal: float | None = None,
                          total_onshift_dashers: float | None = None,
                          total_busy_dashers: float | None = None,
                          total_outstanding_orders: float | None = None) -> str:
    """Predict the delivery time of one order. Values that are not given are filled in with
    typical values; courier numbers come from the same market, weekday and hour."""
    # Check the inputs
    weekday = str(weekday).strip().capitalize()
    store_category = str(store_category).strip().lower()
    if weekday not in WEEKDAYS:
        return f"Error: weekday must be one of {WEEKDAYS}."
    if market_id not in range(1, 7):
        return "Error: market_id must be between 1 and 6."
    if hour not in range(24):
        return "Error: hour must be between 0 and 23."
    if store_category not in _categories:
        return f"Error: unknown store category '{store_category}'. Valid values include: {', '.join(_top_categories)}."

    # Build one row with the same features the model was trained on
    row = _typical_values(market_id, weekday, hour)
    given = {"total_items": total_items, "subtotal": subtotal,
             "total_onshift_dashers": total_onshift_dashers, "total_busy_dashers": total_busy_dashers,
             "total_outstanding_orders": total_outstanding_orders}
    row.update({k: v for k, v in given.items() if v is not None})
    row.update(market_id=market_id, hour=hour, weekday=WEEKDAYS.index(weekday))
    onshift = row["total_onshift_dashers"] or np.nan
    row["orders_per_dasher"] = row["total_outstanding_orders"] / onshift
    row["busy_share"] = row["total_busy_dashers"] / onshift

    X = pd.DataFrame([row])
    X["store_primary_category"] = pd.Categorical([store_category], categories=_categories)
    minutes = float(_model.predict(X[_features])[0])

        # Explain the inputs in plain language (read by the model and shown in the app)
    def note(key):
        if given[key] is not None:
            return "given"
        if key in _MARKET_COLS:
            return f"typical for market {market_id} on {weekday}s at {hour}:00"
        return "typical order"
    pressure = row["orders_per_dasher"]
    lines = [
        f"Predicted delivery time: {minutes:.0f} minutes (typical error about {TYPICAL_ERROR_MIN} minutes).",
        "",
        "Inputs used:",
        f"- Order: {store_category}, market {market_id}, {weekday} at {hour}:00",
        f"- Items: {row['total_items']:.0f} ({note('total_items')})",
        f"- Order value: ${row['subtotal']:.2f} ({note('subtotal')})",
        f"- Couriers on shift: {row['total_onshift_dashers']:.0f} ({note('total_onshift_dashers')})",
        f"- Couriers busy: {row['total_busy_dashers']:.0f} ({note('total_busy_dashers')})",
        f"- Open orders: {row['total_outstanding_orders']:.0f} ({note('total_outstanding_orders')})",
        f"- Courier pressure: {pressure:.1f} open orders per courier on shift" if pd.notna(pressure) else
        "- Courier pressure: unknown",
    ]
    return "\n".join(lines)