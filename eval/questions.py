"""Evaluation set: questions with a known correct answer.

Each question has a check:
- "number":   the answer must contain a number close to the reference value
               (computed from the data with `reference_sql`, or from the model with `reference_predict`)
- "contains": the answer must contain one of the given phrases (lower case)
- "declines": the agent must say the data cannot answer it (or that it only helps with delivery data)
"""

DECLINE_PHRASES = [
    "not in the data", "not included", "does not include", "doesn't include", "does not contain",
    "doesn't contain", "no information", "not available", "not recorded", "no data on",
    "isn't in the data", "is not in the data", "only help", "only answer", "can't help", "cannot help",
    "outside the period", "not covered", "doesn't cover", "does not cover", "no weather",
    "no rating", "no customer rating",
]

QUESTIONS = [
    # --- Looking things up -----------------------------------------------------
    dict(id=1, category="Lookup", question="Which day of the week has the slowest deliveries?",
         check="contains", expected=["friday"]),
    dict(id=2, category="Lookup", question="Which day of the week has the fastest deliveries?",
         check="contains", expected=["tuesday"]),
    dict(id=3, category="Lookup",
         question="Among hours with at least 1,000 orders, at what hour are deliveries slowest?",
         check="contains", expected=["6 pm", "6pm", "6 p.m", "18:00", "hour 18"]),
    dict(id=4, category="Lookup", question="What is the median delivery time across all orders?",
         check="number", reference_sql="SELECT median(delivery_min) FROM deliveries", tolerance=1),
    dict(id=5, category="Lookup", question="How many orders are there in market 1?",
         check="number", reference_sql="SELECT count(*) FROM deliveries WHERE market_id = 1",
         tolerance=0.01, relative=True),
    dict(id=6, category="Lookup", question="What is the median delivery time for Japanese restaurants?",
         check="number",
         reference_sql="SELECT median(delivery_min) FROM deliveries WHERE store_primary_category = 'japanese'",
         tolerance=1),
    dict(id=7, category="Lookup", question="Which market has the slowest median delivery time?",
         check="contains", expected=["market 1", "market_id 1", "market id 1"]),
    dict(id=8, category="Lookup",
         question="On average, how many orders are placed on a Friday between 6 and 7 pm?",
         check="number",
         reference_sql="""SELECT avg(n) FROM (SELECT order_date, count(*) AS n FROM deliveries
                          WHERE weekday = 'Friday' AND hour = 18 GROUP BY order_date)""",
         tolerance=0.03, relative=True),

    # --- Relationships ---------------------------------------------------------
    dict(id=9, category="Relationship",
         question="What is the median delivery time when there are 2 or more open orders per courier on shift?",
         check="number",
         reference_sql="SELECT median(delivery_min) FROM deliveries WHERE orders_per_dasher >= 2",
         tolerance=1.5),
    dict(id=10, category="Relationship",
         question="What is the median delivery time for orders with 9 or more items?",
         check="number",
         reference_sql="SELECT median(delivery_min) FROM deliveries WHERE total_items >= 9",
         tolerance=1.5),

    # --- Predictions -----------------------------------------------------------
    dict(id=11, category="Prediction",
         question="How long would a pizza order in market 2 take on Friday at 6 pm?",
         check="number", reference_predict=dict(market_id=2, store_category="pizza", hour=18, weekday="Friday"),
         tolerance=2),
    dict(id=12, category="Prediction",
         question="Predict the delivery time for a Mexican order of 3 items in market 4 on a Tuesday at 1 pm.",
         check="number",
         reference_predict=dict(market_id=4, store_category="mexican", hour=13, weekday="Tuesday", total_items=3),
         tolerance=2),

    # --- Questions the data cannot answer ---------------------------------------
    dict(id=13, category="Cannot answer", question="Does rain slow down deliveries?", check="declines"),
    dict(id=14, category="Cannot answer", question="What is the average customer rating for pizza stores?",
         check="declines"),
    dict(id=15, category="Cannot answer", question="How many deliveries were there in March 2015?",
         check="declines"),

    # --- Off-topic ----------------------------------------------------------------
    dict(id=16, category="Off-topic", question="Write me a poem about pizza.", check="declines"),
]