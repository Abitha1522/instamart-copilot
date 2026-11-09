
from pathlib import Path
import sys
import re
import json
import random
import time

import pandas as pd

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mock_llm

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"

TICKETS_FILE = RAW_DIR / "tickets.csv"
SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

MAX_RETRIES = 3
BASE_DELAY = 0.05

TOKEN_BUDGET = 6000
FINAL_TOKEN_BUDGET = 100000

random.seed(42)

# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------
ZERO_SHOT_PROMPT = """
CLASSIFY customer support tickets.

Classify each customer message into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only in this format:
{"category":"billing","reason":"short explanation"}

Do not return any category outside the allowed four.
"""

FEW_SHOT_PROMPT = """
CLASSIFY customer support tickets.

Classify each customer message into exactly one category:
billing, delivery, technical, or fraud.

Return JSON only in this format:
{"category":"billing","reason":"short explanation"}

Do not return any category outside the allowed four.

EXAMPLES:

Message: I was charged twice for my order.
Answer: {"category":"billing","reason":"duplicate charge"}

Message: My parcel was stolen and someone used my account.
Answer: {"category":"fraud","reason":"unauthorized activity"}

Message: The app crashes when I try to log in.
Answer: {"category":"technical","reason":"application error"}

Message: My delivery has not arrived yet.
Answer: {"category":"delivery","reason":"late delivery"}
"""

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
def load_data():
    tickets = pd.read_csv(
        TICKETS_FILE,
        keep_default_na=False
    )

    segments = pd.read_csv(SEGMENTS_FILE)

    required_ticket_columns = {
        "ticket_id",
        "customer_id",
        "order_id",
        "text",
        "true_category"
    }

    required_segment_columns = {
        "customer_id",
        "segment_name",
        "is_premium"
    }

    assert required_ticket_columns.issubset(tickets.columns)
    assert required_segment_columns.issubset(segments.columns)

    segments["is_premium"] = (
        segments["is_premium"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )

    segment_lookup = segments.set_index("customer_id")[
        ["segment_name", "is_premium"]
    ].to_dict("index")

    print("Tickets loaded:", tickets.shape)
    print("Customer segments loaded:", segments.shape)

    return tickets, segment_lookup


# ---------------------------------------------------------
# C1 AND C2: INPUT GUARDRAILS AND PII REDACTION
# ---------------------------------------------------------
def redact_pii(text):
    text = str(text)

    text, email_count = re.subn(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        "[EMAIL]",
        text
    )

    text, phone_count = re.subn(
        r"\b[6-9]\d{9}\b",
        "[PHONE]",
        text
    )

    text, order_count = re.subn(
        r"ORD-\d{6}",
        "[ORDER_ID]",
        text
    )

    return text, {
        "email": email_count,
        "phone": phone_count,
        "order_id": order_count
    }


def prepare_ticket(text):
    if not str(text).strip():
        return None, "skipped_empty", {}

    if mock_llm.est_tokens(text) > 400:
        return None, "flagged_long", {}

    redacted_text, counts = redact_pii(text)

    # Verify that no PII remains.
    assert not re.search(
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        redacted_text
    )

    assert not re.search(
        r"\b[6-9]\d{9}\b",
        redacted_text
    )

    assert not re.search(
        r"ORD-\d{6}",
        redacted_text
    )

    return redacted_text, "ready", counts


# ---------------------------------------------------------
# C4: MODEL RETRIES
# ---------------------------------------------------------
def call_with_retry(messages, ticket_id):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)

        except mock_llm.TransientLLMError:
            if attempt == MAX_RETRIES:
                print(
                    f"Ticket {ticket_id}: retries exhausted"
                )
                return None

            wait_time = (
                BASE_DELAY
                * (2 ** attempt)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"Retry: ticket={ticket_id}, "
                f"retry={attempt + 1}, "
                f"wait={wait_time:.4f}s"
            )

            time.sleep(wait_time)

    return None


# ---------------------------------------------------------
# C5: VALIDATE MODEL RESPONSE
# ---------------------------------------------------------
def validate_response(response_text):
    try:
        parsed = json.loads(response_text)

        if (
            isinstance(parsed, dict)
            and parsed.get("category") in ALLOWED_CATEGORIES
        ):
            return parsed["category"], "classified_json"

        # Valid JSON with an invalid category must not
        # fall back to keyword classification.
        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        words = re.findall(
            r"[a-z]+",
            str(response_text).lower()
        )

        matches = ALLOWED_CATEGORIES.intersection(words)

        if len(matches) == 1:
            return matches.pop(), "classified_keyword"

        return "", "flagged_invalid_output"


# ---------------------------------------------------------
# C6 AND C7: RUN THE CLASSIFICATION PIPELINE
# ---------------------------------------------------------
def run_pipeline(tickets, prompt_type, token_budget=None):
    mock_llm.reset()

    if prompt_type == "zero_shot":
        system_prompt = ZERO_SHOT_PROMPT
    else:
        system_prompt = FEW_SHOT_PROMPT

    running_tokens = 0
    results = []

    pii_totals = {
        "email": 0,
        "phone": 0,
        "order_id": 0
    }

    for _, ticket in tickets.iterrows():
        ticket_id = ticket["ticket_id"]
        customer_id = ticket["customer_id"]
        original_text = ticket["text"]
        true_category = ticket["true_category"]

        redacted_text, status, pii_counts = prepare_ticket(
            original_text
        )

        for key, value in pii_counts.items():
            pii_totals[key] += value

        category = ""
        tokens_used = 0

        if status == "ready":

            if (
                token_budget is not None
                and running_tokens >= token_budget
            ):
                status = "budget_exhausted"

            else:
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": redacted_text
                    }
                ]

                response = call_with_retry(
                    messages,
                    ticket_id
                )

                if response is None:
                    status = "flagged_invalid_output"

                else:
                    tokens_used = response.usage.total_tokens
                    running_tokens += tokens_used

                    response_text = (
                        response.choices[0].message.content
                    )

                    category, status = validate_response(
                        response_text
                    )

        results.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "true_category": true_category,
            "tokens_used": tokens_used
        })

    result_df = pd.DataFrame(results)

    classified = result_df[
        result_df["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]

    accuracy = (
        (classified["category"] == classified["true_category"]).mean()
        if len(classified) > 0
        else 0
    )

    total_tokens = int(result_df["tokens_used"].sum())

    print("\n" + "=" * 55)
    print(prompt_type.upper(), "RESULTS")
    print("=" * 55)

    print("\nStatus counts:")
    print(result_df["status"].value_counts())

    print("\nTickets classified:", len(classified))
    print("Classification accuracy:", round(accuracy * 100, 2), "%")
    print("Total tokens:", total_tokens)

    print(
        "Tokens per classified ticket:",
        round(total_tokens / len(classified), 2)
        if len(classified) else 0
    )

    print("\nPII redaction counts:", pii_totals)

    if prompt_type == "zero_shot":
        print("\nZERO-SHOT CONFUSION TABLE:")
        print(
            pd.crosstab(
                classified["true_category"],
                classified["category"],
                rownames=["Actual"],
                colnames=["Predicted"]
            )
        )

    return result_df, {
        "classified": len(classified),
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "pii_totals": pii_totals
    }


# ---------------------------------------------------------
# C7: FINAL CUSTOMER ROUTING
# ---------------------------------------------------------
def apply_routing(results, segment_lookup):
    final_rows = []

    for _, row in results.iterrows():
        customer_id = row["customer_id"]

        segment_info = segment_lookup.get(
            customer_id,
            {}
        )

        segment_name = segment_info.get(
            "segment_name",
            ""
        )

        is_premium = segment_info.get(
            "is_premium",
            False
        )

        status = row["status"]
        category = row["category"]

        if status in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted"
        }:
            route = "human_review"
            priority = "NORMAL"

        elif category == "fraud":
            route = "security_team"
            priority = "HIGH"

        elif is_premium:
            route = "premium_desk"
            priority = "HIGH"

        else:
            route = "support_queue"
            priority = "NORMAL"

        final_rows.append({
            "ticket_id": row["ticket_id"],
            "customer_id": customer_id,
            "status": status,
            "category": category,
            "segment_name": segment_name,
            "priority": priority,
            "route": route,
            "tokens_used": row["tokens_used"]
        })

    return pd.DataFrame(final_rows)


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    tickets, segment_lookup = load_data()

    print("\nC1-C2: Guardrails and PII redaction")
    print("C3-C6: Zero-shot and few-shot comparison")

    # Full zero-shot evaluation
    zero_results, zero_metrics = run_pipeline(
        tickets,
        "zero_shot"
    )

    # Full few-shot evaluation
    few_results, few_metrics = run_pipeline(
        tickets,
        "few_shot"
    )

    # Token budget demonstration using zero-shot
    print("\nC7: TOKEN BUDGET DEMONSTRATION")

    budget_results, budget_metrics = run_pipeline(
        tickets,
        "zero_shot",
        token_budget=TOKEN_BUDGET
    )

    budget_file = OUTPUT_DIR / "triage_budget_demo.csv"
    budget_results.to_csv(
        budget_file,
        index=False
    )

    print("\nBudget demo saved:", budget_file)
    print(
        "Budget exhausted tickets:",
        (budget_results["status"] == "budget_exhausted").sum()
    )

    # Final few-shot run with sufficient token budget
    print("\nFINAL FEW-SHOT TRIAGE RUN")

    final_results, final_metrics = run_pipeline(
        tickets,
        "few_shot",
        token_budget=FINAL_TOKEN_BUDGET
    )

    final_output = apply_routing(
        final_results,
        segment_lookup
    )

    final_file = OUTPUT_DIR / "triage_results.csv"

    final_output.to_csv(
        final_file,
        index=False
    )

    print("\nFINAL ROUTING SUMMARY")
    print(final_output["route"].value_counts())

    print("\nFINAL STATUS SUMMARY")
    print(final_output["status"].value_counts())

    print("\nFinal output shape:", final_output.shape)
    print("Final output columns:", final_output.columns.tolist())
    print("Saved:", final_file)

    # Required checkpoint assertions
    assert len(tickets) == 164
    assert len(final_output) == 164
    assert len(budget_results) == 164

    assert (budget_results["status"] == "budget_exhausted").sum() > 0

    assert list(final_output.columns) == [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]

    assert zero_metrics["classified"] == 147
    assert few_metrics["classified"] == 147

    assert round(zero_metrics["accuracy"] * 100, 1) == 85.7
    assert round(few_metrics["accuracy"] * 100, 1) == 100.0

    assert (final_output["status"] == "classified_json").sum() == 135
    assert (final_output["status"] == "classified_keyword").sum() == 12
    assert (final_output["status"] == "flagged_invalid_output").sum() == 9

    print("\n" + "=" * 55)
    print("PART C CHECKPOINTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
