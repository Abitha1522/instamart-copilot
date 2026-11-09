
import sys
import json
import time
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd

# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import mock_llm
import d1_rag

DB_PATH = ROOT / "data" / "instamart.db"
SCENARIOS_PATH = ROOT / "data" / "raw" / "agent_scenarios.csv"

OUTPUT_DIR = ROOT / "outputs"
STATE_DIR = ROOT / "state"

LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"
RUNS_PATH = OUTPUT_DIR / "agent_runs.jsonl"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# TOOL 1: GET ORDER
# ---------------------------------------------------------

def get_order(order_id):
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        row = connection.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": float(row["order_amount"]),
            "delivery_minutes": (
                float(row["delivery_minutes"])
                if row["delivery_minutes"] is not None
                else None
            )
        }

    finally:
        connection.close()


# ---------------------------------------------------------
# TOOL 2: SEARCH POLICY
# ---------------------------------------------------------

def search_policy(query):
    return {
        "results": d1_rag.retrieve(query, k=3)
    }


# ---------------------------------------------------------
# TOOL 3: CALCULATE REFUND
# ---------------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):

    if delivery_minutes is None:
        return {"refund_inr": 0.0}

    if float(delivery_minutes) <= 45:
        return {"refund_inr": 0.0}

    return {
        "refund_inr": round(float(order_amount), 2)
    }


# ---------------------------------------------------------
# LEDGER WRITER
# ---------------------------------------------------------

def _write_ledger(tool_name, order_id, refund_inr):

    record = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return {
        "refund_inr": round(float(refund_inr), 2)
    }


# ---------------------------------------------------------
# TOOL 4: ISSUE REFUND
# ---------------------------------------------------------

def issue_refund(order_id, refund_inr):

    return _write_ledger(
        "issue_refund",
        order_id,
        refund_inr
    )


# ---------------------------------------------------------
# TOOL 5: ESCALATE TO HUMAN
# ---------------------------------------------------------

def escalate_to_human(order_id, refund_inr):

    return _write_ledger(
        "escalate_to_human",
        order_id,
        refund_inr
    )


# ---------------------------------------------------------
# TOOL SPECIFICATIONS
# ---------------------------------------------------------

TOOLS_SPEC = [

    {
        "name": "get_order",
        "description": "Retrieve order status, amount and delivery time using an order ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },

    {
        "name": "search_policy",
        "description": "Search refund and delivery policy documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },

    {
        "name": "calculate_refund",
        "description": "Calculate the late-delivery refund amount.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },

    {
        "name": "issue_refund",
        "description": "Record an eligible automatic refund in the refund ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },

    {
        "name": "escalate_to_human",
        "description": "Record a refund requiring human support lead approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


# ---------------------------------------------------------
# SAFE TOOL EXECUTION
# ---------------------------------------------------------

def execute_tool(name, arguments):

    if name not in TOOL_MAP:
        return {"error": "unknown tool"}

    if name in ("issue_refund", "escalate_to_human"):

        order = get_order(arguments["order_id"])

        if "error" in order:
            return {"error": "order not found"}

        if order["status"] != "completed":
            return {"error": "only completed orders can be refunded"}

        if (
            order["delivery_minutes"] is None
            or order["delivery_minutes"] <= 45
        ):
            return {"error": "order is not eligible for late-delivery refund"}

        expected = round(order["order_amount"], 2)
        requested = round(float(arguments["refund_inr"]), 2)

        if requested != expected:
            return {"error": "refund amount does not match eligible order amount"}

        if name == "issue_refund" and requested > 1000:
            return {"error": "refund above INR 1000 requires human approval"}

        if name == "escalate_to_human" and requested <= 1000:
            return {"error": "human escalation is only for refunds above INR 1000"}

    try:
        return TOOL_MAP[name](**arguments)

    except (TypeError, KeyError, ValueError) as error:
        return {"error": str(error)}


# ---------------------------------------------------------
# CHECKPOINT SAVING
# ---------------------------------------------------------

def save_checkpoint(state):

    checkpoint_path = STATE_DIR / f"{state['run_id']}.json"

    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, default=str)

    return checkpoint_path


# ---------------------------------------------------------
# AGENT CONTINUATION
# ---------------------------------------------------------

class SimulatedCrash(Exception):
    pass


def _continue_agent(
    state,
    max_iterations,
    token_limit,
    time_limit_seconds,
    crash_after_tools=None
):

    start_time = time.monotonic()

    messages = state["messages"]
    steps = state["steps"]
    tokens_used = state["tokens_used"]
    llm_calls = state["llm_calls"]

    def fallback(status, reason):

        return {
            "status": status,
            "final_answer": (
                f"{status}: {reason} "
                "Please hand this request to a human support agent."
            ),
            "steps": steps,
            "llm_calls": llm_calls,
            "tokens_used": tokens_used
        }

    while llm_calls < max_iterations:

        if time.monotonic() - start_time >= time_limit_seconds:
            return fallback(
                "FALLBACK_TIMEOUT",
                "The agent exceeded its time limit."
            )

        response = mock_llm.chat(
            messages,
            tools=TOOLS_SPEC
        )

        llm_calls += 1

        call_tokens = response.usage.total_tokens
        tokens_used += call_tokens

        state["llm_calls"] = llm_calls
        state["tokens_used"] = tokens_used

        if tokens_used > token_limit:
            return fallback(
                "FALLBACK_TOKEN_LIMIT",
                "The agent exceeded its token budget."
            )

        message = response.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:

            final_answer = (
                message.content or "Unable to process the request."
            )

            status = (
                final_answer.split(":", 1)[0].strip()
                if ":" in final_answer
                else "UNKNOWN"
            )

            state["messages"] = messages
            state["steps"] = steps
            state["completed"] = True
            save_checkpoint(state)

            return {
                "status": status,
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        assistant_tool_calls = []

        for tool_call in tool_calls:

            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            assistant_tool_calls.append({
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": arguments
            })

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls
        })

        for tool_call in tool_calls:

            name = tool_call.name
            arguments = tool_call.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            observation = execute_tool(name, arguments)

            step = {
                "tool": name,
                "arguments": arguments,
                "observation": observation
            }

            steps.append(step)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": json.dumps(observation, default=str)
            })

            state["messages"] = messages
            state["steps"] = steps
            state["step_number"] = len(steps)
            state["tools_executed"] = [
                item["tool"] for item in steps
            ]

            # Save immediately after every tool observation.
            save_checkpoint(state)

            # Used only for the resumability demonstration.
            if (
                crash_after_tools is not None
                and len(steps) == crash_after_tools
            ):
                raise SimulatedCrash(
                    "Simulated crash after checkpoint was saved."
                )

    return fallback(
        "FALLBACK_MAX_ITERATIONS",
        "The agent reached its maximum iteration limit."
    )


# ---------------------------------------------------------
# RUN AGENT
# ---------------------------------------------------------

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None
):

    if run_id is None:
        run_id = str(uuid.uuid4())

    messages = [
        {
            "role": "system",
            "content": (
                "AGENT: You are a refund support agent. "
                "Use tools to retrieve orders, search policies, "
                "calculate refunds and process eligible refunds. "
                "Only completed orders qualify. "
                "Delivery must exceed 45 minutes. "
                "Refunds above INR 1000 require human approval. "
                "Never invent order details or refund amounts."
            )
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    state = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "steps": [],
        "tokens_used": 0,
        "llm_calls": 0,
        "step_number": 0,
        "tools_executed": [],
        "completed": False,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds
    }

    save_checkpoint(state)

    return _continue_agent(
        state,
        max_iterations,
        token_limit,
        time_limit_seconds,
        crash_after_tools
    )


# ---------------------------------------------------------
# RESUME AGENT
# ---------------------------------------------------------

def resume_agent(run_id):

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        state = json.load(file)

    if state.get("completed"):
        raise ValueError("This agent run is already completed.")

    return _continue_agent(
        state,
        state["max_iterations"],
        state["token_limit"],
        state["time_limit_seconds"]
    )


# ---------------------------------------------------------
# RUN ALL SEVEN SCENARIOS
# ---------------------------------------------------------

def run_scenarios():

    scenarios = pd.read_csv(
        SCENARIOS_PATH,
        keep_default_na=False
    )

    results = []

    with open(RUNS_PATH, "w", encoding="utf-8") as output:

        for _, scenario in scenarios.iterrows():

            scenario_id = str(scenario["scenario_id"])
            message = str(scenario["message"])

            print("\n" + "=" * 55)
            print("SCENARIO:", scenario_id)
            print("=" * 55)

            mock_llm.reset()

            result = run_agent(
                message,
                scenario_id
            )

            record = {
                "scenario_id": scenario_id,
                "status": result["status"],
                "final_answer": result["final_answer"],
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"]
            }

            output.write(json.dumps(record, default=str) + "\n")

            results.append(record)

            print("STATUS:", result["status"])
            print("ANSWER:", result["final_answer"])
            print("MODEL CALLS:", result["llm_calls"])
            print("TOKENS:", result["tokens_used"])
            print(
                "TOOLS:",
                [step["tool"] for step in result["steps"]]
            )

    return results


# ---------------------------------------------------------
# RESUMABILITY DEMONSTRATION
# ---------------------------------------------------------

def demonstrate_resume():

    print("\nRESUMABILITY DEMONSTRATION")

    run_id = "resume_S1_demo"

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    mock_llm.reset()

    message = (
        "My order ORD-100048 reached me very late. "
        "I would like a refund."
    )

    try:

        run_agent(
            message,
            "S1_RESUME",
            run_id=run_id,
            crash_after_tools=2
        )

    except SimulatedCrash as error:
        print("SIMULATED CRASH:", error)

    with open(checkpoint_path, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)

    print(
        "TOOLS BEFORE RESUME:",
        checkpoint["tools_executed"]
    )

    resumed_result = resume_agent(run_id)

    print("RESUMED STATUS:", resumed_result["status"])
    print("RESUMED ANSWER:", resumed_result["final_answer"])
    print("TOTAL MODEL CALLS:", resumed_result["llm_calls"])
    print("TOTAL TOKENS:", resumed_result["tokens_used"])

    print(
        "ALL TOOLS:",
        [step["tool"] for step in resumed_result["steps"]]
    )

    return resumed_result


# ---------------------------------------------------------
# FALLBACK TESTS
# ---------------------------------------------------------

def test_fallbacks():

    print("\nFALLBACK TESTS")

    message = (
        "My order ORD-100048 reached me very late. "
        "I would like a refund."
    )

    # Token limit test
    mock_llm.reset()

    token_result = run_agent(
        message,
        "TOKEN_TEST",
        token_limit=200
    )

    print(
        "TOKEN FALLBACK:",
        token_result["status"]
    )

    # Timeout test
    mock_llm.reset()

    timeout_result = run_agent(
        message,
        "TIMEOUT_TEST",
        time_limit_seconds=0
    )

    print(
        "TIMEOUT FALLBACK:",
        timeout_result["status"]
    )

    # Maximum iterations test
    mock_llm.reset()

    loop_message = "[LOOP] Refund order ORD-100048."

    iteration_result = run_agent(
        loop_message,
        "ITERATION_TEST",
        max_iterations=6
    )

    print(
        "ITERATION FALLBACK:",
        iteration_result["status"]
    )

    return {
        "token": token_result,
        "timeout": timeout_result,
        "iteration": iteration_result
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\nPART D2 - REFUND AGENT")

    scenario_results = run_scenarios()

    print("\nSCENARIO SUMMARY")

    for result in scenario_results:
        print(
            result["scenario_id"],
            result["status"],
            result["llm_calls"],
            result["tokens_used"]
        )

    demonstrate_resume()

    test_fallbacks()

    print("\nPART D2 EXECUTION FINISHED")
