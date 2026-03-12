"""
evaluate_tool_calling.py
========================
Evaluates how reliably your HuggingFace LLM performs tool / function calling.

Usage:
    python evaluate_tool_calling.py

The script is self-contained — no Django setup required.
Edit HF_API_KEY and HF_MODEL_ID below to switch models at any time.
"""

# ── Configuration ─────────────────────────────────────────────────────────────
# Change these two variables to target a different model or key.
# The script also accepts env-vars: HF_API_KEY, HF_MODEL_ID
HF_API_KEY  = "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"  # HuggingFace API key
HF_MODEL_ID = "Qwen/Qwen3.5-9B"                         # model with tool-calling support

REPORT_FILE = "tool_call_eval_report.json"  # output path for the JSON report

# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

# Workaround: Python 3.12 importlib.metadata + older importlib_metadata venv
# conflict causes a WindowsPath crash. Patching packages_distributions to be
# fault-tolerant prevents the crash during langchain/transformers import.
import importlib.metadata as _ilm
_orig_pkgs_dist = _ilm.packages_distributions
def _safe_pkgs_dist():
    try:
        return _orig_pkgs_dist()
    except Exception:
        return {}
_ilm.packages_distributions = _safe_pkgs_dist

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TOOL DEFINITIONS
#     Add new tools here by writing a function decorated with @tool.
#     The docstring becomes the tool description sent to the model.
# ══════════════════════════════════════════════════════════════════════════════

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: Name of the city to check weather for.
    """
    # Stub response — replace with a real weather API call if needed
    return f"Weather in {city}: Sunny, 28°C, humidity 55%"


@tool
def get_user_profile(user_id: str) -> str:
    """Retrieve the account / profile information for a user by their user ID.

    Args:
        user_id: The unique identifier of the user.
    """
    return f"Profile for user {user_id}: Name=Jane Smith, Email=jane@example.com, Plan=Premium"


@tool
def get_order_status(order_id: str) -> str:
    """Get the current shipping / fulfilment status of an order by its order ID.

    Args:
        order_id: The unique identifier of the order.
    """
    return f"Order {order_id}: Shipped on 2025-03-10, expected delivery 2025-03-14"


@tool
def calculate_price(product_id: str, quantity: int) -> str:
    """Calculate the total price for a given product SKU and quantity.

    Args:
        product_id: The product SKU or identifier.
        quantity: Number of units to purchase.
    """
    unit_price = 29.99
    total = round(unit_price * quantity, 2)
    return f"Product {product_id} × {quantity} units = ${total}"


@tool
def search_products(query: str) -> str:
    """Search the product catalogue for items matching a search query.

    Args:
        query: Free-text search term (e.g. 'wireless headphones').
    """
    return f"Top results for '{query}': [Product A, Product B, Product C]"


@tool
def search_knowledge_base(query: str) -> str:
    """Search the platform knowledge base documents to answer general questions,
    retrieve definitions, explanations, policies, FAQs, or any information that
    is stored in the organisation's internal document repository.

    USE THIS TOOL when the user asks:
    - General knowledge questions (definitions, explanations, concepts)
    - Policy or FAQ questions ("what is the refund policy?", "how does X work?")
    - Document lookups ("what does the manual say about...")
    - Any question that requires searching stored documents or articles

    Args:
        query: The user's question or search keywords.
    """
    # Stub — in production this calls ChromaDB vector search + LLM synthesis
    return f"Knowledge base results for '{query}': [Doc A excerpt, Doc B excerpt]"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient with a given subject and body.

    Args:
        to: The recipient's email address.
        subject: The email subject line.
        body: The email body / message content.
    """
    return f"Email sent to {to} with subject '{subject}'."


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'TSLA', 'GOOGL').
    """
    return f"Stock {ticker}: $142.50 (+1.2% today)"


@tool
def translate_text(text: str, target_language: str) -> str:
    """Translate a piece of text into the specified target language.

    Args:
        text: The text to translate.
        target_language: The language to translate into (e.g. 'Spanish', 'French', 'German').
    """
    return f"Translation of '{text[:30]}...' to {target_language}: [translated text]"


@tool
def book_appointment(service: str, date: str, time: str) -> str:
    """Book an appointment for a specific service on a given date and time.

    Args:
        service: The type of appointment or service (e.g. 'dentist', 'haircut', 'consultation').
        date: The appointment date in YYYY-MM-DD format.
        time: The appointment time in HH:MM format.
    """
    return f"Appointment booked: {service} on {date} at {time}."


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the current exchange rate between two currencies.

    Args:
        from_currency: The source currency code (e.g. 'USD', 'EUR', 'GBP').
        to_currency: The target currency code (e.g. 'INR', 'JPY', 'AUD').
    """
    return f"1 {from_currency} = 83.12 {to_currency} (as of today)"


# ── All tools in a single list — referenced throughout the script ──────────────
ALL_TOOLS = [
    get_weather,
    get_user_profile,
    get_order_status,
    calculate_price,
    search_products,
    search_knowledge_base,
    send_email,
    get_stock_price,
    translate_text,
    book_appointment,
    get_exchange_rate,
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TEST CASES
#     Each entry: {"prompt": <user message>, "expected_tool": <tool name>}
#     Add or remove entries freely — the rest of the script adapts automatically.
# ══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    # ── get_weather ───────────────────────────────────────────────────────────
    {
        "prompt": "What is the weather in Delhi?",
        "expected_tool": "get_weather",
    },
    {
        "prompt": "How is the weather in New York today?",
        "expected_tool": "get_weather",
    },
    {
        "prompt": "Tell me the current weather in London.",
        "expected_tool": "get_weather",
    },
    {
        "prompt": "Is it raining in Mumbai right now?",
        "expected_tool": "get_weather",
    },

    # ── get_user_profile ──────────────────────────────────────────────────────
    {
        "prompt": "Show me the profile of user id 123.",
        "expected_tool": "get_user_profile",
    },
    {
        "prompt": "Get account details for user ID 456.",
        "expected_tool": "get_user_profile",
    },
    {
        "prompt": "Fetch the profile information for user 789.",
        "expected_tool": "get_user_profile",
    },
    {
        "prompt": "Who is user number 321?",
        "expected_tool": "get_user_profile",
    },

    # ── get_order_status ──────────────────────────────────────────────────────
    {
        "prompt": "Track my order number 456.",
        "expected_tool": "get_order_status",
    },
    {
        "prompt": "What is the status of order 789?",
        "expected_tool": "get_order_status",
    },
    {
        "prompt": "Where is my order ORD-001?",
        "expected_tool": "get_order_status",
    },
    {
        "prompt": "Has order 555 been shipped yet?",
        "expected_tool": "get_order_status",
    },

    # ── calculate_price ───────────────────────────────────────────────────────
    {
        "prompt": "How much would 5 units of product P001 cost?",
        "expected_tool": "calculate_price",
    },
    {
        "prompt": "Calculate the total price for 10 items of SKU X99.",
        "expected_tool": "calculate_price",
    },

    # ── search_products ───────────────────────────────────────────────────────
    {
        "prompt": "Search for wireless headphones in the catalog.",
        "expected_tool": "search_products",
    },
    {
        "prompt": "Find products related to gaming keyboards.",
        "expected_tool": "search_products",
    },
    {
        "prompt": "Look up bluetooth speakers.",
        "expected_tool": "search_products",
    },

    # ── search_knowledge_base ─────────────────────────────────────────────────
    {
        "prompt": "What is the company's refund policy?",
        "expected_tool": "search_knowledge_base",
    },
    {
        "prompt": "Explain how the onboarding process works.",
        "expected_tool": "search_knowledge_base",
    },
    {
        "prompt": "What does the documentation say about data privacy?",
        "expected_tool": "search_knowledge_base",
    },
    {
        "prompt": "Search the knowledge base for information about API rate limits.",
        "expected_tool": "search_knowledge_base",
    },
    {
        "prompt": "Find articles about subscription plan differences.",
        "expected_tool": "search_knowledge_base",
    },

    # ── send_email ────────────────────────────────────────────────────────────
    {
        "prompt": "Send an email to john@example.com with subject 'Meeting' and body 'Can we meet tomorrow?'",
        "expected_tool": "send_email",
    },
    {
        "prompt": "Email alice@company.com about the project update.",
        "expected_tool": "send_email",
    },
    {
        "prompt": "Send a message to hr@example.com with the subject 'Leave Request'.",
        "expected_tool": "send_email",
    },

    # ── get_stock_price ───────────────────────────────────────────────────────
    {
        "prompt": "What is the current stock price of Apple (AAPL)?",
        "expected_tool": "get_stock_price",
    },
    {
        "prompt": "Check the stock price for Tesla.",
        "expected_tool": "get_stock_price",
    },
    {
        "prompt": "How much is Google stock trading at right now?",
        "expected_tool": "get_stock_price",
    },

    # ── translate_text ────────────────────────────────────────────────────────
    {
        "prompt": "Translate 'Hello, how are you?' into Spanish.",
        "expected_tool": "translate_text",
    },
    {
        "prompt": "Can you translate this sentence to French: 'Good morning, have a nice day'?",
        "expected_tool": "translate_text",
    },
    {
        "prompt": "Translate 'Thank you very much' to Japanese.",
        "expected_tool": "translate_text",
    },

    # ── book_appointment ──────────────────────────────────────────────────────
    {
        "prompt": "Book a dentist appointment on 2026-03-20 at 10:00.",
        "expected_tool": "book_appointment",
    },
    {
        "prompt": "Schedule a haircut for March 25th at 2 PM.",
        "expected_tool": "book_appointment",
    },
    {
        "prompt": "I'd like to book a consultation on 2026-04-01 at 09:30.",
        "expected_tool": "book_appointment",
    },

    # ── get_exchange_rate ─────────────────────────────────────────────────────
    {
        "prompt": "What is the exchange rate from USD to INR?",
        "expected_tool": "get_exchange_rate",
    },
    {
        "prompt": "How many Euros can I get for 1 British Pound?",
        "expected_tool": "get_exchange_rate",
    },
    {
        "prompt": "Convert USD to Japanese Yen — what is the current rate?",
        "expected_tool": "get_exchange_rate",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  LLM FACTORY
#     Mirrors the _build_hf_chat_llm() pattern from your existing views.py.
# ══════════════════════════════════════════════════════════════════════════════

def build_llm_with_tools(tools: list):
    """
    Initialise a ChatHuggingFace LLM and bind tool schemas to it.

    The model receives the tool definitions as JSON schema so it knows which
    functions are available and how to call them.

    Args:
        tools: List of @tool-decorated callables to expose to the model.

    Returns:
        A LangChain Runnable that produces responses with optional tool_calls.
    """
    api_key = os.environ.get("HF_API_KEY", HF_API_KEY)
    model_id = os.environ.get("HF_MODEL_ID", HF_MODEL_ID)

    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,           # low temperature → more deterministic tool selection
        huggingfacehub_api_token=api_key,
        do_sample=False,
    )
    chat_llm = ChatHuggingFace(llm=endpoint, verbose=False)

    # bind_tools sends tool schemas in the request so the model can call them
    return chat_llm.bind_tools(tools)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SINGLE-PROMPT EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_single_prompt(llm_with_tools, prompt: str, expected_tool: str) -> dict:
    """
    Send one prompt to the model and determine whether it called the right tool.

    Args:
        llm_with_tools: The bound LLM returned by build_llm_with_tools().
        prompt:         Natural-language user message.
        expected_tool:  Name of the tool that *should* be called.

    Returns:
        A result dict with keys:
            prompt, expected_tool, tool_called, tool_args, outcome, error
        outcome is one of: "correct" | "wrong" | "no_tool" | "error"
    """
    result = {
        "prompt": prompt,
        "expected_tool": expected_tool,
        "tool_called": None,
        "tool_args": None,
        "outcome": None,
        "error": None,
    }

    try:
        response = llm_with_tools.invoke([HumanMessage(content=prompt)])

        # LangChain stores tool calls in response.tool_calls (list of dicts/objects)
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Use the FIRST tool call (most relevant for single-intent prompts)
            tc = response.tool_calls[0]

            # tool_calls entries can be plain dicts or ToolCall objects
            if isinstance(tc, dict):
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
            else:
                tool_name = tc.name
                tool_args = tc.args

            result["tool_called"] = tool_name
            result["tool_args"]   = tool_args
            result["outcome"]     = "correct" if tool_name == expected_tool else "wrong"

        else:
            # Model replied with plain text — no tool was invoked
            result["outcome"] = "no_tool"

    except Exception as exc:
        result["error"]   = str(exc)
        result["outcome"] = "error"

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5.  METRICS CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: list) -> dict:
    """
    Aggregate per-prompt results into summary statistics.

    Args:
        results: List of dicts returned by evaluate_single_prompt().

    Returns:
        Metrics dict (total, correct, wrong, no_tool, errors, success_rate).
    """
    total   = len(results)
    correct = sum(1 for r in results if r["outcome"] == "correct")
    wrong   = sum(1 for r in results if r["outcome"] == "wrong")
    no_tool = sum(1 for r in results if r["outcome"] == "no_tool")
    errors  = sum(1 for r in results if r["outcome"] == "error")

    tool_calls   = correct + wrong          # prompts where any tool was triggered
    success_rate = round((correct / total) * 100, 2) if total else 0.0

    return {
        "total_tests":                total,
        "tool_call_count":            tool_calls,
        "correct_tool_calls":         correct,
        "wrong_tool_calls":           wrong,
        "no_tool_calls":              no_tool,
        "errors":                     errors,
        "tool_call_success_rate_pct": success_rate,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  REPORT WRITER
# ══════════════════════════════════════════════════════════════════════════════

def save_report(results: list, metrics: dict, path: str = REPORT_FILE) -> None:
    """
    Persist the full evaluation results and metrics to a JSON file.

    Args:
        results: Per-prompt result dicts.
        metrics: Aggregated metrics dict.
        path:    Output file path.
    """
    report = {
        "generated_at": datetime.datetime.now().isoformat(),
        "model":        os.environ.get("HF_MODEL_ID", HF_MODEL_ID),
        "metrics":      metrics,
        "results":      results,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(f"\n  Report saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CONSOLE PRINTER
# ══════════════════════════════════════════════════════════════════════════════

# Visual badges for each outcome
_BADGE = {
    "correct": "CORRECT ",
    "wrong":   "WRONG   ",
    "no_tool": "NO TOOL ",
    "error":   "ERROR   ",
}


def print_result(index: int, result: dict) -> None:
    """Print a single test result to stdout in a readable format."""
    badge = _BADGE.get(result["outcome"], "???     ")
    print(f"[{index:02d}]  {badge}  |  Expected: {result['expected_tool']:<22}"
          f"|  Called: {result['tool_called'] or '—'}")
    print(f"       Prompt : {result['prompt']}")
    if result["tool_args"]:
        print(f"       Args   : {result['tool_args']}")
    if result["error"]:
        print(f"       Error  : {result['error']}")
    print()


def print_metrics(metrics: dict) -> None:
    """Print the evaluation summary metrics table to stdout."""
    w = 48
    print("=" * w)
    print("  EVALUATION SUMMARY")
    print("=" * w)
    print(f"  Total tests              : {metrics['total_tests']}")
    print(f"  Tool calls triggered     : {metrics['tool_call_count']}")
    print(f"  Correct tool calls       : {metrics['correct_tool_calls']}")
    print(f"  Wrong tool calls         : {metrics['wrong_tool_calls']}")
    print(f"  No tool called           : {metrics['no_tool_calls']}")
    print(f"  Errors                   : {metrics['errors']}")
    print(f"  Tool call success rate   : {metrics['tool_call_success_rate_pct']}%")
    print("=" * w)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    model_id = os.environ.get("HF_MODEL_ID", HF_MODEL_ID)

    print("\n" + "=" * 65)
    print(f"  Tool-Calling Evaluation")
    print(f"  Model : {model_id}")
    print(f"  Tests : {len(TEST_CASES)}")
    print("=" * 65 + "\n")

    # Build the LLM once (avoids re-initialising the endpoint per prompt)
    print("  Initialising LLM with tool bindings...")
    llm_with_tools = build_llm_with_tools(ALL_TOOLS)
    print("  Ready. Running evaluation...\n")

    results = []
    for idx, test_case in enumerate(TEST_CASES, start=1):
        result = evaluate_single_prompt(
            llm_with_tools,
            prompt=test_case["prompt"],
            expected_tool=test_case["expected_tool"],
        )
        results.append(result)
        print_result(idx, result)

    # Compute and display aggregate metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Persist full report to disk
    save_report(results, metrics, path=REPORT_FILE)


if __name__ == "__main__":
    main()
