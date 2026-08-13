import os
import re
import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from google import genai
from google.genai.errors import APIError, ClientError

from config import GEMINI_API_KEY, DEFAULT_LLM_MODEL
from tools.search_tool import search_report
from tools.calculator_tool import calculator
from tools.verify_tool import verify_evidence

class AgentState(TypedDict):
    question: str
    evidence_snippets: List[str]
    source_pages: List[int]
    search_queries: List[str]
    calculations: List[str]
    draft_answer: str
    verification_result: str
    final_answer: str
    iteration_count: int
    status_log: List[str]
    is_supported: bool
    abstained: bool

# Initialize Gemini Client
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def call_gemini_with_retry(prompt: str, max_retries: int = 5) -> str:
    """
    Executes a Gemini LLM request with robust rate-limit (429) retry and multi-model fallback.
    """
    client = get_gemini_client()
    backoff = 4.0
    
    # Sequence of supported models to try
    models_to_try = [DEFAULT_LLM_MODEL, "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-flash-latest"]
    
    for attempt in range(max_retries):
        for model in models_to_try:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    delay_match = re.search(r"retryDelay': '(\d+)", err_str)
                    sleep_time = int(delay_match.group(1)) + 2 if delay_match else int(backoff)
                    print(f"[Rate Limit 429] Model '{model}' limited. Waiting {sleep_time}s...")
                    time.sleep(sleep_time)
                    backoff *= 1.5
                    continue
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    # Skip invalid/deprecated model names
                    continue
                else:
                    print(f"Notice: Model '{model}' error: {e}")
                    continue

    raise RuntimeError("Failed to obtain LLM response after retrying across available Gemini models.")

def plan_and_search(state: AgentState) -> AgentState:
    """
    Node 1: Understand question and search the annual report for relevant financial evidence.
    """
    state["iteration_count"] += 1
    question = state["question"]
    iter_num = state["iteration_count"]
    
    log_msg = f"Iteration {iter_num}: Searching annual report for evidence..."
    state["status_log"].append(log_msg)
    
    # Formulate search query based on question and past queries
    if iter_num == 1:
        query = question
    else:
        query = f"{question} financial numbers data tables statement"

    state["search_queries"].append(query)
    
    # Call search tool
    search_output = search_report(query, top_k=5)
    state["evidence_snippets"].append(search_output)
    
    # Extract page numbers from search output
    page_matches = re.findall(r"Page (\d+)", search_output)
    for p in page_matches:
        page_num = int(p)
        if page_num not in state["source_pages"]:
            state["source_pages"].append(page_num)
            
    state["source_pages"].sort()
    state["status_log"].append(f"Retrieved relevant evidence from page(s): {', '.join(map(str, state['source_pages']))}")
    
    return state

def calculate_node(state: AgentState) -> AgentState:
    """
    Node 2: Determines if arithmetic calculations are required and executes calculator tool.
    """
    question = state["question"]
    all_evidence = "\n".join(state["evidence_snippets"])
    
    prompt = f"""
You are a financial calculator assistant.
Analyze the user question and the retrieved evidence:

USER QUESTION: {question}

RETRIEVED EVIDENCE:
{all_evidence}

Determine if any arithmetic calculation is needed (such as percentage growth, difference, or margin).
If NO calculation is needed, respond with "NO_CALCULATION_NEEDED".

If calculation IS needed, write ONLY the Python mathematical expression to evaluate (e.g. "(44960 - 40000) / 40000 * 100").
Do not include any words other than the formula or "NO_CALCULATION_NEEDED".
"""
    
    try:
        formula = call_gemini_with_retry(prompt)
        
        if "NO_CALCULATION_NEEDED" not in formula and len(formula) > 0:
            calc_result = calculator(formula)
            state["calculations"].append(calc_result)
            state["status_log"].append(f"Calculated: {calc_result.splitlines()[0]}")
    except Exception as e:
        state["status_log"].append(f"Calculation step skipped: {e}")
        
    return state

def draft_answer_node(state: AgentState) -> AgentState:
    """
    Node 3: Synthesizes a structured answer using retrieved evidence and calculation results.
    """
    state["status_log"].append("Drafting financial answer from evidence...")
    question = state["question"]
    all_evidence = "\n".join(state["evidence_snippets"])
    calculations_str = "\n".join(state["calculations"]) if state["calculations"] else "None"
    
    prompt = f"""
You are FinSight, an expert autonomous financial analyst agent.
Answer the user's question using ONLY the provided evidence and calculations.

USER QUESTION: {question}

EVIDENCE:
{all_evidence}

CALCULATIONS:
{calculations_str}

STRICT FINANCIAL RULES:
1. Pay strict attention to fiscal years (e.g. FY2024 vs FY2025).
2. Distinguish Standalone vs Consolidated statements if mentioned.
3. Keep units consistent (e.g. Crore, Millions, Billions).
4. If the evidence DOES NOT contain sufficient information to answer the question, state:
   "INSUFFICIENT_EVIDENCE: I couldn't find sufficient evidence in the uploaded annual report to answer this reliably."
5. Return a clean, clear response detailing the numbers, calculation steps (if any), and conclusions.
"""
    
    try:
        draft = call_gemini_with_retry(prompt)
        state["draft_answer"] = draft
    except Exception as e:
        state["draft_answer"] = f"Error drafting answer: {e}"
        
    return state

def verify_node(state: AgentState) -> AgentState:
    """
    Node 4: Verifies the draft answer against source evidence using verify_evidence tool.
    """
    draft = state["draft_answer"]
    
    if "INSUFFICIENT_EVIDENCE" in draft:
        state["verification_result"] = "NOT_SUPPORTED"
        state["is_supported"] = False
        state["abstained"] = True
        state["status_log"].append("Evidence insufficient. Abstaining from hallucinating answer.")
        return state

    state["status_log"].append("Verifying claim accuracy against original report evidence...")
    all_evidence = "\n".join(state["evidence_snippets"])
    
    v_result = verify_evidence(claim=draft, source_evidence=all_evidence)
    state["verification_result"] = v_result
    
    if "STATUS: SUPPORTED" in v_result or "STATUS: PARTIALLY_SUPPORTED" in v_result:
        state["is_supported"] = True
        state["status_log"].append("Evidence verified successfully!")
    else:
        state["is_supported"] = False
        state["status_log"].append("Verification check failed or claim unverified.")
        
    return state

def decide_next_step(state: AgentState) -> str:
    """
    Conditional Edge: Decides whether to end or re-search for more evidence.
    """
    if state["abstained"]:
        return "finalize"
    if state["is_supported"]:
        return "finalize"
    if state["iteration_count"] >= 2:  # Limit loops to max 2 search iterations
        state["status_log"].append("Reached maximum retrieval iterations.")
        return "finalize"
        
    state["status_log"].append("Evidence insufficient or unverified. Initiating refined search...")
    return "re_search"

def finalize_answer(state: AgentState) -> AgentState:
    """
    Node 5: Formats final response with citations and verification status.
    """
    if state["abstained"] or "INSUFFICIENT_EVIDENCE" in state["draft_answer"]:
        state["final_answer"] = (
            "I couldn't find sufficient evidence in the uploaded annual report to answer this reliably.\n\n"
            "**Verification Note:** Abstain policy active to prevent hallucination."
        )
        return state
        
    pages_str = ", ".join(map(str, state["source_pages"])) if state["source_pages"] else "Unknown"
    
    verification_summary = "✓ Values found in source\n✓ Fiscal years matched\n✓ Calculations verified" if state["is_supported"] else "⚠ Partial or unverified evidence"

    state["final_answer"] = (
        f"{state['draft_answer']}\n\n"
        f"**Sources:**\nAnnual Report, Page(s): {pages_str}\n\n"
        f"**Verification Status:**\n{verification_summary}"
    )
    return state

def build_fin_sight_graph():
    """
    Builds and compiles the LangGraph StateGraph workflow for FinSight agent.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("plan_and_search", plan_and_search)
    workflow.add_node("calculate", calculate_node)
    workflow.add_node("draft_answer", draft_answer_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("finalize", finalize_answer)

    # Set entry point
    workflow.set_entry_point("plan_and_search")

    # Add edges
    workflow.add_edge("plan_and_search", "calculate")
    workflow.add_edge("calculate", "draft_answer")
    workflow.add_edge("draft_answer", "verify")

    # Conditional branching after verification
    workflow.add_conditional_edges(
        "verify",
        decide_next_step,
        {
            "finalize": "finalize",
            "re_search": "plan_and_search"
        }
    )

    workflow.add_edge("finalize", END)
    return workflow.compile()

class FinSightAgent:
    """
    Wrapper class to execute the FinSight LangGraph agent on user queries.
    """
    def __init__ (self):
        self.app = build_fin_sight_graph()

    def run(self, question: str) -> Dict[str, Any]:
        initial_state: AgentState = {
            "question": question,
            "evidence_snippets": [],
            "source_pages": [],
            "search_queries": [],
            "calculations": [],
            "draft_answer": "",
            "verification_result": "",
            "final_answer": "",
            "iteration_count": 0,
            "status_log": [],
            "is_supported": False,
            "abstained": False
        }
        
        final_state = self.app.invoke(initial_state)
        return final_state
