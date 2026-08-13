import re
import time
from typing import Dict, Any
from google import genai
from google.genai.errors import APIError, ClientError
from config import GEMINI_API_KEY, DEFAULT_LLM_MODEL

def verify_evidence(claim: str, source_evidence: str, max_retries: int = 5) -> str:
    """
    Verify whether a proposed financial claim or calculation is strictly supported by source evidence.
    Includes rate limit retry logic and multi-model fallback.
    """
    if not claim or not claim.strip():
        return "Verification Error: Claim is empty."
    if not source_evidence or not source_evidence.strip():
        return "Verification Result: NOT_SUPPORTED\nReason: No source evidence provided to verify the claim."

    client = genai.Client(api_key=GEMINI_API_KEY)

    prompt = f"""
You are an expert financial auditor and verification engine.

Your task is to verify whether the following CLAIM is strictly supported by the provided SOURCE EVIDENCE from an annual report.

[CLAIM TO VERIFY]
{claim}

[SOURCE EVIDENCE]
{source_evidence}

Perform rigorous checks:
1. Are exact numerical values present and matching in the evidence?
2. Do the fiscal years match exactly (e.g. FY2024 vs FY2025)?
3. Do the units match (e.g. Crore, Millions, Billions, Percentage)?
4. Is the context accurate (Standalone vs Consolidated, Revenue vs Net Profit)?

Output your response in the following structured text format:
STATUS: [SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED]
NUMERICAL_MATCH: [YES / NO / PARTIAL]
FISCAL_YEAR_MATCH: [YES / NO / NOT_APPLICABLE]
UNITS_MATCH: [YES / NO / NOT_APPLICABLE]
EXPLANATION: [Concise 1-3 sentence explanation of the verification outcome]
"""

    backoff = 4.0
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
                    time.sleep(sleep_time)
                    backoff *= 1.5
                    continue
                elif "404" in err_str or "NOT_FOUND" in err_str:
                    continue
                else:
                    continue

    return "STATUS: NOT_SUPPORTED\nEXPLANATION: Gemini verifier hit rate limit."
