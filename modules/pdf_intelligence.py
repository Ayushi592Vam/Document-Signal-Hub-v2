"""
modules/pdf_intelligence.py

PDF Document Intelligence Pipeline — v2

Changes from v1:
  • Prompts now extract ONLY the exact fields relevant to each document type.
  • entities and type_specific are kept strictly on-topic (no generic noise).
  • Confidence values are preserved on every extracted field.

Pipeline:
  1. Text extraction   — from Azure DI page lines
  2. Classification    — FNOL / Legal / Loss Run / Medical
  3. Type-specific analysis — entities, signals, type_specific, judge
"""

from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# AZURE OPENAI CLIENT
# ─────────────────────────────────────────────────────────────────────────────

def _get_openai_client():
    try:
        from openai import AzureOpenAI
        return AzureOpenAI(
            azure_endpoint=os.environ.get("OPENAI_DEPLOYMENT_ENDPOINT", ""),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            api_version=os.environ.get("OPENAI_API_VERSION", "2024-12-01-preview"),
        )
    except Exception:
        return None


def _deployment() -> str:
    return os.environ.get("OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")


def _llm_call(system_prompt: str, user_prompt: str, max_tokens: int = 2500) -> dict | None:
    client = _get_openai_client()
    if not client:
        return None
    try:
        response = client.chat.completions.create(
            model=_deployment(),
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        return json.loads(raw)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — TEXT EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_full_text_from_parsed(parsed: dict) -> str:
    parts: list[str] = []
    for page in parsed.get("pages", []):
        raw = (page.get("raw_text") or "").strip()
        if raw:
            parts.append(f"[PAGE {page['page_num']}]\n{raw}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

_CLASSIFICATION_SYSTEM = textwrap.dedent("""
You are a senior insurance document analyst. Classify the document into exactly one of:
  - FNOL        : First Notice of Loss — initial claim intake / notification
  - Legal       : Court documents, complaints, dockets, attorney correspondence
  - Loss Run    : Tabular claims history, TPA loss run, portfolio reports
  - Medical     : Medical records, bills, EOBs, treatment notes, IMEs

Respond ONLY with valid JSON. No preamble.

{
  "classification": "<FNOL|Legal|Loss Run|Medical>",
  "confidence": <0.0–1.0>,
  "reasoning": "<2-3 sentences>",
  "ambiguities": "<mixed signals or empty string>"
}
""").strip()


def classify_document(full_text: str) -> dict:
    result = _llm_call(
        system_prompt=_CLASSIFICATION_SYSTEM,
        user_prompt=f"Classify this document:\n\n{full_text[:3000]}",
        max_tokens=400,
    )
    if not result:
        return {
            "classification": "Legal",
            "confidence": 0.5,
            "reasoning": "LLM unavailable — defaulted to Legal.",
            "ambiguities": "",
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — TYPE-SPECIFIC ANALYSIS PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUT_SCHEMA = """
Return ONLY valid JSON — no markdown, no preamble:

{
  "summary": "<200-word max factual summary>",
  "entities": {
    "<EXACT_FIELD_NAME>": {
      "value": "<exact value from document — do NOT paraphrase or infer>",
      "source_text": "<verbatim snippet that contains this value>",
      "confidence": <0.0–1.0>
    }
  },
  "signals": [
    {
      "type": "<severity|legal_escalation|fraud_indicator|medical_complexity|coverage_issue>",
      "severity_level": "<Highly Severe|High|Moderate|Low>",
      "description": "<plain-English explanation>",
      "supporting_text": "<verbatim quote>"
    }
  ],
  "type_specific": {
    "<FIELD_NAME>": {
      "value": "<exact value>",
      "source_text": "<verbatim snippet>",
      "confidence": <0.0–1.0>
    }
  },
  "judge": {
    "classification_reasoning": "<why this doc type>",
    "signal_validation": "<are detected signals credible? false positives?>",
    "data_quality": "<what is well-extracted vs uncertain or missing>",
    "recommendations": "<what a claims handler should do next>"
  }
}

CRITICAL RULES:
1. Extract ONLY fields listed under "Fields to extract" for entities AND type_specific.
2. Do NOT invent fields not listed — omit any field not found in the document.
3. Values must be EXACT text from the document — no paraphrasing, no inference.
4. Confidence: 0.95+ explicit text, 0.70–0.94 clear implication, <0.70 uncertain.
5. severity_level for signals:
   - Highly Severe: fatality / catastrophic loss / punitive damages / class action
   - High: active lawsuit / serious injury / large damages / attorney involved
   - Moderate: coverage question / potential fraud / ongoing treatment / escalation risk
   - Low: minor indicator / informational / routine complexity
"""


# ── FNOL ─────────────────────────────────────────────────────────────────────

_FNOL_SYSTEM = textwrap.dedent(f"""
You are an expert FNOL (First Notice of Loss) claims intake specialist.

Fields to extract for `entities` (all FNOL-relevant, skip if not in document):
  Claim Number, Policy Number, Policy Holder Name, Insured Name,
  Loss Date, Loss Time, Date Reported, Description of Loss,
  Location of Loss, Contact Name, Contact Phone, Contact Email,
  Vehicle Make, Vehicle Model, Vehicle Year, VIN,
  Claimant Name, Claimant Address, Claimant Phone,
  Adjuster Name, Adjuster Phone, Adjuster Email,
  Witness Name, Witness Phone, Police Report Number

Fields to extract for `type_specific` (FNOL claim assessment):
  Severity, Litigation Risk, Fraud Indicator, Coverage Concern,
  Estimated Loss Amount, Recommended Next Step

Signal types: severity, legal_escalation, fraud_indicator, coverage_issue

{_OUTPUT_SCHEMA}
""").strip()


# ── LEGAL ─────────────────────────────────────────────────────────────────────

_LEGAL_SYSTEM = textwrap.dedent(f"""
You are a legal claims analyst specialising in insurance litigation documents.

Fields to extract for `entities` (all Legal-relevant, skip if not in document):
  Case Number, Filing Date, Last Refreshed, Filing Location, Filing Court,
  Judge, Category, Practice Area, Matter Type, Status, Case Last Update,
  Docket Prepared For, Line of Business, Docket, Circuit, Division,
  Cause of Loss, Cause of Action, Case Complaint Summary,
  Plaintiff Name, Plaintiff Attorney, Plaintiff Attorney Firm,
  Defendant Name, Defendant Attorney, Defendant Attorney Firm,
  Insurance Carrier, Policy Number, Coverage Type,
  Incident Date, Incident Location, Damages Sought

Fields to extract for `type_specific` (Legal case assessment):
  Severity, Litigation Stage, Coverage Issue, Estimated Exposure,
  Reservation of Rights, Recommended Defense Strategy

Signal types: severity, legal_escalation, fraud_indicator, coverage_issue

{_OUTPUT_SCHEMA}
""").strip()


# ── LOSS RUN ──────────────────────────────────────────────────────────────────

_LOSS_RUN_SYSTEM = textwrap.dedent(f"""
You are a TPA loss run analyst specialising in claims portfolio analysis.

Fields to extract for `entities` (all Loss Run-relevant, skip if not in document):
  Report Date, Policy Number, Policy Period Start, Policy Period End,
  Named Insured, Carrier, TPA Name, Line of Business,
  Total Claims Count, Open Claims Count, Closed Claims Count,
  Total Incurred, Total Paid, Total Reserve, Total Indemnity Paid,
  Total Medical Paid, Total Expense Paid, Largest Claim Amount,
  Average Claim Amount, Loss Ratio, Combined Ratio

Fields to extract for `type_specific` (portfolio-level assessment):
  Portfolio Severity, Frequency Trend, Litigation Rate,
  Large Loss Count, Large Loss Threshold, Recommended Reserve Action

Signal types: severity, legal_escalation, fraud_indicator, coverage_issue

{_OUTPUT_SCHEMA}
""").strip()


# ── MEDICAL ───────────────────────────────────────────────────────────────────

_MEDICAL_SYSTEM = textwrap.dedent(f"""
You are a medical claims analyst specialising in insurance medical documents.

Fields to extract for `entities` (all Medical-relevant, skip if not in document):
  Patient Name, Patient DOB, Patient Gender, Patient ID,
  Provider Name, Provider NPI, Provider Facility, Provider Address,
  Date of Service, Date of Injury, Diagnosis, Primary ICD Code,
  Secondary ICD Codes, Procedure Codes, CPT Codes,
  Treatment Description, Medications Prescribed,
  Billing Amount, Amount Paid, Amount Denied, Adjustment,
  Insurance ID, Group Number, Authorization Number,
  Attending Physician, Referring Physician, Facility Name

Fields to extract for `type_specific` (medical case assessment):
  Severity, Medical Complexity, Treatment Duration,
  Disability Type, MMI Status, Causation Opinion,
  Fraud Indicator, Recommended IME

Signal types: severity, medical_complexity, fraud_indicator, coverage_issue

{_OUTPUT_SCHEMA}
""").strip()


_DOC_TYPE_TO_SYSTEM = {
    "FNOL":     _FNOL_SYSTEM,
    "Legal":    _LEGAL_SYSTEM,
    "Loss Run": _LOSS_RUN_SYSTEM,
    "Medical":  _MEDICAL_SYSTEM,
}


def analyse_document(full_text: str, doc_type: str) -> dict:
    system_prompt = _DOC_TYPE_TO_SYSTEM.get(doc_type, _LEGAL_SYSTEM)
    snippet = full_text[:6000]
    if len(full_text) > 6000:
        snippet += "\n\n[... document truncated ...]"

    result = _llm_call(
        system_prompt=system_prompt,
        user_prompt=(
            f"Document type: {doc_type}\n\n"
            f"Extract relevant fields and analyse:\n\n{snippet}"
        ),
        max_tokens=2500,
    )

    if not result:
        return _empty_analysis(doc_type)

    result.setdefault("summary", "")
    result.setdefault("entities", {})
    result.setdefault("signals", [])
    result.setdefault("type_specific", {})
    result.setdefault("judge", {})
    result["judge"].setdefault("classification_reasoning", "")
    result["judge"].setdefault("signal_validation", "")
    result["judge"].setdefault("data_quality", "")
    result["judge"].setdefault("recommendations", "")
    return result


def _empty_analysis(doc_type: str) -> dict:
    return {
        "summary": "Analysis unavailable — LLM could not be reached.",
        "entities": {},
        "signals": [],
        "type_specific": {},
        "judge": {
            "classification_reasoning": f"Classified as {doc_type}.",
            "signal_validation": "No signals detected.",
            "data_quality": "LLM unavailable.",
            "recommendations": "Manual review required.",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pdf_intelligence(parsed: dict) -> dict:
    """
    Full intelligence pipeline for a parsed PDF.

    Args:
        parsed: Output of parse_pdf_with_azure()

    Returns:
        {
          "full_text":      str,
          "classification": { classification, confidence, reasoning, ambiguities },
          "analysis":       { summary, entities, signals, type_specific, judge },
          "page_count":     int,
          "doc_type":       str,
        }
    """
    full_text  = extract_full_text_from_parsed(parsed)
    page_count = len(parsed.get("pages", []))

    classification = classify_document(full_text)
    doc_type       = classification.get("classification", "Legal")

    analysis = analyse_document(full_text, doc_type)

    return {
        "full_text":      full_text,
        "classification": classification,
        "analysis":       analysis,
        "page_count":     page_count,
        "doc_type":       doc_type,
    }