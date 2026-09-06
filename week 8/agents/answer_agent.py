"""
Answer Agent
============
Job: once the Analyst Agent confirms there's enough evidence, turn the
evidence + analysis into a clear, well-formatted final answer with precise
citations (document name + page number).

Tools implemented (matching the project diagram):
  1. Citation Formatter   - inserts inline (document, page) citations
  2. Source Formatter     - builds a clean "Sources" reference list
  3. Response Formatter   - final polish pass (structure, tables, tone)

Grounding safeguards (important - added after testing surfaced citation
hallucination on a real run):
  - temperature=0 for all Answer Agent LLM calls (determinism > creativity
    for a task that must not invent facts).
  - The "Sources" footer is built programmatically from the real evidence
    metadata, never written by the LLM - this guarantees it can't drift
    from the truth.
  - validate_and_fix_citations() checks every inline (document, page)
    citation the LLM wrote against the real evidence and forces a single
    corrective LLM pass if anything doesn't match, rather than trusting
    the model to have followed the "don't invent citations" instruction.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import tool
from langchain_groq import ChatGroq

from agents.retriever_agent import EvidenceChunk

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# Matches inline citations like "(Paula Hanna CV - 1.pdf, p.2)"
_CITATION_RE = re.compile(
    r"\(([^()]+?\.(?:pdf|docx|txt|pptx|xlsx)),\s*p\.?\s*(\d+)\)",
    re.IGNORECASE,
)


def _llm():
    # temperature=0: this agent must never "creatively" vary facts/citations
    return ChatGroq(model=GROQ_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# 1. CITATION FORMATTER
# ---------------------------------------------------------------------------
@tool
def citation_formatter(analysis_text: str, evidence: List[EvidenceChunk]) -> str:
    """Rewrite analysis text so every factual claim carries an inline
    citation in the form (document.pdf, p.X), based strictly on the
    provided evidence. Never invents a citation not backed by the evidence.

    Args:
        analysis_text: The Analyst Agent's raw analysis/comparison text.
        evidence: The evidence chunks that back the analysis (for grounding).
    """
    valid_pages = _valid_pages_by_doc(evidence)
    valid_pages_block = "\n".join(
        f"- {doc}: ONLY these pages exist in the evidence: {sorted(pages)}"
        for doc, pages in valid_pages.items()
    )
    evidence_block = "\n".join(
        f"- {c['document']} (p.{c['page']}): {c['content'][:200]}" for c in evidence
    )
    prompt = f"""Here is analysis text and the evidence it's based on.

Analysis:
{analysis_text}

Evidence available for citation:
{evidence_block}

Valid (document, page) combinations - DO NOT cite any page outside this list:
{valid_pages_block}

Rewrite the analysis so that every factual claim ends with an inline
citation like (document.pdf, p.X), using ONLY the exact document/page
combinations listed above. Never invent, guess, or vary a page number -
copy it exactly from the valid list. Do not add new facts. Keep the same
meaning and structure, just add citations.

Cited analysis:"""
    return _llm().invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# 2. SOURCE FORMATTER
# ---------------------------------------------------------------------------
class SourceEntry(TypedDict):
    document: str
    pages: List[int]


def _valid_pages_by_doc(evidence: List[EvidenceChunk]) -> dict:
    grouped: dict[str, set] = {}
    for c in evidence:
        if c.get("page") is not None:
            grouped.setdefault(c["document"], set()).add(c["page"])
    return grouped


@tool
def source_formatter(evidence: List[EvidenceChunk]) -> List[SourceEntry]:
    """Build a clean, de-duplicated list of sources (document + all pages
    referenced) from the evidence bundle, suitable for a 'Sources' section.

    Args:
        evidence: The evidence chunks used in the final answer.
    """
    grouped = _valid_pages_by_doc(evidence)
    return [{"document": doc, "pages": sorted(pages)} for doc, pages in grouped.items()]


def _render_sources_section(sources: List[SourceEntry]) -> str:
    """Build the 'Sources' footer directly from real data - never via the
    LLM - so it can never drift from what was actually retrieved."""
    if not sources:
        return ""
    lines = [
        f"- {s['document']}" + (f" (page{'s' if len(s['pages']) > 1 else ''} "
                                  f"{', '.join(map(str, s['pages']))})" if s["pages"] else "")
        for s in sources
    ]
    return "**Sources**\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. RESPONSE FORMATTER
# ---------------------------------------------------------------------------
@tool
def response_formatter(question: str, cited_analysis: str) -> str:
    """Produce the final, polished, user-facing response body: clear prose
    or tables as appropriate, a short direct answer up top, and supporting
    detail below. Does NOT write the Sources section (that's appended
    programmatically afterward from verified evidence, not by the LLM).

    Args:
        question: The original user question.
        cited_analysis: Analysis text that already contains inline citations.
    """
    prompt = f"""User question: {question}

Cited analysis:
{cited_analysis}

Write the final answer body for the user:
- Start with a direct, concise answer (1-3 sentences).
- Follow with supporting detail, using a Markdown table if comparing
  multiple items/numbers.
- Keep every inline citation EXACTLY as given - do not change, remove, or
  add any (document, page) citation.
- Do NOT write a "Sources" or "References" section - that is added
  automatically afterward from verified data. Stop after the supporting
  detail.

Final answer body:"""
    text = _llm().invoke(prompt).content.strip()
    # Defensive cleanup: strip any Sources/References section the model
    # might still add out of habit, since we render that ourselves.
    text = re.split(r"\n#{0,3}\s*\*{0,2}(?:Sources|References)\*{0,2}\s*:?\s*\n",
                     text, maxsplit=1, flags=re.IGNORECASE)[0].rstrip()
    return text


# ---------------------------------------------------------------------------
# CITATION VALIDATION / REPAIR
# ---------------------------------------------------------------------------
def validate_and_fix_citations(text: str, evidence: List[EvidenceChunk]) -> str:
    """Scan every inline (document, page) citation in `text` against the
    real evidence. If any citation references a document/page combination
    that doesn't actually exist in the evidence, run one corrective LLM
    pass that fixes only those citations, rather than trusting the model
    to have gotten it right on the first try."""
    valid_pages = _valid_pages_by_doc(evidence)
    invalid_found = []

    for doc_name, page_str in _CITATION_RE.findall(text):
        doc_name = doc_name.strip()
        page = int(page_str)
        matched_doc = next((d for d in valid_pages if d.lower() == doc_name.lower()), None)
        if matched_doc is None or page not in valid_pages[matched_doc]:
            invalid_found.append((doc_name, page, sorted(valid_pages.get(matched_doc, set()))))

    if not invalid_found:
        return text

    issues_block = "\n".join(
        f"- You cited ({doc}, p.{page}), but the only valid page(s) for that "
        f"document are: {valid or 'NONE - remove this citation entirely'}"
        for doc, page, valid in invalid_found
    )
    prompt = f"""The following text has some incorrect citations that reference
pages not present in the actual evidence:

{text}

Problems found:
{issues_block}

Fix ONLY the incorrect citations listed above by replacing them with a
correct valid page for that document (or removing the citation if no valid
page exists). Do not change anything else - keep all facts, wording,
structure, and correct citations exactly as they are. Return the full
corrected text, nothing else.

Corrected text:"""
    return _llm().invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# FULL ANSWER AGENT ENTRYPOINT
# ---------------------------------------------------------------------------
def generate_answer(question: str, analysis_text: str, evidence: List[EvidenceChunk]) -> str:
    """Runs the Answer Agent pipeline: cite -> format body -> validate/fix
    citations -> append a programmatically-built, guaranteed-accurate
    Sources section."""
    cited = citation_formatter.invoke({"analysis_text": analysis_text, "evidence": evidence})
    body = response_formatter.invoke({"question": question, "cited_analysis": cited})
    body = validate_and_fix_citations(body, evidence)

    sources = source_formatter.invoke({"evidence": evidence})
    sources_section = _render_sources_section(sources)

    return f"{body}\n\n{sources_section}" if sources_section else body


ANSWER_TOOLS = [citation_formatter, source_formatter, response_formatter]


if __name__ == "__main__":
    from agents.retriever_agent import retrieve_evidence
    from agents.analyst_agent import analyze

    q = sys.argv[1] if len(sys.argv) > 1 else "What tools does the Retriever Agent use?"
    evidence = retrieve_evidence(q)
    result = analyze(q, evidence)
    final_answer = generate_answer(q, result["analysis"], result["evidence"])
    print(final_answer)
