"""
Analyst Agent
=============
Job: take the evidence bundle from the Retriever Agent and do the heavy
lifting - evaluate data, compare facts across documents, extract tables,
run calculations, and decide whether there's ENOUGH evidence to answer. If
not, it triggers a Feedback Loop back to the Retriever Agent for more.

This is implemented as a real LangChain tool-calling agent: the LLM itself
decides which tool(s) to call, in what order, and how many times (bounded
by max_iterations as a safety net), rather than a fixed hardcoded pipeline.
This mirrors the "Brain Core" / decision module shown in the project diagram.

Tools available to the agent (matching the project diagram):
  1. Calculator                  - safe arithmetic/statistics, no LLM math errors
  2. Table Extractor             - pulls real structured tables out of source PDFs
  3. Document Comparison         - LLM-structured comparison across multiple sources
  4. Data Analysis               - pandas-backed stats over extracted numeric data
  5. Search/Retrieve More Evidence - feedback loop back to the Retriever Agent

Entry point: analyze(question, initial_evidence) -> AnalysisResult
"""

import ast
import json
import operator
import os
import sys
from pathlib import Path
from typing import List, Optional, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pdfplumber
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from agents.retriever_agent import EvidenceChunk, retrieve_evidence

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
UPLOADS_DIR = Path(__file__).resolve().parent.parent / "data" / "uploads"


def _llm():
    return ChatGroq(model=GROQ_MODEL, temperature=0)


# ---------------------------------------------------------------------------
# 1. CALCULATOR
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


@tool
def calculator(expression: str) -> float:
    """Safely evaluate a numeric arithmetic expression instead of relying on
    an LLM to do math (which is error-prone). Supports +, -, *, /, %, ** and
    parentheses. For example: '(92 + 95 + 89) / 3' or '(95 - 89) / 89 * 100'.

    Args:
        expression: A plain arithmetic expression as a string.
    """
    tree = ast.parse(expression, mode="eval")
    return round(float(_safe_eval(tree.body)), 6)


@tool
def average(numbers: List[float]) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        numbers: List of numeric values.
    """
    if not numbers:
        raise ValueError("Cannot average an empty list.")
    return round(sum(numbers) / len(numbers), 6)


# ---------------------------------------------------------------------------
# 2. TABLE EXTRACTOR
# ---------------------------------------------------------------------------
class ExtractedTable(TypedDict):
    document: str
    page: int
    rows: List[List[str]]


@tool
def table_extractor(document: str, page: Optional[int] = None) -> List[ExtractedTable]:
    """Extract structured tables (rows/columns) from a source PDF, using the
    real page layout rather than treating the table as plain text. Useful for
    tables of results, metrics (accuracy/precision/recall/F1), or financial
    figures. If page is omitted, scans every page of the document.

    Args:
        document: The source filename (as seen in evidence metadata, e.g. 'paper.pdf').
        page: Optional 1-indexed page number to restrict extraction to.
    """
    file_path = UPLOADS_DIR / document
    if not file_path.exists() or file_path.suffix.lower() != ".pdf":
        return []

    tables_out: List[ExtractedTable] = []
    with pdfplumber.open(file_path) as pdf:
        page_indices = [page - 1] if page else range(len(pdf.pages))
        for idx in page_indices:
            if idx < 0 or idx >= len(pdf.pages):
                continue
            pdf_page = pdf.pages[idx]
            for table in pdf_page.extract_tables():
                cleaned_rows = [
                    [cell.strip() if cell else "" for cell in row]
                    for row in table
                ]
                tables_out.append({
                    "document": document,
                    "page": idx + 1,
                    "rows": cleaned_rows,
                })
    return tables_out


# ---------------------------------------------------------------------------
# 3. DOCUMENT COMPARISON
# ---------------------------------------------------------------------------
@tool
def document_comparison(question: str, evidence: List[EvidenceChunk]) -> str:
    """Compare information across multiple source documents - similarities,
    differences, advantages/disadvantages, methodologies, or results. Use
    when evidence spans 2+ distinct documents and the question asks for a
    comparison (e.g. 'which model performs best across these papers?').

    Args:
        question: The user's original question.
        evidence: Evidence chunks gathered by the Retriever Agent.
    """
    by_doc: dict[str, list[str]] = {}
    for c in evidence:
        by_doc.setdefault(c["document"], []).append(
            f"(page {c['page']}) {c['content']}"
        )

    sources_block = "\n\n".join(
        f"=== {doc} ===\n" + "\n---\n".join(chunks)
        for doc, chunks in by_doc.items()
    )

    prompt = f"""You are comparing evidence from multiple documents to answer:
"{question}"

For each document below, extract the relevant facts. Then produce a
structured comparison highlighting similarities, differences, and any
conclusion about which is best (only if the evidence supports it). Cite
document name and page number for every factual claim. Do not invent facts
not present in the evidence.

{sources_block}

Structured comparison:"""
    return _llm().invoke(prompt).content.strip()


# ---------------------------------------------------------------------------
# 4. DATA ANALYSIS
# ---------------------------------------------------------------------------
@tool
def data_analysis(records: List[dict], value_column: str) -> dict:
    """Run structured statistical analysis (average, min, max, std deviation,
    ranking) over a list of numeric records extracted from documents/tables.

    Args:
        records: List of dicts, e.g. [{"model": "CNN-A", "accuracy": 92}, ...].
        value_column: The key in each record holding the numeric value to analyze.
    """
    if not records:
        return {"error": "No records provided."}

    df = pd.DataFrame(records)
    if value_column not in df.columns:
        return {"error": f"Column '{value_column}' not found in records."}

    df[value_column] = pd.to_numeric(df[value_column], errors="coerce")
    df = df.dropna(subset=[value_column])
    if df.empty:
        return {"error": f"No numeric values found in column '{value_column}'."}

    ranked = df.sort_values(value_column, ascending=False).to_dict(orient="records")

    return {
        "count": int(len(df)),
        "average": round(float(df[value_column].mean()), 4),
        "min": round(float(df[value_column].min()), 4),
        "max": round(float(df[value_column].max()), 4),
        "std_dev": round(float(df[value_column].std() or 0), 4),
        "ranked": ranked,
    }


# ---------------------------------------------------------------------------
# 5. SEARCH / RETRIEVE MORE EVIDENCE  (the Feedback Loop)
# ---------------------------------------------------------------------------
def make_request_more_evidence_tool(source_filter: str = ""):
    """Factory that builds the request_more_evidence tool bound to a
    specific document scope (or none). This is called fresh per-question
    so that if the user has scoped the question to one document, the
    feedback loop automatically stays scoped to that same document too -
    without relying on the LLM to remember/pass a source_filter argument
    itself, which would be fragile."""

    @tool
    def request_more_evidence(refined_query: str, k: int = 6) -> List[EvidenceChunk]:
        """Ask the Retriever Agent for additional, more specific evidence when
        the current evidence bundle is insufficient to answer the question. This
        is the feedback loop: instead of guessing or answering with incomplete
        information, request exactly what's missing.

        Args:
            refined_query: A precise description of the missing information
                (e.g. 'computational complexity of Model X' rather than repeating
                the original broad question).
            k: Number of additional chunks to retrieve.
        """
        return retrieve_evidence(refined_query, final_size=k, source_filter=source_filter)

    return request_more_evidence


# Default (unscoped) instance, kept for direct standalone use/testing.
request_more_evidence = make_request_more_evidence_tool()


# ---------------------------------------------------------------------------
# FULL ANALYST AGENT  (real tool-calling agent, matching the "Brain Core"
# in the diagram: the LLM itself decides which tools to call and when)
# ---------------------------------------------------------------------------
from langchain.agents import create_agent

ANALYST_TOOLS = [
    calculator,
    average,
    table_extractor,
    document_comparison,
    data_analysis,
    request_more_evidence,
]

ANALYST_SYSTEM_PROMPT = """You are the Analyst Agent in a multi-agent RAG system.

You receive a user's question plus an initial evidence bundle (text chunks
with document name, page number, and content) gathered by a separate
Retriever Agent. Your job is to reason over that evidence and produce a
clear, well-grounded analysis for a downstream Answer Agent to format.

Tool usage rules (IMPORTANT):
- NEVER perform arithmetic yourself. Always call the `calculator` or
  `average` tool for any math, including sums, percentages, differences,
  or unit conversions (e.g. months to years).
- If the question needs data from a real table (metrics, results, figures),
  call `table_extractor` on the relevant document/page rather than guessing
  values from the text chunk alone.
- If the evidence spans 2+ distinct documents and the question compares
  them, call `document_comparison`.
- If you have structured numeric records and need averages/rankings/stats,
  call `data_analysis`.
- If the initial evidence bundle does NOT contain enough information to
  fully and accurately answer the question, call `request_more_evidence`
  with a precise description of what's missing. Do this at most 2 times -
  if you still lack evidence after that, clearly state what's missing in
  your final answer rather than guessing or inventing facts.
- Every factual claim in your final analysis MUST cite (document, page).
  Never state a fact that isn't backed by the evidence you were given or
  retrieved via tools.

When you have enough grounded information, respond with your final
analysis as plain text (not a tool call). This should be a clear,
structured write-up - use a comparison/table structure if helpful - ready
for a formatting agent to polish and cite.
"""


def build_analyst_agent(source_filter: str = ""):
    """Builds the Analyst as a real tool-calling agent (LangChain 1.x
    create_agent, LangGraph-based under the hood): the LLM decides which
    of the 5 tools to call, in what order, and how many times. The
    request_more_evidence tool is rebuilt per-call bound to source_filter
    so the feedback loop respects the same document scope as the rest of
    the question."""
    tools = [
        calculator,
        average,
        table_extractor,
        document_comparison,
        data_analysis,
        make_request_more_evidence_tool(source_filter),
    ]
    return create_agent(
        model=_llm(),
        tools=tools,
        system_prompt=ANALYST_SYSTEM_PROMPT,
    )


class AnalysisResult(TypedDict):
    question: str
    evidence: List[EvidenceChunk]
    analysis: str
    loops_used: int


def _format_evidence_block(evidence: List[EvidenceChunk]) -> str:
    return "\n\n".join(
        f"[{c['document']} p{c['page']}] (chunk_id={c['chunk_id']}) {c['content']}"
        for c in evidence
    ) or "(no evidence retrieved)"


def analyze(question: str, initial_evidence: List[EvidenceChunk], source_filter: str = "") -> AnalysisResult:
    """
    ...
    source_filter: if set, the feedback loop (request_more_evidence) stays
    scoped to only this document, matching how the initial evidence was
    retrieved.
    """
    agent = build_analyst_agent(source_filter=source_filter)
    input_text = f"""Question: {question}

Initial evidence bundle from the Retriever Agent:
{_format_evidence_block(initial_evidence)}"""

    # recursion_limit caps total graph steps (agent<->tools cycles) as a
    # safety net against infinite loops - roughly ~6-7 tool calls max.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": input_text}]},
        config={"recursion_limit": 15},
    )
    messages = result["messages"]
    analysis_text = messages[-1].content

    # Reconstruct the full evidence set (initial + anything fetched via the
    # feedback loop) by scanning tool messages for request_more_evidence
    # results. Tool outputs are JSON-serialized by LangChain when possible.
    all_evidence = list(initial_evidence)
    existing_ids = {c["chunk_id"] for c in all_evidence}
    loops_used = 0

    for msg in messages:
        if getattr(msg, "name", None) != "request_more_evidence":
            continue
        loops_used += 1
        content = msg.content
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            for c in parsed:
                if isinstance(c, dict) and c.get("chunk_id") not in existing_ids:
                    all_evidence.append(c)
                    existing_ids.add(c.get("chunk_id"))

    return {
        "question": question,
        "evidence": all_evidence,
        "analysis": analysis_text,
        "loops_used": loops_used,
    }


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What tools does the Analyst Agent use?"
    initial = retrieve_evidence(q)
    result = analyze(q, initial)
    print(json.dumps({
        "question": result["question"],
        "loops_used": result["loops_used"],
        "num_evidence_chunks": len(result["evidence"]),
        "analysis": result["analysis"],
    }, indent=2))
