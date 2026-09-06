"""
Orchestrator
============
Coordinates the complete agentic workflow shown in the project diagram:

    USER -> Orchestrator -> Retriever Agent -> Analyst Agent
                                 ^                  |
                                 |__ feedback loop __|
                                                      v
                                              Answer Agent -> USER

Also exposes a simple CLI chat loop and the document-ingestion entrypoint.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.document_pipeline import ingest_files
from agents.retriever_agent import retrieve_evidence
from agents.analyst_agent import analyze
from agents.answer_agent import generate_answer


def ask_stream(question: str, conversation_context: str = "", source_filter: str = ""):
    """
    Generator version of the pipeline: yields a dict at each stage
    transition instead of just printing. This is what powers the live
    "Retriever -> Analyst -> Answer" progress indicator in the web frontend,
    since it reflects the real pipeline state rather than a fake animation.

    source_filter: if set to a filename, scopes both the initial retrieval
    AND the Analyst's feedback loop to only that document.

    Yields dicts shaped like:
      {"stage": "retriever", "status": "start"}
      {"stage": "retriever", "status": "done", "evidence_count": int}
      {"stage": "analyst",   "status": "start"}
      {"stage": "analyst",   "status": "done", "loops_used": int, "evidence_count": int}
      {"stage": "answer",    "status": "start"}
      {"stage": "answer",    "status": "done", "answer": str, "evidence_used": list, "feedback_loops": int}
    """
    yield {"stage": "retriever", "status": "start"}
    initial_evidence = retrieve_evidence(
        question, conversation_context=conversation_context, source_filter=source_filter
    )
    yield {"stage": "retriever", "status": "done", "evidence_count": len(initial_evidence)}

    yield {"stage": "analyst", "status": "start"}
    analysis_result = analyze(question, initial_evidence, source_filter=source_filter)
    yield {
        "stage": "analyst",
        "status": "done",
        "loops_used": analysis_result["loops_used"],
        "evidence_count": len(analysis_result["evidence"]),
    }

    yield {"stage": "answer", "status": "start"}
    final_answer = generate_answer(question, analysis_result["analysis"], analysis_result["evidence"])
    yield {
        "stage": "answer",
        "status": "done",
        "answer": final_answer,
        "evidence_used": analysis_result["evidence"],
        "feedback_loops": analysis_result["loops_used"],
    }


def ask(question: str, conversation_context: str = "", source_filter: str = "") -> dict:
    """
    Run the full pipeline for a single user question (CLI-friendly wrapper
    around ask_stream() that prints progress and returns the final result).
    """
    print(f"\n[orchestrator] Question: {question}")
    final = {}

    for event in ask_stream(question, conversation_context=conversation_context, source_filter=source_filter):
        stage, status = event["stage"], event["status"]
        if stage == "retriever" and status == "start":
            print("[orchestrator] Step 1/3 -> Retriever Agent gathering evidence...")
        elif stage == "retriever" and status == "done":
            print(f"[orchestrator]   -> {event['evidence_count']} initial evidence chunks")
        elif stage == "analyst" and status == "start":
            print("[orchestrator] Step 2/3 -> Analyst Agent analyzing (with feedback loop if needed)...")
        elif stage == "analyst" and status == "done":
            print(f"[orchestrator]   -> feedback loops used: {event['loops_used']}, "
                  f"final evidence count: {event['evidence_count']}")
        elif stage == "answer" and status == "start":
            print("[orchestrator] Step 3/3 -> Answer Agent formatting final response...")
        elif stage == "answer" and status == "done":
            final = event

    return {
        "question": question,
        "answer": final["answer"],
        "evidence_used": final["evidence_used"],
        "feedback_loops": final["feedback_loops"],
    }


def ingest(*file_paths: str):
    """Ingest one or more documents into the vector DB before asking questions."""
    return ingest_files(list(file_paths))


def chat_loop():
    """Simple REPL: ask questions against whatever is currently in the vector DB."""
    print("=" * 60)
    print("Agentic RAG Chat - type 'exit' to quit")
    print("=" * 60)

    history = ""
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = ask(question, conversation_context=history)
        print(f"\nAssistant:\n{result['answer']}")
        history = f"Q: {question}\nA: {result['answer'][:300]}"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest(*sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "ask":
        result = ask(" ".join(sys.argv[2:]))
        print("\n" + "=" * 60)
        print(result["answer"])
    else:
        chat_loop()