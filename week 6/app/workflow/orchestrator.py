"""
Evaluator-Generator Feedback Loop
-----------------------------------
User Question -> Generator -> Generated Answer -> Evaluator -> Decision
  - accept  -> Exit -> Final Answer
  - revise  -> Feedback -> Generator -> Improved Answer -> Evaluator (...)

Hard-capped at settings.max_loops (default 4). If the cap is hit without
an "accept" decision, the latest answer is returned with a disclaimer.
"""
from app.agents.generator import generate_answer
from app.agents.evaluator import evaluate_answer
from app.config import settings


def run_workflow(session_id: str, question: str) -> dict:
    feedback = ""
    answer = ""
    loop_count = 0
    history = []

    while loop_count < settings.max_loops:
        loop_count += 1

        gen_result = generate_answer(session_id, question, feedback)
        answer, context = gen_result["answer"], gen_result["context"]

        eval_result = evaluate_answer(session_id, question, answer, context)

        history.append({
            "loop": loop_count,
            "answer": answer,
            "decision": eval_result.get("decision"),
            "score": eval_result.get("score"),
            "feedback": eval_result.get("feedback"),
        })

        if eval_result.get("decision") == "accept":
            return {
                "session_id": session_id,
                "final_answer": answer,
                "status": "accepted",
                "loops_used": loop_count,
                "sources": gen_result.get("sources", []),
                "history": history,
            }

        feedback = eval_result.get("feedback", "") or "Improve accuracy and grounding."

    # Max loops reached without acceptance
    disclaimer = (
        f"\n\n[Note: This answer went through {settings.max_loops} improvement "
        "iterations and could not be fully validated by the Evaluator. "
        "Please review it critically.]"
    )
    return {
        "session_id": session_id,
        "final_answer": answer + disclaimer,
        "status": "max_loops_reached",
        "loops_used": loop_count,
        "history": history,
    }
