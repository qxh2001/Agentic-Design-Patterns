"""List synthetic cases, or explicitly opt into paid model calls with --live."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Send synthetic cases to the configured OpenAI model; incurs API charges")
    parser.add_argument("--limit", type=int, default=1, help="Number of cases to run, default 1")
    parser.add_argument("--output", type=Path, default=ROOT / "eval-results" / "run.json")
    args = parser.parse_args()
    cases = json.loads((ROOT / "evals" / "cases.json").read_text())
    if not args.live:
        for case in cases:
            print(f"{case['id']}: {case['review_focus']}")
        print("No model calls made. Use --live --limit 1 to opt into a paid run.")
        return 0
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY before a live run.")
    if not 1 <= args.limit <= len(cases):
        parser.error(f"--limit must be between 1 and {len(cases)}")
    spec = importlib.util.spec_from_file_location("diagnostic_tutor", ROOT / "Chapter 1" / "Langchain.py")
    tutor = importlib.util.module_from_spec(spec); sys.modules[spec.name] = tutor; spec.loader.exec_module(tutor)
    results = []
    for case in cases[:args.limit]:
        started = time.monotonic()
        result = {"case_id": case["id"], "reference": case, "human_review": None}
        try:
            state = tutor.AgentState(problem_statement=case["question"], student_response=case["answer"], expected_solution=case["expected_solution"])
            # A noninteractive initial pass; clarification is recorded, not answered
            # automatically. This is distinct from the interactive CLI workflow.
            for node in [tutor.node_parse, tutor.node_hypotheses, tutor.node_fuse, tutor.node_feedback]:
                state = tutor.AgentState.model_validate({**state.model_dump(), **node(state)})
            proposed_question = state.diagnosis.what_to_ask_next
            # The authored answer belongs to this fixed question, not an arbitrary
            # generated one. Evaluate post-feedback under that explicit protocol.
            state.diagnosis.what_to_ask_next = case["followup_question"]
            state = tutor.AgentState.model_validate({**state.model_dump(), **tutor.node_generate_mastery(state)})
            initial = state.output.model_dump()
            state.mastery_answer = case["followup_answer"]
            final = tutor.node_final_from_mastery(state)["output"]
            result.update({
                "initial_output": initial, "final_output": final.model_dump(),
                "proposed_followup_question": proposed_question,
                "evaluation_followup_question": case["followup_question"],
                "automatic_checks": {
                    "structured_output_valid": True,
                    "has_student_evidence": bool(state.diagnosis.evidence),
                    "evidence_quotes_match_answer": all(e.strip() and e in case["answer"] for e in state.diagnosis.evidence),
                    "feedback_fields_present": all(getattr(final, f).strip() for f in ["feedback_to_student", "micro_activity", "teacher_note"]),
                },
            })
        except Exception as error:
            result["error_type"] = type(error).__name__
        result["latency_seconds"] = round(time.monotonic() - started, 3)
        results.append(result)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "model": tutor.MODEL, "mode": "noninteractive synthetic cases", "results": results,
              "note": "Post-feedback uses an authored question/answer pair; the model's proposed follow-up is recorded separately. Automatic checks are heuristics, not diagnosis accuracy or learning-effectiveness scores. Human review is required."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(results)} cases to {args.output}. Review the outputs using evals/README.md.")
    return 1 if any("error_type" in result for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
