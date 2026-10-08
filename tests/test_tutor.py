import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
# A placeholder credential permits constructing SDK objects, but all model calls
# are replaced in these tests. No request is made and no user key is read.
with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-only"}):
    spec = importlib.util.spec_from_file_location("diagnostic_tutor", ROOT / "Chapter 1" / "Langchain.py")
    tutor = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = tutor
    spec.loader.exec_module(tutor)


class TutorTests(unittest.TestCase):
    def state(self, confidence=0.5):
        return tutor.AgentState(
            problem_statement="Why does a metal spoon feel colder than wood?",
            student_response="Metal starts colder.",
            diagnosis=tutor.Diagnosis(primary_kind="concept", primary_claim="Confuses temperature and heat transfer", confidence=confidence, evidence=["Metal starts colder."], what_to_ask_next="What would a thermometer show?"),
        )

    def test_uncertainty_routes_to_clarification_with_a_bounded_loop(self):
        state = self.state()
        self.assertEqual(tutor.route_confidence(state), "need_clarify")
        state.clarify_round = tutor.MAX_CLARIFY_ROUNDS
        self.assertEqual(tutor.route_confidence(state), "force_finalize")
        self.assertEqual(tutor.route_confidence(self.state(0.9)), "high")

    def test_clarification_collects_answers_instead_of_repeating_without_evidence(self):
        state = self.state()
        chain = Mock()
        chain.invoke.return_value = tutor.ClarifyOut(questions=["What would a thermometer show?"])
        with patch.object(tutor, "clarify_chain", chain), patch("builtins.input", return_value="The same temperature."):
            update = tutor.node_clarify(state)
        self.assertEqual(update["clarifying_answers"], ["The same temperature."])
        self.assertEqual(update["clarifying_questions"], ["What would a thermometer show?"])
        self.assertEqual(update["clarify_round"], 1)

    def test_followup_rubric_is_generated_for_the_question_actually_asked(self):
        state = self.state()
        chain = Mock()
        chain.invoke.return_value = tutor.MasteryRubric(expected_key_points=["Same temperature"], common_wrong_signals=["Metal starts colder"], grading_rule="Explain heat transfer")
        with patch.object(tutor, "rubric_chain", chain):
            update = tutor.node_generate_mastery(state)
        question = update["mastery_check"].question
        self.assertEqual(question, state.diagnosis.what_to_ask_next)
        self.assertEqual(chain.invoke.call_args.args[0]["question"], question)

    def test_post_feedback_uses_both_answers_and_preserves_diagnosis(self):
        state = self.state()
        state.mastery_answer = "They have the same temperature. Metal conducts heat faster."
        state.mastery_check = tutor.MasteryCheck(question="What would a thermometer show?", expected_key_points=["Same temperature"], common_wrong_signals=[], grading_rule="Explain heat flow")
        chain = Mock()
        chain.invoke.return_value = tutor.FeedbackCore(feedback_to_student="You distinguished temperature from heat transfer.", micro_activity="Compare a metal and wooden spoon.", teacher_note="Check transfer direction.")
        with patch.object(tutor, "post_mastery_chain", chain):
            output = tutor.node_final_from_mastery(state)["output"]
        inputs = chain.invoke.call_args.args[0]
        self.assertEqual(inputs["initial_student_response"], state.student_response)
        self.assertEqual(inputs["mastery_answer"], state.mastery_answer)
        self.assertEqual(output.diagnosis, state.diagnosis)
        self.assertIsNone(output.mastery_passed)

    def test_failure_fallback_does_not_claim_learning_or_a_completed_assessment(self):
        state = self.state()
        state.mastery_answer = "I do not know."
        chain = Mock(); chain.invoke.side_effect = RuntimeError("offline")
        with patch.object(tutor, "post_mastery_chain", chain):
            output = tutor.node_final_from_mastery(state)["output"]
        self.assertIn("cannot assess", output.feedback_to_student)
        self.assertIn("no model assessment", output.teacher_note)
        self.assertIsNone(output.mastery_passed)

    def test_missing_key_has_an_actionable_cli_error_without_network_access(self):
        env = os.environ.copy(); env.pop("OPENAI_API_KEY", None)
        result = subprocess.run([sys.executable, str(ROOT / "Chapter 1" / "Langchain.py")], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Set OPENAI_API_KEY", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
