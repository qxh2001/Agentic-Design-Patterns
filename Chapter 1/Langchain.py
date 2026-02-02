from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END


# =========================================================
# 0) Config
# =========================================================
MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model=MODEL, temperature=0)

CONFIDENCE_THRESHOLD = 0.72
MAX_CLARIFY_ROUNDS = 2
MAX_MASTERY_ROUNDS = 2


# =========================================================
# 1) State schemas
# =========================================================
class Hypothesis(BaseModel):
    kind: Literal["concept", "reasoning", "language"]
    claim: str
    evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    missing_info: Optional[str] = None


class Diagnosis(BaseModel):
    primary_kind: Literal["concept", "reasoning", "language"]
    primary_claim: str
    alt_claims: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0, le=1)
    evidence: List[str] = Field(default_factory=list)
    what_to_ask_next: Optional[str] = None  # helpful if low confidence


class ClarifyOut(BaseModel):
    questions: List[str] = Field(..., min_items=1, max_items=3)


class FeedbackCore(BaseModel):
    feedback_to_student: str
    micro_activity: str  # 2–5 minute intervention
    teacher_note: str


class MasteryCheck(BaseModel):
    question: str
    expected_key_points: List[str]  # 2-5
    common_wrong_signals: List[str]
    grading_rule: str  # concise pass/fail rule


class MasteryGradeOut(BaseModel):
    passed: bool
    justification: str
    remaining_issue: Optional[str] = None


class FinalOutput(BaseModel):
    diagnosis: Diagnosis
    feedback_to_student: str
    micro_activity: str
    teacher_note: str

    clarifying_questions: List[str] = Field(default_factory=list)
    clarifying_answers: List[str] = Field(default_factory=list)

    mastery_check: Optional[MasteryCheck] = None
    mastery_answer: Optional[str] = None
    mastery_passed: Optional[bool] = None
    mastery_justification: Optional[str] = None

    reteach_if_failed: Optional[str] = None


class AgentState(BaseModel):
    # Inputs
    problem_statement: str
    expected_solution: Optional[str] = None
    student_response: str
    grading_criteria: Optional[str] = None

    # Internal memory
    parsed_summary: Optional[str] = None
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    diagnosis: Optional[Diagnosis] = None

    clarify_round: int = 0
    clarifying_questions: List[str] = Field(default_factory=list)
    clarifying_answers: List[str] = Field(default_factory=list)

    mastery_round: int = 0
    mastery_check: Optional[MasteryCheck] = None
    mastery_answer: Optional[str] = None
    mastery_passed: Optional[bool] = None
    mastery_justification: Optional[str] = None
    remaining_issue: Optional[str] = None

    # Final
    output: Optional[FinalOutput] = None


# =========================================================
# 2) Prompts / chains
# =========================================================
parse_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an education assessment assistant for science. "
     "Summarize the student's response briefly and neutrally. Do NOT diagnose yet."),
    ("human",
     "Problem:\n{problem_statement}\n\n"
     "Student response:\n{student_response}\n\n"
     "Expected solution (optional):\n{expected_solution}\n\n"
     "Grading criteria (optional):\n{grading_criteria}")
])
parse_chain = parse_prompt | llm | StrOutputParser()

hypothesis_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Generate ONE diagnostic hypothesis of the given kind for a science concept question.\n"
     "- Be specific and evidence-based.\n"
     "- If evidence is weak, lower confidence and state what info is missing.\n"
     "- Avoid jargon unless necessary."),
    ("human",
     "Kind: {kind}\n\n"
     "Problem:\n{problem_statement}\n\n"
     "Student response:\n{student_response}\n\n"
     "Parsed summary:\n{parsed_summary}\n\n"
     "Expected solution (optional):\n{expected_solution}\n\n"
     "Clarifying answers (optional):\n{clarifying_answers}\n\n"
     "Grading criteria (optional):\n{grading_criteria}")
])
hypothesis_chain = hypothesis_prompt | llm.with_structured_output(Hypothesis)


class TopicQuestion(BaseModel):
    question: str


topic_question_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Generate ONE short diagnostic question (1 sentence) that assesses understanding of the given topic."
     " The question should be clear, unambiguous, and appropriate for a science concept check."),
    ("human",
     "Topic: {topic}\n\n"
     "Problem context (optional):\n{problem_statement}\n\n"
     "Produce exactly one question.")
])
topic_question_chain = topic_question_prompt | llm.with_structured_output(TopicQuestion)


class ProblemsOut(BaseModel):
    problems: List[str] = Field(..., min_items=5, max_items=5)


problem_generation_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Generate FIVE distinct example problems (each 1-2 sentences) related to the given topic."
     " Make them suitable as short science concept problems and label them clearly in a list."),
    ("human",
     "Topic: {topic}\n\n"
     "Context (optional):\n{problem_statement}\n\n"
     "Return exactly FIVE problems as an array of strings.")
])
problem_generation_chain = problem_generation_prompt | llm.with_structured_output(ProblemsOut)

class FuseOut(BaseModel):
    diagnosis: Diagnosis

fuse_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Fuse hypotheses into ONE primary diagnosis.\n"
     "- Pick the most evidence-supported claim.\n"
     "- Provide up to 3 alternative claims if plausible.\n"
     "- Set confidence conservatively.\n"
     "- If confidence is low, suggest what to ask next."),
    ("human",
     "Problem:\n{problem_statement}\n\n"
     "Student response:\n{student_response}\n\n"
     "Hypotheses:\n{hypotheses}\n\n"
     "Expected solution (optional):\n{expected_solution}\n\n"
     "Clarifying answers (optional):\n{clarifying_answers}\n\n"
     "Grading criteria (optional):\n{grading_criteria}")
])
fuse_chain = fuse_prompt | llm.with_structured_output(FuseOut)

clarify_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ask 1 clarifying question to disambiguate the student's misconception.\n"
     "Questions must be short, answerable, and targeted."),
    ("human",
     "Problem:\n{problem_statement}\n\n"
     "Student response:\n{student_response}\n\n"
     "Current diagnosis attempt:\n{diagnosis}\n\n"
     "Ask clarifying questions.")
])
clarify_chain = clarify_prompt | llm.with_structured_output(ClarifyOut)

feedback_prompt = ChatPromptTemplate.from_messages([
    ("system",
    "Write feedback that is kind, student-facing, and instructionally useful for a science concept question.\n"
    "Address the student directly using 'you' (e.g., 'You did X'). Keep an encouraging tone.\n"
    "Requirements:\n"
    "1) Start with one specific positive note about the student's response.\n"
    "2) Point to evidence from the student's response.\n"
    "3) Explain the correct idea in plain, student-friendly language.\n"
    "4) Give a clear 2–5 minute micro-activity the student can do next.\n"
    "5) Add a brief teacher note: what to look for next time."),
    ("human",
     "Problem:\n{problem_statement}\n\n"
     "Student response:\n{student_response}\n\n"
     "Diagnosis:\n{diagnosis}\n\n"
     "Clarifying Qs:\n{clarifying_questions}\n\n"
     "Clarifying answers:\n{clarifying_answers}\n\n"
     "Grading criteria (optional):\n{grading_criteria}")
])
feedback_chain = feedback_prompt | llm.with_structured_output(FeedbackCore)

mastery_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Create a minimal mastery check question that specifically tests whether the primary misconception is fixed.\n"
     "- Answerable in 1–3 sentences.\n"
     "- Avoid heavy computation.\n"
     "- Provide expected key points, common wrong signals, and a strict grading rule."),
    ("human",
     "Primary diagnosis:\n{primary_claim}\n\n"
     "Original problem context:\n{problem_statement}")
])
mastery_chain = mastery_prompt | llm.with_structured_output(MasteryCheck)

grade_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Grade the student's mastery check answer strictly using the grading rule and wrong signals.\n"
     "Return pass/fail, justification, and remaining issue if fail."),
    ("human",
     "Question:\n{question}\n\n"
     "Expected key points:\n{expected_key_points}\n\n"
     "Common wrong signals:\n{common_wrong_signals}\n\n"
     "Grading rule:\n{grading_rule}\n\n"
     "Student answer:\n{mastery_answer}")
])
grade_chain = grade_prompt | llm.with_structured_output(MasteryGradeOut)

reteach_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Generate a very targeted 2–3 minute reteach micro-activity focusing ONLY on the remaining issue.\n"
     "Be concrete: steps, a quick example, and what correct reasoning sounds like."),
    ("human",
     "Remaining issue:\n{remaining_issue}\n\n"
     "Original problem:\n{problem_statement}\n\n"
     "Primary diagnosis:\n{primary_claim}")
])
reteach_chain = reteach_prompt | llm | StrOutputParser()


# Post-mastery feedback: generate final student-facing feedback using the LLM
post_mastery_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an instructional coach. After a student has answered both the original question and a short mastery-check question, write final feedback addressed to the student."
     " Be kind, student-facing, and specific. Consider both responses when deciding what to praise, what remains unclear, and what next steps to recommend. Include:"
     " 1) one brief encouraging opening; 2) a clear explanation that integrates both answers;"
     " 3) a 2–3 minute micro-activity the student can do next; 4) a short teacher note about what to watch for next."),
    ("human",
     "Problem:\n{problem_statement}\n\n"
     "Original student answer:\n{initial_student_response}\n\n"
     "Mastery question:\n{mastery_question}\n\n"
     "Mastery answer:\n{mastery_answer}\n\n"
     "Expected key points:\n{expected_key_points}\n\n"
     "Common wrong signals:\n{common_wrong_signals}\n\n"
     "Write a short, student-facing feedback paragraph, a concise micro-activity (2–3 min), and a teacher note. Make sure the feedback explains how the mastery answer changes or confirms the original diagnosis."),
])
post_mastery_chain = post_mastery_prompt | llm.with_structured_output(FeedbackCore)


# =========================================================
# 3) Nodes
# =========================================================
def node_parse(state: AgentState) -> Dict[str, Any]:
    return {"parsed_summary": parse_chain.invoke(state.model_dump())}


def node_choose_problem(state: AgentState) -> Dict[str, Any]:
    # Offer a choice: 1) Topic selection (generate 5 problems) 2) Enter own question
    try:
        mode = input("Select option: 1) Topic selection (generate 5 problems) 2) Enter own question\nChoose 1 or 2 (default 1): ").strip()
    except Exception:
        mode = ""

    if mode == "" or mode == "1":
        # Topic-driven flow
        try:
            topic = input("Enter a topic to generate example problems for (e.g., 'photosynthesis'): ").strip()
        except Exception:
            topic = ""

        if not topic:
            topic = "general science concept"

        out = problem_generation_chain.invoke({"topic": topic, "problem_statement": state.problem_statement or ""})

        problems = out.problems
        print("\nGenerated example problems:")
        for i, p in enumerate(problems, start=1):
            print(f"{i}. {p}")

        choice = None
        while choice is None:
            try:
                sel = input("Choose a problem by number (1-5), or press Enter to pick 1: ").strip()
                if sel == "":
                    idx = 1
                else:
                    idx = int(sel)
                if 1 <= idx <= 5:
                    choice = problems[idx - 1]
                else:
                    print("Please enter a number between 1 and 5.")
            except Exception:
                print("Invalid input — please enter 1..5.")

        print(f"\nSelected problem:\n{choice}\n")
        try:
            student_resp = input("Enter student response for the selected problem (leave empty to use example): ").strip()
        except Exception:
            student_resp = ""

        return {"problem_statement": choice, "student_response": student_resp}
    else:
        # Custom problem entry
        try:
            custom = input("Enter your custom problem/question: ").strip()
        except Exception:
            custom = ""

        if not custom:
            custom = state.problem_statement or "Custom science question"

        try:
            student_resp = input("Enter student response for your custom problem (leave empty to use example): ").strip()
        except Exception:
            student_resp = ""

        # print(f"\nSelected custom problem:\n{custom}\n")
        return {"problem_statement": custom, "student_response": student_resp}

def node_hypotheses(state: AgentState) -> Dict[str, Any]:
    # Generate hypotheses for each kind using the hypothesis_chain (no interactive topic here).
    base = state.model_dump()
    hyps: List[Hypothesis] = []
    for kind in ["concept", "reasoning", "language"]:
        hyps.append(hypothesis_chain.invoke({**base, "kind": kind}))
    return {"hypotheses": hyps}

def node_fuse(state: AgentState) -> Dict[str, Any]:
    out = fuse_chain.invoke({
        **state.model_dump(),
        "hypotheses": [h.model_dump() for h in state.hypotheses]
    })
    return {"diagnosis": out.diagnosis}

def node_clarify(state: AgentState) -> Dict[str, Any]:
    out = clarify_chain.invoke({
        "problem_statement": state.problem_statement,
        "student_response": state.student_response,
        "diagnosis": state.diagnosis.model_dump() if state.diagnosis else None,
    })
    return {
        "clarifying_questions": out.questions,
        "clarify_round": state.clarify_round + 1
    }

def node_feedback(state: AgentState) -> Dict[str, Any]:
    core = feedback_chain.invoke({
        "problem_statement": state.problem_statement,
        "student_response": state.student_response,
        "diagnosis": state.diagnosis.model_dump() if state.diagnosis else None,
        "clarifying_questions": state.clarifying_questions,
        "clarifying_answers": state.clarifying_answers,
        "grading_criteria": state.grading_criteria,
    })
    out = FinalOutput(
        diagnosis=state.diagnosis,
        feedback_to_student=core.feedback_to_student,
        micro_activity=core.micro_activity,
        teacher_note=core.teacher_note,
        clarifying_questions=state.clarifying_questions,
        clarifying_answers=state.clarifying_answers,
        mastery_check=None,
        mastery_answer=state.mastery_answer,
        mastery_passed=state.mastery_passed,
        mastery_justification=state.mastery_justification,
        reteach_if_failed=None
    )
    return {"output": out}


def node_feedback_post_mastery(state: AgentState) -> Dict[str, Any]:
    """Generate feedback after a mastery attempt, ignoring any clarifying questions.
    This produces feedback based only on the current diagnosis and student's latest answer.
    """
    core = feedback_chain.invoke({
        "problem_statement": state.problem_statement,
        "student_response": state.student_response,
        "diagnosis": state.diagnosis.model_dump() if state.diagnosis else None,
        "clarifying_questions": [],
        "clarifying_answers": [],
        "grading_criteria": state.grading_criteria,
    })
    out = FinalOutput(
        diagnosis=state.diagnosis,
        feedback_to_student=core.feedback_to_student,
        micro_activity=core.micro_activity,
        teacher_note=core.teacher_note,
        clarifying_questions=[],
        clarifying_answers=[],
        mastery_check=state.mastery_check,
        mastery_answer=state.mastery_answer,
        mastery_passed=state.mastery_passed,
        mastery_justification=state.mastery_justification,
        reteach_if_failed=state.output.reteach_if_failed if state.output else None
    )
    return {"output": out}


def node_final_from_mastery(state: AgentState, _grade_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a final, self-contained feedback output based on both the original student answer
    and the student's mastery-check answer. Use the LLM (post_mastery_chain) to produce student-facing
    feedback. If the LLM call fails, fall back to a deterministic summary using available signals.
    """
    # Prefer generating post-mastery feedback via the LLM so it's not hard-coded.
    try:
        core = post_mastery_chain.invoke({
            "problem_statement": state.problem_statement or "",
            "initial_student_response": state.student_response or "",
            "mastery_question": state.mastery_check.question if state.mastery_check else "",
            "mastery_answer": state.mastery_answer or "",
            "expected_key_points": state.mastery_check.expected_key_points if state.mastery_check else [],
            "common_wrong_signals": state.mastery_check.common_wrong_signals if state.mastery_check else [],
        })
        feedback_text = core.feedback_to_student
        micro = core.micro_activity
        teacher = core.teacher_note
    except Exception:
        # Fallback deterministic feedback that integrates both answers
        initial = state.student_response or ""
        mastery = state.mastery_answer or ""
        if mastery and initial:
            feedback_text = (
                "Thanks for your responses — I can see how your second answer clarifies the first. "
                "Continue to connect your ideas by explicitly linking your reasoning to the key points below."
            )
        elif mastery:
            feedback_text = (
                "Thanks for your mastery answer — it helps check the earlier diagnosis. "
                "Use the micro-activity to practice applying the key idea."
            )
        else:
            feedback_text = (
                "Thanks for your work — consider revisiting the key idea and trying the short practice below."
            )

        # Construct a simple micro-activity from available key points or a generic prompt
        if state.mastery_check and state.mastery_check.expected_key_points:
            kp = state.mastery_check.expected_key_points[0]
            micro = (
                f"Micro-activity (2–3 min): Apply the key point '{kp}' in a short example and write 1–2 sentences explaining your reasoning."
            )
        else:
            micro = (
                "Micro-activity (2–3 min): Try a short targeted example and write 1–2 sentences explaining your reasoning."
            )
        teacher = "Review student responses for clearer linkage between reasoning steps."

    out = FinalOutput(
        diagnosis=state.diagnosis if state.diagnosis else Diagnosis(
            primary_kind="concept",
            primary_claim=(state.mastery_check.question if state.mastery_check else "mastery check"),
            alt_claims=[],
            confidence=0.0,
            evidence=[],
        ),
        feedback_to_student=feedback_text,
        micro_activity=micro,
        teacher_note=teacher,
        clarifying_questions=[],
        clarifying_answers=[],
        mastery_check=state.mastery_check,
        mastery_answer=state.mastery_answer,
        mastery_passed=state.mastery_passed,
        mastery_justification=state.mastery_justification,
        reteach_if_failed=state.output.reteach_if_failed if state.output else None
    )
    return {"output": out}

def node_generate_mastery(state: AgentState) -> Dict[str, Any]:
    mc = mastery_chain.invoke({
        "primary_claim": state.diagnosis.primary_claim if state.diagnosis else "",
        "problem_statement": state.problem_statement
    })

    # Prefer to use the clarifying/follow-up question as the mastery question when available.
    desired_q = None
    try:
        if state.clarifying_questions:
            desired_q = state.clarifying_questions[0]
    except Exception:
        desired_q = None
    if not desired_q:
        try:
            desired_q = state.diagnosis.what_to_ask_next if state.diagnosis and getattr(state.diagnosis, 'what_to_ask_next', None) else None
        except Exception:
            desired_q = None

    if desired_q:
        # Replace the question field in the generated MasteryCheck with the desired follow-up question
        try:
            mc_dict = mc.model_dump() if hasattr(mc, 'model_dump') else dict(mc)
            mc_dict['question'] = desired_q
            mc = MasteryCheck.parse_obj(mc_dict)
        except Exception:
            # If parsing fails, at least attach the original object and set question attribute if possible
            try:
                setattr(mc, 'question', desired_q)
            except Exception:
                pass

    if state.output:
        state.output.mastery_check = mc
    return {"mastery_check": mc, "mastery_round": state.mastery_round + 1, "output": state.output}

def node_grade_mastery(state: AgentState) -> Dict[str, Any]:
    # If no answer provided, stop after generating mastery check (external interaction needed)
    if not state.mastery_answer or not state.mastery_check:
        return {"mastery_passed": None, "mastery_justification": None}

    out = grade_chain.invoke({
        "question": state.mastery_check.question,
        "expected_key_points": state.mastery_check.expected_key_points,
        "common_wrong_signals": state.mastery_check.common_wrong_signals,
        "grading_rule": state.mastery_check.grading_rule,
        "mastery_answer": state.mastery_answer
    })

    if state.output:
        state.output.mastery_answer = state.mastery_answer
        state.output.mastery_passed = out.passed
        state.output.mastery_justification = out.justification

    return {
        "mastery_passed": out.passed,
        "mastery_justification": out.justification,
        "remaining_issue": out.remaining_issue,
        "output": state.output,
    }

def node_targeted_reteach(state: AgentState) -> Dict[str, Any]:
    if state.mastery_passed is None or state.mastery_passed is True:
        return {}

    reteach = reteach_chain.invoke({
        "remaining_issue": state.remaining_issue or "Misconception still present.",
        "problem_statement": state.problem_statement,
        "primary_claim": state.diagnosis.primary_claim if state.diagnosis else ""
    })

    if state.output:
        state.output.reteach_if_failed = reteach
    return {"output": state.output}


# =========================================================
# 4) Routing logic
# =========================================================
def route_confidence(state: AgentState) -> str:
    if not state.diagnosis:
        return "need_clarify"
    if state.diagnosis.confidence >= CONFIDENCE_THRESHOLD:
        return "high"
    if state.clarify_round >= MAX_CLARIFY_ROUNDS:
        return "force_finalize"
    return "need_clarify"

def route_mastery(state: AgentState) -> str:
    # No mastery answer yet -> stop after generating mastery check (ask student externally)
    if state.mastery_answer is None or state.mastery_answer.strip() == "":
        return "await_answer"
    if state.mastery_passed is True:
        return "pass"
    if state.mastery_round >= MAX_MASTERY_ROUNDS:
        return "stop"
    return "fail"


# =========================================================
# 5) Build graph
# =========================================================
graph = StateGraph(AgentState)

graph.add_node("choose_problem", node_choose_problem)
graph.add_node("parse", node_parse)
graph.add_node("hypotheses", node_hypotheses)
graph.add_node("fuse", node_fuse)
graph.add_node("clarify", node_clarify)

graph.add_node("feedback", node_feedback)
graph.add_node("generate_mastery", node_generate_mastery)
graph.add_node("targeted_reteach", node_targeted_reteach)

graph.set_entry_point("choose_problem")

graph.add_edge("choose_problem", "parse")
graph.add_edge("parse", "hypotheses")
graph.add_edge("hypotheses", "fuse")

graph.add_conditional_edges(
    "fuse",
    route_confidence,
    {
        "high": "feedback",
        "need_clarify": "clarify",
        "force_finalize": "feedback",
    }
)

# Clarify loop: if you already filled clarifying_answers, it will help hypotheses/fuse next time.
graph.add_edge("clarify", "hypotheses")

graph.add_edge("feedback", "generate_mastery")
graph.add_edge("generate_mastery", END)

graph.add_edge("targeted_reteach", "generate_mastery")

app = graph.compile()


# =========================================================
# 6) Demo
# =========================================================
if __name__ == "__main__":
    # Simplified single-run flow implementing:
    # question -> student answer -> feedback (+ follow-up) -> student follow-up answer -> final feedback

    import json

    # Introduction shown at program start
    intro = (
        "Diagnostic Tutor Tool for Science Concept Questions\n"
        "This interactive tool helps you: (1) pick or enter a science question, (2) provide a student response,\n"
        "(3) get an evidence-based diagnosis and kind, actionable feedback, (4) generate a short follow-up\n"
        "mastery question, and (5) produce final student-facing feedback.\n"
        "Follow the prompts below to run one session.\n"
    )
    print(intro)

    def pretty_print(obj):
        if not obj:
            print("<no data>")
            return

        try:
            # Convert pydantic models to plain dicts when possible
            if hasattr(obj, "model_dump"):
                data = obj.model_dump()
            elif isinstance(obj, dict):
                data = obj
            else:
                print(obj)
                return

            def clean(value):
                # Recursively remove None values and empty lists/dicts
                if isinstance(value, dict):
                    out = {}
                    for k, v in value.items():
                        cv = clean(v)
                        if cv is None:
                            continue
                        if isinstance(cv, (list, dict)) and len(cv) == 0:
                            continue
                        out[k] = cv
                    return out
                if isinstance(value, list):
                    lst = [clean(v) for v in value]
                    lst = [v for v in lst if v is not None and not (isinstance(v, (list, dict)) and len(v) == 0)]
                    return lst
                if value is None:
                    return None
                return value

            cleaned = clean(data)
            if not cleaned or (isinstance(cleaned, (list, dict)) and len(cleaned) == 0):
                print("<no data>")
                return

            # Render human-friendly output without JSON punctuation
            lines = []
            def add_heading(title):
                lines.append(f"{title}")
                lines.append("-" * len(title))

            # Diagnosis
            diag = cleaned.get("diagnosis")
            if diag:
                add_heading("Diagnosis")
                pk = diag.get("primary_kind")
                pc = diag.get("primary_claim")
                conf = diag.get("confidence")
                if pc:
                    lines.append(f"Primary claim ({pk}, confidence={conf}): {pc}")
                ev = diag.get("evidence") or []
                if ev:
                    lines.append("Evidence:")
                    for e in ev:
                        lines.append(f"  - {e}")
                if diag.get("what_to_ask_next"):
                    lines.append(f"What to ask next: {diag.get('what_to_ask_next')}")
                lines.append("")

            # Feedback
            if cleaned.get("feedback_to_student"):
                add_heading("Feedback to student")
                lines.append(cleaned.get("feedback_to_student"))
                lines.append("")

            if cleaned.get("micro_activity"):
                add_heading("Micro-activity")
                lines.append(cleaned.get("micro_activity"))
                lines.append("")

            if cleaned.get("teacher_note"):
                add_heading("Teacher note")
                lines.append(cleaned.get("teacher_note"))
                lines.append("")

            # Clarifying Qs/As
            cqs = cleaned.get("clarifying_questions") or []
            cas = cleaned.get("clarifying_answers") or []
            if cqs:
                add_heading("Clarifying questions and answers")
                for i, q in enumerate(cqs, start=1):
                    ans = cas[i-1] if i-1 < len(cas) else ""
                    lines.append(f"{i}. Q: {q}")
                    if ans:
                        lines.append(f"   A: {ans}")
                lines.append("")

            # Mastery check
            mc = cleaned.get("mastery_check")
            if mc:
                add_heading("Mastery check")
                if mc.get("question"):
                    lines.append(f"Question: {mc.get('question')}")
                ek = mc.get("expected_key_points") or []
                if ek:
                    lines.append("Expected key points:")
                    for p in ek:
                        lines.append(f"  - {p}")
                cws = mc.get("common_wrong_signals") or []
                if cws:
                    lines.append("Common wrong signals:")
                    for w in cws:
                        lines.append(f"  - {w}")
                lines.append("")

            # Mastery answer / pass / justification
            if cleaned.get("mastery_answer"):
                add_heading("Mastery answer")
                lines.append(cleaned.get("mastery_answer"))
                lines.append("")

            if cleaned.get("mastery_passed") is not None:
                add_heading("Mastery result")
                lines.append(f"Passed: {cleaned.get('mastery_passed')}")
                if cleaned.get("mastery_justification"):
                    lines.append(f"Justification: {cleaned.get('mastery_justification')}")
                lines.append("")

            if cleaned.get("reteach_if_failed"):
                add_heading("Reteach suggestion")
                lines.append(cleaned.get("reteach_if_failed"))
                lines.append("")

            print("\n".join(lines))
        except Exception:
            try:
                print(json.dumps(obj, indent=2, default=str))
            except Exception:
                print(obj)

    # 1) Run the graph to get initial diagnosis + feedback + (possibly) mastery_check
    s1 = AgentState(problem_statement="", expected_solution=None, student_response="", grading_criteria=None)
    out1 = app.invoke(s1)
    print("\n=== RUN 1 (diagnosis + feedback + mastery check) ===")
    pretty_print(out1.get("output"))

    # 2) Determine follow-up question: prefer clarifying_questions, then diagnosis.what_to_ask_next, then clarify_chain
    clar_q = None
    final_out = None
    if out1.get("output"):
        prior = out1.get("output")
        try:
            final_out = FinalOutput.parse_obj(prior) if isinstance(prior, dict) else prior
        except Exception:
            final_out = prior

    if final_out:
        clar_list = getattr(final_out, "clarifying_questions", []) or []
        if clar_list:
            clar_q = clar_list[0]
        else:
            try:
                clar_q = getattr(final_out, "diagnosis", None).what_to_ask_next
            except Exception:
                clar_q = None

    if not clar_q:
        try:
            clar_out = clarify_chain.invoke({
                "problem_statement": out1.get("problem_statement", ""),
                "student_response": out1.get("student_response", ""),
                "diagnosis": getattr(final_out, "diagnosis", None) if final_out else None,
            })
            clar_q = clar_out.questions[0] if clar_out and getattr(clar_out, "questions", None) else None
        except Exception:
            clar_q = None

    # 3) Ask follow-up (mastery) question and grade using a minimal state (no run1 context)
    if clar_q:
        print("\nFollow-up (mastery) question:\n", clar_q)
        follow_ans = input("Paste the student's short answer (leave empty to stop): ").strip()
        if follow_ans:
            # Build a minimal state for final feedback; include original problem and initial student response
            s2 = AgentState(problem_statement="", expected_solution=None, student_response="", grading_criteria=None)
            # copy original problem and student response from run1 so final feedback can integrate both
            try:
                s2.problem_statement = out1.get("problem_statement", "") or ""
            except Exception:
                s2.problem_statement = ""
            try:
                s2.student_response = out1.get("student_response", "") or ""
            except Exception:
                s2.student_response = ""

            # attach mastery_check from run1 if present
            try:
                if final_out and getattr(final_out, "mastery_check", None):
                    s2.mastery_check = final_out.mastery_check
                else:
                    s2.mastery_check = None
            except Exception:
                s2.mastery_check = None
            s2.mastery_answer = follow_ans

            # final feedback that integrates the original and mastery answers
            final_post = node_final_from_mastery(s2)
            print("\n=== FINAL FEEDBACK (post-mastery) ===")
            pretty_print(final_post.get("output"))
        else:
            print("No follow-up answer provided; stopping after feedback.")
    else:
        print("No follow-up question available; finished.")