# Diagnostic Tutor — Agentic Design Patterns (CH1)

A small interactive CLI tool that uses LLM prompt-chains to diagnose student responses to short science concept questions and provide kind, targeted instructional feedback including a short follow-up (mastery) check.

## Key functionality
- Pick or enter a science question (topic-driven generation of 5 problems or custom question).  
- Accept a student's initial short answer.  
- Generate evidence-backed diagnostic hypotheses and fuse them into a single diagnosis.  
- Produce student-facing feedback (encouraging, concrete, micro-activity).  
- Ask one follow-up clarifying/mastery question, collect the student's second short answer, and produce final LLM-generated post-mastery feedback that integrates both answers.  
- CLI-friendly, cleans output for readable teacher/student display.

## How LangChain concepts are used
- Prompts-as-chains: Each step is implemented as a separate prompt-chain (parse, hypothesis generation, fuse, clarify, feedback, mastery, post-mastery feedback).  
- Structured outputs: Chains use structured output parsers (Pydantic models) so intermediate artifacts (Hypothesis, Diagnosis, MasteryCheck, FeedbackCore) are typed and machine-readable.  
- Composition: Chains are composed and orchestrated to form higher-level nodes (e.g., generate hypotheses → fuse → feedback) so responsibilities are separated and testable.  
- Minimal state graph: A small StateGraph wires nodes into a clear flow (choose_problem → parse → hypotheses → fuse → feedback → generate_mastery → END), enabling iterative clarify/mastery loops when needed.

## Graph & Nodes — technical overview

This project is orchestrated by a small StateGraph that composes focused nodes (functions/chains). Each node receives a minimal AgentState, performs one responsibility, and returns a dict of outputs that update the global state. Intermediate artifacts are typed with Pydantic models so chains produce structured, inspectable outputs.

Nodes (concise):
- choose_problem — prompts user to pick/generate or enter a problem; returns problem_statement and student_response.  
- parse — normalizes/cleans the problem and response into canonical fields.  
- hypotheses — runs separate hypothesis-generation chains (e.g., concept, reasoning, language) and returns a list[Hypothesis].  
- fuse — combines hypotheses via a fuse chain into a single Diagnosis (primary claim, alt_claims, evidence, confidence, what_to_ask_next).  
- clarify — when the fuse/feedback suggests follow-up probes, generate a clarifying question (or use diagnosis.what_to_ask_next) to disambiguate student thinking.  
- feedback — produce initial, student-facing feedback, micro-activity, and teacher notes based on the diagnosis and initial answer.  
- generate_mastery — create a mastery (follow-up) question; when a clarifying question exists it is promoted to the mastery question.  
- final_from_mastery (post-mastery feedback) — given the initial problem/answer, the mastery question, and the mastery answer, call an LLM chain to produce the final student-facing feedback that integrates both answers and recommends next steps.

Routing and control:
- Nodes are intentionally small and composable; the StateGraph wires them into a linear/iterative flow with optional clarify/mastery loops.  
- Decision points use simple thresholds (CONFIDENCE_THRESHOLD, MAX_CLARIFY_ROUNDS) to decide whether to probe further or finalize feedback.  
- The graph ends at END after producing post-mastery feedback; no graded-pass logic is retained (final feedback is LLM-generated and combines both student answers).

Why this design:
- Single-responsibility nodes improve prompt reliability and make debugging easier.  
- Typed outputs (Pydantic) make intermediate decisions explicit and testable.  
- Composing chains into a graph mirrors instructional workflows (diagnose → probe → reteach/check → finalize) while keeping the implementation modular and extensible.

## Why this approach solves the problem
- Modularity: Breaking the task into focused chains reduces prompt complexity and improves reliability (each chain does one thing well).  
- Interpretability: Structured outputs make internal decisions (diagnosis, evidence, next questions) explicit and inspectable for debugging or human review.  
- Iterative assessment: The clarify → fuse → feedback → mastery pattern mirrors human tutoring: diagnose, probe, reteach/check, then finalize.  
- Student-facing quality: A dedicated post-mastery LLM chain generates final feedback that explicitly incorporates the initial question, initial answer, mastery question, and mastery answer for coherent, kind instruction.

## Quick start
1. Install dependencies (example):
   - Python 3.10+
   - pip install pydantic langchain_openai langchain_core langgraph
2. Set your OpenAI/LLM credentials per your provider (e.g., OPENAI_API_KEY).  
3. Run the demo:
```bash
python3 "/Users/katherine/Desktop/Agentic Design Patterns/CH1-Langchain.py"
```

## Notes & customization
- Configure model / temperature via the `MODEL` and `llm` settings at the top of `CH1-Langchain.py`.  
- Prompt templates live next to their chain definitions for easy tuning.  
- The StateGraph is deliberately small so you can adapt routing (clarify vs. finalize thresholds) or remove/add nodes (e.g., reteach) without changing chain internals.

## License
MIT — adapt freely for instructional use.