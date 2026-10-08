# Evaluation protocol

`cases.json` contains six authored synthetic cases: misconceptions, a correct answer, an ambiguous answer, and insufficient evidence. It is a starter set, not a representative benchmark or student dataset.

```bash
# Lists cases without importing the model client or calling an API.
python evals/run.py
# Explicit opt-in to paid OpenAI calls. Start with one synthetic case.
python evals/run.py --live --limit 1
```

The runner records initial/final structured outputs, the selected model, UTC time, latency, references, and automatic checks. It exercises a noninteractive initial pass and an authored follow-up question/answer pair. The model's proposed question is recorded separately; the fixed question is used to generate the rubric so the supplied answer is assessed against the question it actually answers. This evaluates post-feedback under a fixed protocol, rather than an end-to-end generated conversation. It does not simulate the CLI's clarification conversation. There can be up to nine model requests per case before SDK retries. Outputs stay in ignored `eval-results/` unless deliberately shared.

## Human review rubric

For each case, record a 0/1/2 score and a short justification for each dimension in `human_review`:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Scientific correctness | Incorrect or misleading | Partially correct | Accurate and appropriate to the question |
| Evidence and diagnosis | Invents a misconception/evidence | Plausible but weakly supported | Matches the response; abstains or clarifies when evidence is insufficient |
| Feedback usefulness | Generic or confusing | Relevant but incomplete | Specific explanation and a feasible targeted activity |
| Follow-up alignment | Unrelated question/rubric | Partly aligned | Question and reference key points test the diagnosed concept |
| Integration of answers | Ignores or misreads an answer | Uses both superficially | Explains how the second answer changes or supports the initial interpretation |

The evidence-substring check is intentionally strict and can flag legitimate paraphrases. Nonempty fields and valid schemas do not imply good tutoring. Model confidence is self-reported and uncalibrated. Review correct and uncertain answers for false diagnosis, and inspect failure fallbacks separately.

Record model/prompt version, repeat cases to assess variability, and compare against a simpler single-prompt baseline before reporting performance. Do not infer student learning gains from these synthetic cases. No live-run scores are claimed in this repository.
