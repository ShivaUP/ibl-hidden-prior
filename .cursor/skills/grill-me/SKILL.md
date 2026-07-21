---
name: grill-me
description: Stress-test a plan or design through one-question-at-a-time interviewing until shared understanding is reached. Use before implementation for architecture, research, experiment, or project-definition work.
---

Interview the user relentlessly about every aspect of the plan until shared understanding is reached.

Behavior:
- Ask exactly one question at a time.
- For each question, provide your recommended answer before asking the user to decide.
- Walk down the design tree branch-by-branch, resolving dependencies in the right order.
- If a question can be answered by inspecting the repository, files, or existing notes, inspect them instead of asking.
- Do not start implementing until the plan is concrete and the user confirms the shared understanding is sufficient.

Questioning priorities:
1. Clarify the objective and success criteria.
2. Resolve scope boundaries and non-goals.
3. Resolve data dependencies and assumptions.
4. Resolve evaluation criteria and comparison fairness.
5. Resolve artifacts/deliverables and implementation order.
6. Surface risks, ambiguities, and missing definitions.

Default response format:
- Current branch
- Question
- Recommended answer
- Why this matters
- Await user answer

Stopping condition:
- Stop only when the project has a clear objective, scope, success criteria, evaluation plan, and implementation path.
- Then summarize the agreed plan in a compact spec before any coding begins.
