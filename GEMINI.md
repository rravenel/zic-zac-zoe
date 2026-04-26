always use .venv when appropriate (like running python)

## Git & Workflow
- **Commit & Push:** All work must be committed and pushed to the remote repository.
- **Feature Branch Workflow:** This is a solo project. We are working directly in a feature branch; do not submit a PR for every task.
- **Commit Granularity:** Commits must contain work for a single task only. The commit message MUST explicitly indicate which task is being addressed (e.g., "Task 1.1: Implement scoring engine").
- **Progress Tracking:** Always read `PROGRESS.md` and follow the maintenance instructions there to ensure the project state is accurately reflected. Whenever a task is marked as ✅ **Completed** or ❌ **Blocked**, the corresponding work must be committed to git and pushed.
- **Incomplete Work:** If a task cannot be completed in one go, the partial work must be preserved (e.g., as a draft or WIP commit) and must not interfere with the functionality of the rest of the project.

## Coding Standards
- **Style & Best Practices:** Adhere to established coding styles and industry best practices.
- **Organization:** Keep code well-organized. Strive for conciseness without sacrificing human readability.
- **Documentation:**
  - Every class and function must have a clear descriptive comment at the top.
  - Include comments for any non-trivial logical sections within functions to explain the "why" and "how".
