always use .venv when appropriate (like running python)

## Git & Workflow
- **Commit & Push:** All work must be committed and pushed to the remote repository.
- **Feature Branch Workflow:** This is a solo project. We are working directly in a feature branch.
- **Bootstrapping:** Before beginning work, always read `STATUS.md` to understand the current project phase, active game variant, and AI architectural state.
- **History:** Completed build documents and specifications are archived in `docs/archive/`.

### Initial Build Procedure
- **Context:** Applies when tasks in `TASKS.md` are defined and ongoing.
- **Commit Granularity:** Commits must contain work for a single task only.
- **Commit Messages:** MUST explicitly indicate the Task ID (e.g., "Task 1.1: Implement scoring engine").
- **Tracking:** Maintain `PROGRESS.md` and specification files for every change. Whenever a task is marked as ✅ **Completed** or ❌ **Blocked**, the corresponding work must be committed to git and pushed.

### Active Feature Development
- **Workflow:** For non-trivial features or refactors, use `CURRENT_TASK.md` to define the scope and track progress.
- **Task Definition:** 
  - Every task must have a unique **Task Name**.
  - Always include a **QA** sub-task as the final step.
- **Status Tracking:**
  - **Sub-tasks:** Use explicit states: `pending`, `in-progress`, `blocked`, `complete`.
  - **Overall Status:** Represented as a roll-up of progress (e.g., `Status: 1/3 Complete`).
- **Atomic Commits:**
  - Each sub-task corresponds to a single commit.
  - **Commit Message Format:** `[Task Name] Sub-task X/Y: <Description>`
- **Completion & Cleanup Sequence:**
  1. **User Approval:** The task is only finished when the **user** explicitly declares that the QA sub-task has passed.
  2. **Final Task Commit:** Once QA passes, mark all sub-tasks (including QA) as `complete` and set the overall state to ✅ **Complete** in `CURRENT_TASK.md`. Commit this file immediately so the git history contains the full record of the finished task.
  3. **Workspace Reset:** After the final task commit:
     - Update `STATUS.md` and `README.md` to reflect the new state of the project.
     - Wipe `CURRENT_TASK.md`, leaving only an empty template.
     - Commit these changes together as a "Cleanup & Reset" commit.

### Maintenance & Debugging Procedure
- **Context:** Applies once initial tasks are complete, or for surgical bug fixes and iterative tweaks.
- **Commit Messages:** Do NOT use Task IDs. Use descriptive comments explaining the bug identified and the fix applied, or the clear intent of the feature refinement.
- **Tracking:** Documentation and progress updates are not required for minor iterative improvements or bug fixes.
- **Incomplete Work:** If a task or fix cannot be completed in one go, the partial work must be preserved (e.g., as a draft or WIP commit) and must not interfere with the functionality of the rest of the project.

## Coding Standards
- **Style & Best Practices:** Adhere to established coding styles and industry best practices.
- **Organization:** Keep code well-organized. Strive for conciseness without sacrificing human readability.
- **Documentation:**
  - Every class and function must have a clear descriptive comment at the top.
  - Include comments for any non-trivial logical sections within functions to explain the "why" and "how".
