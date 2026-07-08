"# Data workflow placeholder" 
# Team GitHub Workflow

## Branching Strategy

- Main branch contains stable and releasable code.
- Every new task is completed in a feature branch.
- Branch names follow:
  - feature/task-name
  - fix/task-name
  - docs/task-name
- Feature branches are deleted after merging.

---

## Commit Message Convention

Format:

[type]: description

Types used:
- feat
- fix
- docs
- refactor
- chore

This makes project history clear and easy to understand.

---

## Pull Request Process

- Create a Pull Request from the feature branch.
- Link the related GitHub issue.
- At least one teammate reviews the PR.
- Merge only after approval.

Review checks:
- Code correctness
- Readability
- Data quality
- Documentation

---

## GitHub Issue Tracking

- Every new task starts with a GitHub Issue.
- Each issue has:
  - Title
  - Description
  - Label
  - Assignee
- Issues are automatically closed after the related Pull Request is merged.