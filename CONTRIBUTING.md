# Contributing Guidelines

Thank you for contributing to this project.  
To maintain consistency, safety, and clarity across all repositories, contributors must follow the rules below.  
These guidelines apply to **every repository** in the organisation.

---

## 1. Branching Strategy

### Main Principles
- `main` is always **stable**, **tested**, and **deployable**.
- `dev` is the integration branch for ongoing development.
- No one commits directly to `main` under any circumstances.

### Branch Naming
Use the following structure:

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/<module>/<short-desc>` | `feature/mavlink/altitude-handler` |
| Bugfix | `bugfix/<issue-id>-<short-desc>` | `bugfix/112-fix-timeout` |
| Hotfix | `hotfix/<desc>` | `hotfix/fpv-crash` |
| Documentation | `docs/<desc>` | `docs/api-update` |

---

## 2. Commit Rules

### Conventional Commit Format
All commits **must** use the format:
`<type>: <short description>`

Types allowed:
- `feat:` new feature  
- `fix:` bug fix  
- `refactor:` restructuring without behavior change  
- `test:` testing additions  
- `docs:` documentation changes  
- `perf:` performance improvements  
- `chore:` tooling/maintenance

### Commit Expectations
- The first line must be **≤ 72 characters**.
- Commits must be **atomic** — one logical change per commit.
- Do not commit commented-out code or unused files.
- No commit should break build, tests, or runtime behavior.

---

## 3. Pull Request (PR) Rules

### Requirements for All PRs
Every PR must include:
- **Summary:** What was changed and why.  
- **Context:** Link to tasks, issues, or mission requirements.  
- **Testing:** Steps performed, logs, or evidence (screenshots if applicable).  
- **Risks:** Any potential breaks or impacts.  
- **Checklist:**  
  - Code follows standards  
  - Tests pass  
  - No hardcoded credentials  
  - Documentation updated if needed  

### PR Reviews
- Minimum **1 reviewer for non-critical** changes.  
- Minimum **2 reviewers for flight-critical** or mission logic changes.  
- PRs touching safety, flight behavior, MAVLink routing, or hardware I/O **require approval from the subsystem lead**.

### Merging Rules
- No merge if tests fail.  
- Squash merge all PRs unless otherwise instructed.  
- PR must not add untracked files, logs, or large binaries.

---

## 4. Code Safety Rules

- No direct hardware edits without approval from subsystem lead.
- No unreviewed changes to:  
  - Flight control logic  
  - MAVLink messaging  
  - Sensor calibration code  
  - Safety overrides (spotter priority, kill switch behavior)

---

## 5. Documentation Requirements

All contributions must maintain:
- Function/module-level documentation  
- High-level architecture notes for new modules  
- Updated README or API documentation if functionality changes  

---

## 6. Non-Negotiable Rules

- No sensitive data in commits: passwords, WiFi SSIDs, tokens.  
- No large files (>10 MB) unless approved.  
- No code generation or auto-formatting that breaks consistency.  
- No "quick fixes" directly in main/dev — follow proper PR flow.

---

## 7. Communication

- Use clear, technical language.  
- If unsure, ask the subsystem lead before coding.  
- Discuss major design changes in issues **before** implementation.

---

By contributing to this project, you agree to follow all the rules listed above.  
Violation of safety-critical guidelines may result in revoked write access.
