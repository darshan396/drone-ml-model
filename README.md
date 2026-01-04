<p align="center">
  <img src="https://img.shields.io/badge/Project-UURT%20Template-4B9CD3?style=for-the-badge&logo=github" />
  <img src="https://img.shields.io/badge/Language-Python%20%7C%20C++%20%7C%20Java-3776AB?style=for-the-badge&logo=codefactor" />
  <img src="https://img.shields.io/badge/Standards-PEP8%20%7C%20Google%20Java%20Style%20%7C%20C++%20Core%20Guidelines-009688?style=for-the-badge&logo=readthedocs" />
  <img src="https://img.shields.io/badge/AI%20Agent-Safe%20Mode%20Enabled-7A288A?style=for-the-badge&logo=openai" />
  <img src="https://img.shields.io/badge/Tests-Required%20%2F%20Enforced-00C853?style=for-the-badge&logo=pytest" />
  <img src="https://img.shields.io/badge/PR%20Review-Strict%20%2F%20Mandatory-F44336?style=for-the-badge&logo=git" />
  <img src="https://img.shields.io/badge/Licence-UGV--DTU%20Internal-blue?style=for-the-badge&logo=creative-commons" />
</p>

# UURT — UGV Universal Repo Template  
A unified, multi-language engineering template for all UGV-DTU software repositories.

UURT provides a consistent structure, strict code-quality rules, AI agent protocol, 
and development workflows across Python, C++, and Java projects.  
It ensures that every repository follows the same engineering discipline, regardless 
of subsystem or developer.

UURT is designed for robotics, embedded systems, ground-station software, and 
mission-logic pipelines used at UGV-DTU.

---

## Purpose of UURT

This template exists to:
- Standardize **coding style, structure, and documentation** across all repos  
- Provide consistent rules for **LLM-generated code**  
- Maintain strict **safety and review controls** for flight-critical systems  
- Enforce a unified workflow for commits, branches, PRs, and reviews  
- Help new contributors onboard quickly  
- Ensure project reproducibility and engineering discipline

---

## Repository Structure

UURT/ <br>
│ <br>
├── CONTRIBUTING.md # Contribution rules & PR workflow <br>
├── STANDARDS.md # Python / C++ / Java style & architecture rules <br>
├── AGENTS.md # Protocol for LLM coding guidance <br>
├── REPO_STRUCTURE.md # Directory and module structure guidelines <br>
├── TESTING_STANDARDS.md # Unit, integration, and hardware test rules <br>
├── SECURITY_POLICY.md # Security rules for code, data, and credentials <br>
├── CODEOWNERS # Reviewer mapping for automatic PR assignment <br>
│ <br>
└── docs/ <br>
├── architecture_template.md <br>
├── api_documentation_template.md <br>
└── design_decision_record.md <br>


---

## How to Use This Template

1. Create a new repository using **UURT** as the base template.  
2. Replace placeholder files in `/docs` with project-specific documentation.  
3. Follow `REPO_STRUCTURE.md` when creating modules.  
4. Before contributing, read:
   - `CONTRIBUTING.md`
   - `STANDARDS.md`
   - `AGENTS.md`
5. Use feature branches and submit PRs according to the rules.

---

## Naming Conventions (Global)

| Type        | Style                        | Example                    |
|-------------|-------------------------------|----------------------------|
| Classes     | PascalCase                    | `SensorManager`            |
| Functions   | camelCase                     | `readAltitude()`           |
| Variables   | camelCase                     | `packetSize`               |
| Constants   | UPPER_CASE_WITH_UNDERSCORES   | `MAX_PACKET_SIZE`          |
| Modules     | snake_case                    | `mission_logic`            |
| Packages    | lowercase                     | `org.valkyrie.gcs.core`    |

Consistent naming ensures cross-language readability and lowers cognitive load when switching between repos.

---

## Testing Requirements

UURT enforces strict validation:

- Unit tests for all logic-heavy modules  
- Integration tests for multi-module systems  
- Hardware-in-loop test documentation for embedded or mission systems  
- Standardized result logs via `docs/test_reports/`  

See `TESTING_STANDARDS.md` for expectations.

---

## 🤖 AI Agent (LLM) Usage Rules

Any LLM assisting with code must follow the protocol in `AGENTS.md`.  
Violations can lead to unsafe, inconsistent, or unreviewable code.

The agent must:
- Follow all standards (Python PEP8, Google Java Style, C++ Core Guidelines)
- Never hallucinate APIs or modify safety-critical logic without approval
- Ask clarifying questions when requirements lack detail

---

## Security

The `SECURITY_POLICY.md` outlines:
- Credential handling  
- Prohibited data in commits  
- Rules for network configs, logs, and hardware IDs  
- Safe environment setup for Raspberry Pi and ground systems  

---

## CODEOWNERS

The template includes a base `CODEOWNERS` file, which automatically assigns reviewers based on file paths or subsystems.

Update this file when adapting UURT to a specific repo.

---

## Philosophy

UURT is designed around these core software engineering principles:

- **Consistency > Speed**  
- **Safety > Convenience**  
- **Clarity > Cleverness**  
- **Architecture > Quick Fixes**  
- **Reproducibility > Tribal Knowledge**

Robotics code must be predictable, documented, testable, and reviewable.

---

## Technologies Supported

- **Python** (PEP8, typed, documented, modular)
- **C++** (C++ Core Guidelines, RAII, no raw pointers unless required)
- **Java** (Google Java Style, clean packaging, strict JavaDoc rules)
- **LLMs as development accelerators, not architecture decision-makers**

---

## License

This template may be reused internally across all UGV-DTU projects.  
External use should credit the maintainers.

---

## Maintainers

[@ugv-dtu](https://github.com/ugv-dtu) <br>
[@vedantsinggh](https://github.com/vedantsinggh)
