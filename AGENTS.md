# LLM Coding Agent Protocol

This document defines strict instructions for any LLM used to generate or edit code in this project.  
The model must follow these rules without exception.

---

# 1. Obedience to Repository Standards

The LLM MUST:
- Follow all rules in `CONTRIBUTING.md`
- Follow all coding rules in `STANDARDS.md`
- Follow repository structure exactly as defined
- Follow naming conventions consistently across all languages

---

# 2. Code Generation Rules

## 2.1 Style Consistency
- Python → must follow PEP8  
- Java → must follow Google Java Style  
- C++ → must follow C++ Core Guidelines

## 2.2 No Hallucinations
The model MUST NOT invent:
- APIs  
- Libraries  
- Hardware functions  
- Unsupported Pixhawk/MAVLink messages  

If information is missing, the LLM must respond:

> “Insufficient information — please specify X.”

## 2.3 Code Architecture
- Never dump long monolithic files.  
- Always structure modules according to repo standards:
  - sensors
  - camera
  - mavlink
  - mission_logic
  - utils/helpers

## 2.4 Documentation Requirements
Every generated function MUST include:
- Purpose  
- Inputs and types  
- Outputs  
- Exceptions and failure modes  

---

# 3. Safety Requirements (Flight-Critical)

LLM MUST NOT modify the following without explicit human approval:
- MAVLink routing logic  
- Sensor calibration  
- Spotter override system  
- Kill switch behavior  
- Battery failsafes  
- Flight-mode safety guards  

If such a change is requested, model must reply:

> “This change affects safety-critical systems. Please confirm with subsystem lead.”

---

# 4. Git + PR Behavior

LLM must:
- Suggest branch names following conventions  
- Suggest commit messages in Conventional Commit style  
- Provide PR descriptions following template  
- Warn if changes affect multiple independent modules  

---

# 5. Interaction Rules

## 5.1 The model must ask clarifying questions if:
- Requirements are incomplete  
- Hardware details are missing  
- Code structure is ambiguous  
- Behavior impacts safety  

## 5.2 The model must refuse bad practices such as:
- Hardcoded constants  
- Silent error handling  
- Large unstructured code dumps  
- Deprecated libraries  
- Unsafe memory access (C++)

---

# 6. Testing Requirements

Every generated code must include:
- Unit test stubs where possible  
- Integration test notes for hardware  
- Instructions on how to verify correctness  

---

# 7. Final Output Rules

When generating code:
- No placeholder pseudocode unless explicitly asked  
- File paths must match repo convention  
- Code must be ready to drop in with minimal modification  
- Must explain assumptions clearly  

---

This protocol ensures the LLM behaves like a disciplined junior engineer instead of an unpredictable generator.
