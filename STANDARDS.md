# Coding Standards & Style Guide

This repository follows strict programming standards across all languages to ensure consistency, maintainability, and safety.  
All code contributions must follow these conventions.

---

# 1. General Standards (All Languages)

## 1.1 Naming Conventions
- **Modules/Packages:** lowercase_with_underscores
- **Classes:** PascalCase
- **Functions/Methods:** camelCase
- **Variables:** descriptiveName (camelCase)
- **Constants:** UPPER_CASE_WITH_UNDERSCORES

## 1.2 Documentation
- Every file must begin with a module header comment.
- Every function must include:
  - Purpose  
  - Params  
  - Return values  
  - Exceptions or failure modes  

## 1.3 File Structure
- One class per file (Java, C++).
- Keep files < 500 lines where possible.
- Break large modules into submodules.

## 1.4 Error Handling
- Never swallow exceptions.  
- Always log actionable messages.  
- Use typed exceptions where possible.

## 1.5 Architectural Consistency
- No tight coupling  
- Use interfaces/abstract classes when appropriate  
- Follow dependency inversion in multi-module code  

---

# 2. Python Standards (PEP8)

Python code **must** follow PEP8.

Reference: https://peps.python.org/pep-0008/

## 2.1 Formatting
- Indentation: 4 spaces  
- Line length: 88 chars (Black-compatible)  
- Imports grouped: stdlib → third-party → project  

## 2.2 Specific Python Rules
- Type hints required for all public functions.
- Use docstrings in Google-style or NumPy-style.
- Avoid global variables.
- Avoid mutable default arguments.
- Use pathlib instead of os.path for file handling.

## 2.3 Logging
- Use `logging` library, not print().  
- All logs must include module name and severity.

---

# 3. Java Standards (Google Java Style Guide)

Reference: https://google.github.io/styleguide/javaguide.html

## 3.1 Formatting
- Indentation: 2 spaces  
- Line length: 100 chars  
- Use `final` wherever possible.  
- Use explicit types; avoid wildcard imports.

## 3.2 Java-Specific Rules
- Classes should be small and single-purpose.
- JavaDoc required for all public classes/methods.
- Exceptions:
  - Use checked exceptions for recoverable errors.
  - Never catch Exception blindly.

## 3.3 Packaging
- Package names: lowercase.nodots  
  Example:  
  `org.valkyrie.gcs.telemetry`

---

# 4. C++ Standards (C++ Core Guidelines)

Reference: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines

## 4.1 Formatting
- Indentation: 4 spaces  
- Header files `.h`, implementation `.cpp`  
- Include guards or `#pragma once`  

## 4.2 Modern C++ Required
- Use `std::unique_ptr`, `std::shared_ptr`, not raw `new`.  
- Use `auto` when type is obvious.  
- Never use C-style casts.  
- Avoid macros unless absolutely necessary.

## 4.3 Memory & Safety
- No raw pointers unless interfacing hardware.  
- RAII principles mandatory.  
- Objects must own clear lifecycles.

## 4.4 Error Handling
- Use exceptions for errors, not return codes.  
- Avoid catching by value.  
- Use references instead of pointers whenever possible.

---

# 5. Cross-Language Consistency Rules

## 5.1 Directory Structure
- `/src` for source code  
- `/tests` for unit/integration tests  
- `/config` for parameters  
- `/docs` for design/reference files  

## 5.2 Testing Standards
- All languages must include:
  - Unit tests  
  - Integration tests  
  - Mocking for hardware interfaces  
  - Clear pass/fail criteria  

## 5.3 Logging Rules
- Logs must follow consistent structure:
`[timestamp] [module] [severity]: message`

## 5.4 Prohibited Across All Languages
- Hardcoded credentials or constants  
- Inconsistent naming  
- Magic numbers  
- Silent failure  
- Global mutable state  
- Code without documentation  

---

By contributing code, you agree to follow these standards completely.
