# Testing Standards

## Unit Tests
- Mandatory for all logic-heavy functions
- Must not require hardware access
- Coverage target: minimum 60%

## Integration Tests
- For multi-module interactions
- Must clearly define expected results and logs
- No silent failures allowed

## Hardware-in-Loop Tests
- Required for sensor, MAVLink, and mission logic
- Must include:
  - Procedure
  - Expected behavior
  - Safe abort conditions
  - Logs and plots

## Test Reporting
Store results inside:
`docs/test_reports/<date>_<module>.md`
