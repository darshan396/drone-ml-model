# Security Policy

## Prohibited in Repository
- Passwords, WiFi keys, SSIDs
- API tokens or SSH private keys
- Sensitive GPS logs
- Personal data of team members

## Raspberry Pi & Ground Station
- SSH keys must be stored outside repository
- Use environment variables for secrets
- Do not expose ports publicly

## Code Security Rules
- No hardcoded credentials
- Validate all external inputs
- Use logging instead of silent failures
