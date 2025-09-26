# Overview

Semantic Router provides content safety features to protect against malicious inputs, sensitive data exposure, and adversarial attacks at the routing layer.

## Core Concepts

### PII Detection

Automatically detects and protects personally identifiable information in user queries.

### Jailbreak Protection

Detects and blocks adversarial prompts and prompt injection attempts.

## Key Features

- **Real-time Protection**: Analyzes requests before they reach LLM endpoints
- **Model-specific Policies**: Configure different PII policies for different models
- **Automatic Filtering**: Models that don't meet security requirements are filtered out
- **Comprehensive Logging**: Complete audit trail of all security decisions
