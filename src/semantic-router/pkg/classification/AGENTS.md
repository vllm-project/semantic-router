# Classification runtime

- Request-time inference is separate from model discovery/bootstrap and service
  assembly.
- Category, jailbreak, PII, and other families own their backend and mapping
  behavior; do not add another shared backend-selection matrix to
  `classifier.go`.
- Keep the unified batch path and fallbacks behind explicit family adapters.
