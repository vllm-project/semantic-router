package consts

// UnknownLabel is a canonical fallback label value used across the codebase
// when a more specific value (e.g., model, category, reason) is not available.
const UnknownLabel = "unknown"

// Decision engine strategies.
const (
	PriorityStrategy   = "priority"
	ConfidenceStrategy = "confidence"
)

// LLM message types
const (
	USER      = "user"
	ASSISTANT = "assistant"
	SYSTEM    = "system"
)
