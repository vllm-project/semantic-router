package headers

// Package headers provides constants for all custom HTTP headers used in the semantic router.
// All custom headers follow the "x-" prefix convention for non-standard HTTP headers.

// Request Headers
// These headers are used in incoming requests to the semantic router.
const (
	// RequestID is the unique identifier for tracking a request through the system.
	// This header is case-insensitive when read from incoming requests.
	RequestID = "x-request-id"

	// GatewayDestinationEndpoint specifies the backend endpoint address selected by the router.
	// This header is set by the router to direct Envoy to the appropriate upstream service.
	GatewayDestinationEndpoint = "x-vsr-destination-endpoint"

	// SelectedModel indicates the model that was selected by the router for processing.
	// This header is set during the routing decision phase.
	SelectedModel = "x-selected-model"
)

// VSR Decision Tracking Headers
// These headers are added to successful responses (HTTP 200-299) to track
// Vector Semantic Router decision-making information for debugging and monitoring.
// Headers are only added when the request is successful and did not hit the cache.
const (
	// VSRSelectedCategory indicates the category selected by VSR during classification.
	// Example values: "math", "business", "biology", "computer_science"
	VSRSelectedCategory = "x-vsr-selected-category"

	// VSRSelectedReasoning indicates whether reasoning mode was determined to be used.
	// Values: "on" (reasoning enabled) or "off" (reasoning disabled)
	VSRSelectedReasoning = "x-vsr-selected-reasoning"

	// VSRSelectedModel indicates the model selected by VSR for processing the request.
	// Example values: "deepseek-v31", "phi4", "gpt-4"
	VSRSelectedModel = "x-vsr-selected-model"

	// VSRInjectedSystemPrompt indicates whether a system prompt was injected into the request.
	// Values: "true" or "false"
	VSRInjectedSystemPrompt = "x-vsr-injected-system-prompt"

	// VSRCacheHit indicates that the response was served from cache.
	// Value: "true"
	VSRCacheHit = "x-vsr-cache-hit"
)

// Security Headers
// These headers are added to responses when security policies are violated
// or security checks detect potential threats.
const (
	// VSRPIIViolation indicates that the request was blocked due to PII policy violation.
	// Value: "true"
	VSRPIIViolation = "x-vsr-pii-violation"

	// VSRJailbreakBlocked indicates that a jailbreak attempt was detected and blocked.
	// Value: "true"
	VSRJailbreakBlocked = "x-vsr-jailbreak-blocked"

	// VSRJailbreakType specifies the type of jailbreak attempt that was detected.
	// Example values depend on the jailbreak detection classifier.
	VSRJailbreakType = "x-vsr-jailbreak-type"

	// VSRJailbreakConfidence indicates the confidence level of the jailbreak detection.
	// Value: floating point number formatted as string (e.g., "0.950")
	VSRJailbreakConfidence = "x-vsr-jailbreak-confidence"
)
