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
	// VSRSelectedCategory indicates the category selected by VSR during domain classification.
	// This comes from the domain classifier (MMLU categories).
	// Example values: "math", "business", "biology", "computer_science"
	VSRSelectedCategory = "x-vsr-selected-category"

	// VSRSelectedDecision indicates the decision selected by VSR during decision evaluation.
	// This is the final routing decision made by the DecisionEngine.
	// Example values: "math_decision", "business_decision", "thinking_decision"
	VSRSelectedDecision = "x-vsr-selected-decision"

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

	// RouterReplayID carries the identifier for a captured replay record.
	// Value: opaque replay token
	RouterReplayID = "x-vsr-replay-id"
)

// VSR Signal Tracking Headers
// These headers track which signals were matched during request evaluation.
// They provide visibility into the signal-driven decision process.
const (
	// VSRMatchedKeywords contains comma-separated list of matched keyword rule names.
	// Example: "code_keywords,urgent_keywords"
	VSRMatchedKeywords = "x-vsr-matched-keywords"

	// VSRMatchedEmbeddings contains comma-separated list of matched embedding rule names.
	// Example: "code_debug,technical_help"
	VSRMatchedEmbeddings = "x-vsr-matched-embeddings"

	// VSRMatchedDomains contains comma-separated list of matched domain rule names.
	// Example: "computer_science,mathematics"
	VSRMatchedDomains = "x-vsr-matched-domains"

	// VSRMatchedFactCheck contains the fact-check signal result.
	// Values: "needs_fact_check" or "no_fact_check_needed"
	VSRMatchedFactCheck = "x-vsr-matched-fact-check"

	// VSRMatchedUserFeedback contains comma-separated list of matched user feedback signals.
	// Example: "need_clarification,wrong_answer"
	VSRMatchedUserFeedback = "x-vsr-matched-user-feedback"

	// VSRMatchedPreference contains comma-separated list of matched preference signals.
	// Example: "creative_writing,technical_analysis"
	VSRMatchedPreference = "x-vsr-matched-preference"

	// VSRMatchedLanguage contains comma-separated list of matched language signals.
	// Example: "en,zh,es"
	VSRMatchedLanguage = "x-vsr-matched-language"

	// VSRMatchedLatency contains comma-separated list of matched latency signals.
	// Example: "low_latency,medium_latency"
	VSRMatchedLatency = "x-vsr-matched-latency"

	// VSRMatchedContext contains comma-separated list of matched context rule names.
	// Example: "low_token_count,high_token_count"
	VSRMatchedContext = "x-vsr-matched-context"

	// VSRContextTokenCount contains the actual token count for the request.
	// Example: "1500"
	//nolint:gosec
	VSRContextTokenCount = "x-vsr-context-token-count"

	// VSRMatchedComplexity contains comma-separated list of matched complexity rules with difficulty levels.
	// Example: "code_complexity:hard,math_complexity:easy"
	VSRMatchedComplexity = "x-vsr-matched-complexity"
)

// Security Headers
// These headers are added to responses when security policies are violated
// or security checks detect potential threats.
const (
	// VSRPIIViolation indicates that the request was blocked due to PII policy violation.
	// Value: "true"
	VSRPIIViolation = "x-vsr-pii-violation"

	// VSRPIITypes contains the comma-separated list of PII types that were detected and denied.
	// Value: "EMAIL_ADDRESS,US_SSN" (example)
	VSRPIITypes = "x-vsr-pii-types"

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

// Hallucination Mitigation Headers
// These headers are added to responses when hallucination detection is enabled
// and potential hallucinations are detected in the LLM response.
const (
	// HallucinationDetected indicates that potential hallucination was detected in the response.
	// Value: "true"
	HallucinationDetected = "x-vsr-hallucination-detected"

	// HallucinationSpans contains a summary of unsupported claims found in the response.
	// Value: semicolon-separated list of claim summaries (truncated if too long)
	HallucinationSpans = "x-vsr-hallucination-spans"

	// FactCheckNeeded indicates whether the original prompt was classified as needing fact-checking.
	// Value: "true" or "false"
	FactCheckNeeded = "x-vsr-fact-check-needed"

	// UnverifiedFactualResponse indicates the response contains factual claims that could not be verified.
	// This occurs when the prompt was classified as needing fact-checking but no tool/RAG context
	// was available to verify the response against.
	// Value: "true"
	UnverifiedFactualResponse = "x-vsr-unverified-factual-response"

	// VerificationContextMissing indicates that no tool/RAG context was available for verification.
	// This header is set alongside UnverifiedFactualResponse to explain why verification couldn't occur.
	// Value: "true"
	VerificationContextMissing = "x-vsr-verification-context-missing"
)

// Looper Request Headers
// These headers are added to looper internal requests to identify them
// and allow the extproc to lookup decision configuration and apply plugins.
const (
	// VSRLooperRequest indicates this is an internal looper request.
	// When present, extproc should lookup the decision and execute configured plugins.
	// Value: "true"
	VSRLooperRequest = "x-vsr-looper-request"

	// VSRLooperIteration indicates the current iteration number in the looper loop.
	// Value: "1", "2", "3", etc.
	VSRLooperIteration = "x-vsr-looper-iteration"

	// VSRLooperDecision indicates the decision name for looper internal requests.
	// Used by extproc to lookup decision configuration and apply plugins.
	// Value: decision name (e.g., "remom_low_effort")
	VSRLooperDecision = "x-vsr-looper-decision"
)

// Looper Response Headers
// These headers are added to responses when looper mode is used.
const (
	// VSRLooperModel indicates the final model used by the looper.
	// Value: model name (e.g., "qwen-max")
	VSRLooperModel = "x-vsr-looper-model"

	// VSRLooperModelsUsed contains the comma-separated list of models that were called.
	// Value: "qwen-flash,qwen-max" (example)
	VSRLooperModelsUsed = "x-vsr-looper-models-used"

	// VSRLooperIterations indicates the total number of model calls made.
	// Value: "2", "3", etc.
	VSRLooperIterations = "x-vsr-looper-iterations"

	// VSRLooperAlgorithm indicates the algorithm used by the looper.
	// Value: "confidence", "ratings", "cost-aware"
	VSRLooperAlgorithm = "x-vsr-looper-algorithm"
)
