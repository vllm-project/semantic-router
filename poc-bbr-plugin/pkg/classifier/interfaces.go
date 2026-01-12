/*
Package classifier provides interfaces for classification, PII detection, and jailbreak detection.
These interfaces wrap the actual vSR implementations for use in the BBR plugin.
*/
package classifier

// CategoryResult represents the result of category classification
type CategoryResult struct {
	Category   string  `json:"category"`
	Confidence float32 `json:"confidence"`
	LatencyMs  int64   `json:"latency_ms"`
}

// PIIResult represents the result of PII detection
type PIIResult struct {
	HasPII    bool     `json:"has_pii"`
	PIITypes  []string `json:"pii_types"`
	Blocked   bool     `json:"blocked"`
	LatencyMs int64    `json:"latency_ms"`
}

// JailbreakResult represents the result of jailbreak detection
type JailbreakResult struct {
	IsJailbreak bool    `json:"is_jailbreak"`
	ThreatType  string  `json:"threat_type"`
	Confidence  float32 `json:"confidence"`
	Blocked     bool    `json:"blocked"`
	LatencyMs   int64   `json:"latency_ms"`
}

// Classifier is the interface for category/intent classification
type Classifier interface {
	// Classify classifies the input text and returns the category result
	Classify(text string) (CategoryResult, error)
	// IsEnabled returns whether the classifier is enabled
	IsEnabled() bool
}

// PIIDetector is the interface for PII detection
type PIIDetector interface {
	// DetectPII checks the input text for PII and returns the result
	DetectPII(text string) (PIIResult, error)
	// IsPIIEnabled returns whether PII detection is enabled
	IsPIIEnabled() bool
}

// JailbreakDetector is the interface for jailbreak detection
type JailbreakDetector interface {
	// DetectJailbreak checks the input text for jailbreak attempts and returns the result
	DetectJailbreak(text string) (JailbreakResult, error)
	// IsJailbreakEnabled returns whether jailbreak detection is enabled
	IsJailbreakEnabled() bool
}

// SemanticRouterClassifier combines all three classifiers into one interface
type SemanticRouterClassifier interface {
	Classifier
	PIIDetector
	JailbreakDetector

	// ProcessAll runs all enabled classifiers and returns combined results
	ProcessAll(text string) (CategoryResult, PIIResult, JailbreakResult, error)
}
