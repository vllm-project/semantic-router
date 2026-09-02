package config

import "fmt"

// ModalityDetectionMethod defines how modality is detected
const (
	// ModalityDetectionClassifier uses the mmBERT-32K ML classifier (3-class: AR/DIFFUSION/BOTH)
	ModalityDetectionClassifier = "classifier"
	// ModalityDetectionKeyword uses keyword pattern matching (2-class: AR/DIFFUSION)
	ModalityDetectionKeyword = "keyword"
	// ModalityDetectionHybrid uses classifier first, keyword as fallback/confirmation (default)
	ModalityDetectionHybrid = "hybrid"
)

// ModalityDetectorConfig configures the modality signal detector.
// Lives in InlineModels alongside hallucination_mitigation and feedback_detector.
// The detector classifies user prompts into AR / DIFFUSION / BOTH signals.
type ModalityDetectorConfig struct {
	// Enabled activates the modality detector. When false, modality signals are not evaluated.
	Enabled bool `yaml:"enabled" json:"enabled"`

	// PromptPrefixes are prefix strings stripped from the user prompt before
	// sending it to the generation backend (e.g. "generate an image of ", "draw ").
	// Matched case-insensitively; the first match is stripped. Optional.
	PromptPrefixes []string `yaml:"prompt_prefixes,omitempty" json:"prompt_prefixes,omitempty"`

	// Detection configuration (inlined from ModalityDetectionConfig)
	ModalityDetectionConfig `yaml:",inline"`
}

// ModalityDetectionConfig configures how modality routing detects whether a prompt
// should be routed to an AR (text) model, a Diffusion (image) model, or both.
type ModalityDetectionConfig struct {
	// Method specifies the detection strategy: "classifier", "keyword", or "hybrid" (default).
	//   - "classifier": Use mmBERT-32K ML classifier only — requires classifier.model_path
	//   - "keyword":    Use keyword pattern matching only — requires keywords list
	//   - "hybrid":     Classifier primary + keyword fallback — requires at least one of the above
	Method string `json:"method,omitempty" yaml:"method,omitempty"`

	// Classifier configuration (required when method is "classifier"; recommended for "hybrid")
	Classifier *ModalityClassifierConfig `json:"classifier,omitempty" yaml:"classifier,omitempty"`

	// Keywords for image generation detection (required when method is "keyword"; recommended for "hybrid")
	// These are matched case-insensitively against the user prompt.
	Keywords []string `json:"keywords,omitempty" yaml:"keywords,omitempty"`

	// BothKeywords are additional keywords that indicate the user wants BOTH text and image.
	// These are matched case-insensitively when the prompt also contains image keywords.
	// Examples: "explain and illustrate", "describe with a picture", "write ... with an image"
	BothKeywords []string `json:"both_keywords,omitempty" yaml:"both_keywords,omitempty"`

	// ConfidenceThreshold is the minimum classifier confidence to accept its prediction.
	// Below this threshold, the system falls back to keyword detection (hybrid mode).
	// Required when method is "classifier" or "hybrid".
	ConfidenceThreshold float32 `json:"confidence_threshold,omitempty" yaml:"confidence_threshold,omitempty"`

	// LowerThresholdRatio controls the hybrid mode disagreement behavior.
	// When classifier and keyword methods disagree, the classifier is still preferred
	// if its confidence >= confidence_threshold * lower_threshold_ratio.
	// Required when method is "hybrid". Typical value: 0.7 (i.e. 70% of confidence_threshold).
	LowerThresholdRatio float32 `json:"lower_threshold_ratio,omitempty" yaml:"lower_threshold_ratio,omitempty"`
}

// ModalityClassifierConfig configures the ML-based modality classifier
type ModalityClassifierConfig struct {
	// ModelPath is the path to the merged modality classifier model directory.
	// Required when method is "classifier" or "hybrid" with a classifier.
	ModelPath string `json:"model_path,omitempty" yaml:"model_path,omitempty"`

	// UseCPU forces CPU inference even when GPU is available.
	UseCPU bool `json:"use_cpu,omitempty" yaml:"use_cpu,omitempty"`
}

// GetMethod returns the configured modality detection method.
// Returns empty string if not set — callers should use Validate() to ensure
// Method is one of "classifier", "keyword", or "hybrid" before calling this.
func (c *ModalityDetectionConfig) GetMethod() string {
	if c == nil {
		return ""
	}
	return c.Method
}

// GetConfidenceThreshold returns the configured confidence threshold.
// This value must be explicitly set in config when method is "classifier" or "hybrid";
// Validate() enforces this requirement.
func (c *ModalityDetectionConfig) GetConfidenceThreshold() float32 {
	if c == nil {
		return 0
	}
	return c.ConfidenceThreshold
}

// GetLowerThresholdRatio returns the configured lower threshold ratio.
// This value must be explicitly set in config when method is "hybrid";
// Validate() enforces this requirement.
func (c *ModalityDetectionConfig) GetLowerThresholdRatio() float32 {
	if c == nil {
		return 0
	}
	return c.LowerThresholdRatio
}

// Validate validates the modality detection configuration.
// It ensures that:
//   - Method (if set) is one of "classifier", "keyword", or "hybrid"
//   - For "classifier": Classifier config with a non-empty model_path is required
//   - For "keyword": At least one keyword must be configured
//   - For "hybrid": At least one of Classifier or Keywords must be configured
//   - ConfidenceThreshold (if set) is in the range (0, 1]
//   - ConfidenceThreshold is required when method is "classifier" or "hybrid"
func (c *ModalityDetectionConfig) Validate() error {
	if c == nil {
		return nil // nil config is valid (not referenced by any signal when unset)
	}

	method := c.GetMethod()
	if err := validateModalityDetectionMethod(method); err != nil {
		return err
	}
	if err := c.validateMethodRequirements(method); err != nil {
		return err
	}
	return c.validateThresholds(method)
}

func validateModalityDetectionMethod(method string) error {
	if method == "" {
		return fmt.Errorf("modality_detection.method is required (one of %q, %q, or %q)",
			ModalityDetectionClassifier, ModalityDetectionKeyword, ModalityDetectionHybrid)
	}
	if method != ModalityDetectionClassifier && method != ModalityDetectionKeyword && method != ModalityDetectionHybrid {
		return fmt.Errorf("modality_detection.method must be one of %q, %q, or %q, got %q",
			ModalityDetectionClassifier, ModalityDetectionKeyword, ModalityDetectionHybrid, method)
	}
	return nil
}

func (c *ModalityDetectionConfig) validateMethodRequirements(method string) error {
	switch method {
	case ModalityDetectionClassifier:
		if c.Classifier == nil || c.Classifier.ModelPath == "" {
			return fmt.Errorf("modality_detection: method %q requires classifier.model_path to be set", method)
		}

	case ModalityDetectionKeyword:
		if len(c.Keywords) == 0 {
			return fmt.Errorf("modality_detection: method %q requires at least one entry in keywords", method)
		}

	case ModalityDetectionHybrid:
		hasClassifier := c.Classifier != nil && c.Classifier.ModelPath != ""
		hasKeywords := len(c.Keywords) > 0
		if !hasClassifier && !hasKeywords {
			return fmt.Errorf("modality_detection: method %q requires at least one of classifier.model_path or keywords to be configured", method)
		}
	}
	return nil
}

func (c *ModalityDetectionConfig) validateThresholds(method string) error {
	if c.ConfidenceThreshold != 0 && (c.ConfidenceThreshold < 0 || c.ConfidenceThreshold > 1) {
		return fmt.Errorf("modality_detection.confidence_threshold must be between 0 and 1, got %.4f", c.ConfidenceThreshold)
	}
	if (method == ModalityDetectionClassifier || method == ModalityDetectionHybrid) && c.ConfidenceThreshold == 0 {
		return fmt.Errorf("modality_detection.confidence_threshold is required when method is %q (e.g. 0.6)", method)
	}
	if c.LowerThresholdRatio != 0 && (c.LowerThresholdRatio < 0 || c.LowerThresholdRatio > 1) {
		return fmt.Errorf("modality_detection.lower_threshold_ratio must be between 0 and 1, got %.4f", c.LowerThresholdRatio)
	}
	if method == ModalityDetectionHybrid && c.LowerThresholdRatio == 0 {
		return fmt.Errorf("modality_detection.lower_threshold_ratio is required when method is %q (e.g. 0.7)", method)
	}

	return nil
}
