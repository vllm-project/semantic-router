package classification

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type PIIInitializer interface {
	Init(modelID string, useCPU bool, numClasses int) error
}

type PIIInitializerImpl struct {
	usedModernBERT bool // Track which init path succeeded for inference routing
}

func (c *PIIInitializerImpl) Init(modelID string, useCPU bool, numClasses int) error {
	// A ModernBERT config.json is incompatible with the traditional Candle BERT
	// loader, so probing Candle first logs alarming "Failed to initialize" errors
	// before the ModernBERT fallback succeeds (#2096). When the model is detected
	// as ModernBERT, try the ModernBERT initializer first to skip that doomed probe.
	if isModernBertModel(modelID) {
		if err := candle_binding.InitModernBertPIITokenClassifier(modelID, useCPU); err == nil {
			c.usedModernBERT = true
			logging.ComponentEvent("classifier", "pii_detector_initialized", map[string]interface{}{
				"backend":   "modernbert",
				"model_ref": modelID,
			})
			return nil
		}
		// Detected as ModernBERT but its initializer failed (e.g. a LoRA model
		// whose base is ModernBERT); fall through to the auto-detect path.
		logging.ComponentDebugEvent("classifier", "pii_detector_fallback_enabled", map[string]interface{}{
			"fallback_backend": "candle_bert_auto",
			"model_ref":        modelID,
		})
	}

	// Try auto-detecting Candle BERT init - checks for lora_config.json
	// This enables LoRA PII models when available
	success := candle_binding.InitCandleBertTokenClassifier(modelID, numClasses, useCPU)
	if success {
		c.usedModernBERT = false
		logging.ComponentEvent("classifier", "pii_detector_initialized", map[string]interface{}{
			"backend":   "candle_bert_auto",
			"model_ref": modelID,
		})
		return nil
	}

	// Fallback to ModernBERT-specific init for backward compatibility.
	// This handles models with incomplete configs (missing hidden_act, etc.).
	logging.ComponentDebugEvent("classifier", "pii_detector_fallback_enabled", map[string]interface{}{
		"fallback_backend": "modernbert",
		"model_ref":        modelID,
	})
	err := candle_binding.InitModernBertPIITokenClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize PII token classifier (both auto-detect and ModernBERT): %w", err)
	}
	c.usedModernBERT = true
	logging.ComponentEvent("classifier", "pii_detector_initialized", map[string]interface{}{
		"backend":   "modernbert",
		"model_ref": modelID,
	})
	return nil
}

// createPIIInitializer creates the PII initializer (auto-detecting).
func createPIIInitializer() PIIInitializer {
	return &PIIInitializerImpl{}
}

// MmBERT32KPIIInitializerImpl uses mmBERT-32K (YaRN RoPE, 32K context) for PII detection.
type MmBERT32KPIIInitializerImpl struct {
	usedMmBERT32K bool
}

func (c *MmBERT32KPIIInitializerImpl) Init(modelID string, useCPU bool, numClasses int) error {
	logging.ComponentDebugEvent("classifier", "pii_detector_backend_loading", map[string]interface{}{
		"backend":   "mmbert_32k",
		"model_ref": modelID,
	})
	err := candle_binding.InitMmBert32KPIIClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize mmBERT-32K PII detector: %w", err)
	}
	c.usedMmBERT32K = true
	logging.ComponentEvent("classifier", "pii_detector_initialized", map[string]interface{}{
		"backend":   "mmbert_32k",
		"model_ref": modelID,
	})
	return nil
}

// createMmBERT32KPIIInitializer creates an mmBERT-32K PII initializer.
func createMmBERT32KPIIInitializer() PIIInitializer {
	return &MmBERT32KPIIInitializerImpl{}
}

type PIIInference interface {
	ClassifyTokens(text string) (candle_binding.TokenClassificationResult, error)
}

type PIIInferenceImpl struct{}

func (c *PIIInferenceImpl) ClassifyTokens(text string) (candle_binding.TokenClassificationResult, error) {
	// Auto-detecting inference - uses whichever classifier was initialized (LoRA or Traditional)
	return candle_binding.ClassifyCandleBertTokens(text)
}

// createPIIInference creates the PII inference (auto-detecting).
func createPIIInference() PIIInference {
	return &PIIInferenceImpl{}
}

// MmBERT32KPIIInferenceImpl uses mmBERT-32K for PII token classification.
// Entity types are returned as "LABEL_{class_id}" by Rust and translated Go-side via PIIMapping.
type MmBERT32KPIIInferenceImpl struct{}

func (c *MmBERT32KPIIInferenceImpl) ClassifyTokens(text string) (candle_binding.TokenClassificationResult, error) {
	entities, err := candle_binding.ClassifyMmBert32KPII(text)
	if err != nil {
		return candle_binding.TokenClassificationResult{}, err
	}
	return candle_binding.TokenClassificationResult{Entities: entities}, nil
}

// createMmBERT32KPIIInference creates mmBERT-32K PII inference.
func createMmBERT32KPIIInference() PIIInference {
	return &MmBERT32KPIIInferenceImpl{}
}

// PIIDetection represents detected PII entities in content.
type PIIDetection struct {
	EntityType string  `json:"entity_type"` // Type of PII entity (e.g., "PERSON", "EMAIL", "PHONE")
	Start      int     `json:"start"`       // Start character position in original text
	End        int     `json:"end"`         // End character position in original text
	Text       string  `json:"text"`        // Actual entity text
	Confidence float32 `json:"confidence"`  // Confidence score (0.0 to 1.0)
}

// PIIAnalysisResult represents the result of PII analysis for content.
type PIIAnalysisResult struct {
	Content      string         `json:"content"`
	HasPII       bool           `json:"has_pii"`
	Entities     []PIIDetection `json:"entities"`
	ContentIndex int            `json:"content_index"`
}
