package classification

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type JailbreakInitializer interface {
	Init(modelID string, useCPU bool, numClasses ...int) error
}

type JailbreakInitializerImpl struct {
	usedModernBERT bool // Track which init path succeeded for inference routing
}

func (c *JailbreakInitializerImpl) Init(modelID string, useCPU bool, numClasses ...int) error {
	// A ModernBERT config.json is incompatible with the traditional Candle BERT
	// loader, so probing Candle first logs alarming "Failed to initialize" errors
	// before the ModernBERT fallback succeeds (#2096). When the model is detected
	// as ModernBERT, try the ModernBERT initializer first to skip that doomed probe.
	if isModernBertModel(modelID) {
		if err := candle_binding.InitModernBertJailbreakClassifier(modelID, useCPU); err == nil {
			c.usedModernBERT = true
			logging.ComponentEvent("classifier", "jailbreak_detector_initialized", map[string]interface{}{
				"backend":   "modernbert",
				"model_ref": modelID,
			})
			return nil
		}
		// Detected as ModernBERT but its initializer failed (e.g. a LoRA model
		// whose base is ModernBERT); fall through to the auto-detect path.
		logging.ComponentDebugEvent("classifier", "jailbreak_detector_fallback_enabled", map[string]interface{}{
			"fallback_backend": "candle_bert_auto",
			"model_ref":        modelID,
		})
	}

	// Try auto-detecting jailbreak classifier init - checks for lora_config.json.
	// This enables LoRA Jailbreak models when available. InitJailbreakClassifier routes
	// to LORA_JAILBREAK_CLASSIFIER or BERT_JAILBREAK_CLASSIFIER.
	err := candle_binding.InitJailbreakClassifier(modelID, numClasses[0], useCPU)
	if err == nil {
		c.usedModernBERT = false
		logging.ComponentEvent("classifier", "jailbreak_detector_initialized", map[string]interface{}{
			"backend":   "candle_bert_auto",
			"model_ref": modelID,
		})
		return nil
	}

	// Fallback to ModernBERT-specific init for backward compatibility.
	// This handles models with incomplete configs (missing hidden_act, etc.).
	logging.ComponentDebugEvent("classifier", "jailbreak_detector_fallback_enabled", map[string]interface{}{
		"fallback_backend": "modernbert",
		"model_ref":        modelID,
	})
	err = candle_binding.InitModernBertJailbreakClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize jailbreak classifier (both auto-detect and ModernBERT): %w", err)
	}
	c.usedModernBERT = true
	logging.ComponentEvent("classifier", "jailbreak_detector_initialized", map[string]interface{}{
		"backend":   "modernbert",
		"model_ref": modelID,
	})
	return nil
}

// createJailbreakInitializer creates the jailbreak initializer (auto-detecting).
func createJailbreakInitializer() JailbreakInitializer {
	return &JailbreakInitializerImpl{}
}

// MmBERT32KJailbreakInitializerImpl uses mmBERT-32K (YaRN RoPE, 32K context) for jailbreak detection.
type MmBERT32KJailbreakInitializerImpl struct {
	usedMmBERT32K bool
}

func (c *MmBERT32KJailbreakInitializerImpl) Init(modelID string, useCPU bool, numClasses ...int) error {
	logging.ComponentDebugEvent("classifier", "jailbreak_detector_backend_loading", map[string]interface{}{
		"backend":   "mmbert_32k",
		"model_ref": modelID,
	})
	err := candle_binding.InitMmBert32KJailbreakClassifier(modelID, useCPU)
	if err != nil {
		return fmt.Errorf("failed to initialize mmBERT-32K jailbreak detector: %w", err)
	}
	c.usedMmBERT32K = true
	logging.ComponentEvent("classifier", "jailbreak_detector_initialized", map[string]interface{}{
		"backend":   "mmbert_32k",
		"model_ref": modelID,
	})
	return nil
}

// createMmBERT32KJailbreakInitializer creates an mmBERT-32K jailbreak initializer.
func createMmBERT32KJailbreakInitializer() JailbreakInitializer {
	return &MmBERT32KJailbreakInitializerImpl{}
}

type JailbreakInference interface {
	Classify(text string) (candle_binding.ClassResult, error)
}

type JailbreakInferenceImpl struct{}

func (c *JailbreakInferenceImpl) Classify(text string) (candle_binding.ClassResult, error) {
	// Try jailbreak-specific classifier first, fall back to ModernBERT if it fails
	result, err := candle_binding.ClassifyJailbreakText(text)
	if err != nil {
		// Jailbreak classifier not initialized or failed, try ModernBERT
		return candle_binding.ClassifyModernBertJailbreakText(text)
	}
	return result, nil
}

// createJailbreakInferenceCandle creates Candle-based jailbreak inference (auto-detecting).
func createJailbreakInferenceCandle() JailbreakInference {
	return &JailbreakInferenceImpl{}
}

// MmBERT32KJailbreakInferenceImpl uses mmBERT-32K for jailbreak detection.
type MmBERT32KJailbreakInferenceImpl struct{}

func (c *MmBERT32KJailbreakInferenceImpl) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyMmBert32KJailbreak(text)
}

// createMmBERT32KJailbreakInference creates mmBERT-32K jailbreak inference.
func createMmBERT32KJailbreakInference() JailbreakInference {
	return &MmBERT32KJailbreakInferenceImpl{}
}

// createJailbreakInference creates the appropriate jailbreak inference based on configuration.
// Checks UseMmBERT32K and UseVLLM flags to decide between mmBERT-32K, vLLM, or Candle implementation.
// When UseMmBERT32K is true, uses mmBERT-32K (32K context, YaRN RoPE, multilingual).
// When UseVLLM is true, it will try to find external model config with role="guardrail".
func createJailbreakInference(promptGuardCfg *config.PromptGuardConfig, routerCfg *config.RouterConfig) (JailbreakInference, error) {
	// Check for mmBERT-32K first (takes precedence)
	if promptGuardCfg.UseMmBERT32K {
		logging.ComponentEvent("classifier", "jailbreak_detector_backend_selected", map[string]interface{}{
			"backend": "mmbert_32k",
		})
		return createMmBERT32KJailbreakInference(), nil
	}

	if promptGuardCfg.UseVLLM {
		// Try to find external model configuration with role="guardrail"
		externalCfg := routerCfg.FindExternalModelByRole(config.ModelRoleGuardrail)
		if externalCfg == nil {
			return nil, fmt.Errorf("external model with model_role='%s' is required when use_vllm=true", config.ModelRoleGuardrail)
		}

		// Validate required fields
		if externalCfg.ModelEndpoint.Address == "" {
			return nil, fmt.Errorf("external guardrail model endpoint address is required")
		}
		if externalCfg.ModelName == "" {
			return nil, fmt.Errorf("external guardrail model name is required")
		}

		logging.ComponentEvent("classifier", "jailbreak_detector_backend_selected", map[string]interface{}{
			"backend":  "external_guardrail",
			"provider": externalCfg.Provider,
		})

		// Use vLLM-based inference with external config.
		// Pass default threshold from PromptGuardConfig.
		return NewVLLMJailbreakInference(externalCfg, promptGuardCfg.Threshold)
	}
	// Use Candle-based inference.
	return createJailbreakInferenceCandle(), nil
}

// JailbreakDetection represents the result of jailbreak analysis for a piece of content.
type JailbreakDetection struct {
	Content       string  `json:"content"`
	IsJailbreak   bool    `json:"is_jailbreak"`
	JailbreakType string  `json:"jailbreak_type"`
	Confidence    float32 `json:"confidence"`
	ContentIndex  int     `json:"content_index"`
}
