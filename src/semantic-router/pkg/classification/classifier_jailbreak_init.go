package classification

import (
	"context"
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

// SequenceClassificationResult is the classification-owned result contract
// every SequenceClassifierBackend returns: the full class-probability
// distribution, indexed the same way as JailbreakMapping. It deliberately
// does not carry a pre-computed argmax class/confidence - deriveArgmax
// derives that once in the policy layer (classifier_jailbreak_risk.go) from
// Probabilities, so no backend implements its own argmax logic and every
// backend (local Candle, mmBERT-32K, or a remote HTTP/generative model) is
// scored identically. This type belongs to the classification package, not
// candle_binding, so remote/generative backends never need to depend on a
// Candle FFI DTO to satisfy the interface.
type SequenceClassificationResult struct {
	Probabilities []float32
}

// deriveArgmax returns the index and score of the highest-probability class
// in a complete distribution. It is the single place argmax/confidence is
// computed for jailbreak classification, so every SequenceClassifierBackend
// only ever returns raw probabilities.
func deriveArgmax(probabilities []float32) (int, float32) {
	bestIdx := -1
	var bestScore float32
	for idx, p := range probabilities {
		if bestIdx == -1 || p > bestScore {
			bestIdx = idx
			bestScore = p
		}
	}
	return bestIdx, bestScore
}

// SequenceClassifierBackend is implemented by every jailbreak classification
// backend (local Candle, mmBERT-32K, or a remote model). It always returns the
// complete class-probability distribution, never an argmax-only result, so
// callers can read the probability of a specific class (e.g. jailbreak)
// directly instead of the confidence of whichever class wins argmax. ctx
// carries the caller's cancellation/deadline/tracing so a remote backend
// (http_chat, http_classify) can be cancelled with the request instead of
// always running to its own internal timeout.
type SequenceClassifierBackend interface {
	Classify(ctx context.Context, text string) (SequenceClassificationResult, error)
}

// candleResultToSequenceClassification drops the argmax fields Candle's FFI
// layer computes and keeps only the probability distribution, so local
// backends return the same classification-owned type as every other backend.
func candleResultToSequenceClassification(result candle_binding.ClassResultWithProbs) SequenceClassificationResult {
	return SequenceClassificationResult{Probabilities: result.Probabilities}
}

type JailbreakInferenceImpl struct{}

func (c *JailbreakInferenceImpl) Classify(_ context.Context, text string) (SequenceClassificationResult, error) {
	// Try jailbreak-specific classifier first, fall back to ModernBERT if it fails
	result, err := candle_binding.ClassifyJailbreakTextWithProbs(text)
	if err != nil {
		// Jailbreak classifier not initialized or failed, try ModernBERT
		result, err = candle_binding.ClassifyModernBertJailbreakTextWithProbs(text)
		if err != nil {
			return SequenceClassificationResult{}, err
		}
	}
	return candleResultToSequenceClassification(result), nil
}

// createJailbreakInferenceCandle creates Candle-based jailbreak inference (auto-detecting).
func createJailbreakInferenceCandle() SequenceClassifierBackend {
	return &JailbreakInferenceImpl{}
}

// MmBERT32KJailbreakInferenceImpl uses mmBERT-32K for jailbreak detection.
type MmBERT32KJailbreakInferenceImpl struct{}

func (c *MmBERT32KJailbreakInferenceImpl) Classify(_ context.Context, text string) (SequenceClassificationResult, error) {
	result, err := candle_binding.ClassifyMmBert32KJailbreakWithProbs(text)
	if err != nil {
		return SequenceClassificationResult{}, err
	}
	return candleResultToSequenceClassification(result), nil
}

// createMmBERT32KJailbreakInference creates mmBERT-32K jailbreak inference.
func createMmBERT32KJailbreakInference() SequenceClassifierBackend {
	return &MmBERT32KJailbreakInferenceImpl{}
}

// createJailbreakInference creates the appropriate jailbreak inference based on
// the configured prompt_guard.backend (candle, mmbert32k, http_chat, or
// http_classify). An empty/unset backend value reaching this switch falls
// back to candle - but a canonical-resolved config never actually reaches
// here empty: canonical defaults set backend to mmbert32k explicitly (see
// config.PromptGuardBackendCandle's doc comment). This fallback only fires
// for configs built without going through canonical resolution.
func createJailbreakInference(promptGuardCfg *config.PromptGuardConfig, routerCfg *config.RouterConfig, jailbreakMapping *JailbreakMapping) (SequenceClassifierBackend, error) {
	switch promptGuardCfg.Backend {
	case config.PromptGuardBackendMmBERT32K:
		logging.ComponentEvent("classifier", "jailbreak_detector_backend_selected", map[string]interface{}{
			"backend": config.PromptGuardBackendMmBERT32K,
		})
		return createMmBERT32KJailbreakInference(), nil

	case config.PromptGuardBackendHTTPChat:
		externalCfg, err := findGuardrailExternalModel(routerCfg)
		if err != nil {
			return nil, err
		}
		logging.ComponentEvent("classifier", "jailbreak_detector_backend_selected", map[string]interface{}{
			"backend":  config.PromptGuardBackendHTTPChat,
			"provider": externalCfg.Provider,
		})
		// Pass default threshold from PromptGuardConfig.
		return NewVLLMJailbreakInference(externalCfg, promptGuardCfg.Threshold, jailbreakMapping, promptGuardCfg.PositiveLabels)

	case config.PromptGuardBackendHTTPClassify:
		externalCfg, err := findGuardrailExternalModel(routerCfg)
		if err != nil {
			return nil, err
		}
		logging.ComponentEvent("classifier", "jailbreak_detector_backend_selected", map[string]interface{}{
			"backend":  config.PromptGuardBackendHTTPClassify,
			"provider": externalCfg.Provider,
		})
		return NewHTTPClassifierJailbreakInference(externalCfg, jailbreakMapping)

	default:
		// Empty/unset or "candle": use Candle-based inference.
		logging.ComponentEvent("classifier", "jailbreak_detector_backend_selected", map[string]interface{}{
			"backend": config.PromptGuardBackendCandle,
		})
		return createJailbreakInferenceCandle(), nil
	}
}

// findGuardrailExternalModel looks up and validates the external model
// configuration required by the http_chat/http_classify jailbreak backends.
func findGuardrailExternalModel(routerCfg *config.RouterConfig) (*config.ExternalModelConfig, error) {
	externalCfg := routerCfg.FindExternalModelByRole(config.ModelRoleGuardrail)
	if externalCfg == nil {
		return nil, fmt.Errorf("external model with model_role='%s' is required for this prompt_guard.backend", config.ModelRoleGuardrail)
	}
	if externalCfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("external guardrail model endpoint address is required")
	}
	if externalCfg.ModelName == "" {
		return nil, fmt.Errorf("external guardrail model name is required")
	}
	return externalCfg, nil
}

// JailbreakDetection represents the result of jailbreak analysis for a piece of content.
type JailbreakDetection struct {
	Content       string  `json:"content"`
	IsJailbreak   bool    `json:"is_jailbreak"`
	JailbreakType string  `json:"jailbreak_type"`
	Confidence    float32 `json:"confidence"`
	ContentIndex  int     `json:"content_index"`
}
