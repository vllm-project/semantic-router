package classification

import (
	"context"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FactCheckResult represents the result of fact-check classification
type FactCheckResult struct {
	NeedsFactCheck      bool    `json:"needs_fact_check"`
	Confidence          float32 `json:"confidence"`
	ConfidenceAvailable bool    `json:"confidence_available"`
	PolicyDefault       string  `json:"policy_default,omitempty"`
	Label               string  `json:"label"` // "FACT_CHECK_NEEDED" or "NO_FACT_CHECK_NEEDED"
}

// FactCheckClassifier handles fact-check classification to determine if a prompt
// requires external factual verification using the halugate-sentinel ML model
type FactCheckClassifier struct {
	backend     *ownedSequenceBackend
	config      *config.FactCheckModelConfig
	mapping     *FactCheckMapping
	initialized bool
	gate        admission.Admissioner
	mu          sync.RWMutex
}

// SetAdmissioner installs the deployment's admission gate for model inference.
func (c *FactCheckClassifier) SetAdmissioner(gate admission.Admissioner) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.gate = gate
}

// NewFactCheckClassifier creates a new fact-check classifier
func NewFactCheckClassifier(cfg *config.FactCheckModelConfig, models ...*classifierModelRuntime) (*FactCheckClassifier, error) {
	if cfg == nil {
		return nil, nil // Disabled
	}

	runtime := consumerModelRuntime(models)
	adapter := "modernbert"
	if cfg.UseMmBERT32K {
		adapter = "mmbert32k"
	}
	spec := runtime.localSpec("fact_check_classifier", cfg.ModelID, adapter, config.RemoteClassifierContractLabelDistribution, cfg.UseCPU)
	classifier := &FactCheckClassifier{
		backend: &ownedSequenceBackend{runtime: runtime.runtime, spec: spec},
		config:  cfg,
	}

	return classifier, nil
}

// Initialize initializes the fact-check classifier with the halugate-sentinel ML model
func (c *FactCheckClassifier) Initialize() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.initialized {
		return nil
	}

	// Use default mapping (no external mapping file needed)
	c.mapping = &FactCheckMapping{
		LabelToIdx: map[string]int{
			FactCheckLabelNotNeeded: 0,
			FactCheckLabelNeeded:    1,
		},
		IdxToLabel: map[string]string{
			"0": FactCheckLabelNotNeeded,
			"1": FactCheckLabelNeeded,
		},
	}

	// Initialize ML model - ModelID is required
	if c.config.ModelID == "" {
		return fmt.Errorf("fact-check classifier requires ModelID to be configured")
	}

	c.backend.labels = indexedNativeLabels(c.mapping.IdxToLabel)
	if err := c.backend.Init(c.config.ModelID, c.config.UseCPU, 2); err != nil {
		return err
	}

	c.initialized = true
	logging.ComponentEvent("classifier", "fact_check_classifier_initialized", map[string]interface{}{
		"backend":   c.backend.spec.Deployment.Provider,
		"model_ref": c.config.ModelID,
	})

	return nil
}

// Classify determines if a prompt needs fact-checking using the ML model
func (c *FactCheckClassifier) Classify(ctx context.Context, text string) (*FactCheckResult, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.initialized {
		return nil, fmt.Errorf("fact-check classifier not initialized")
	}

	if text == "" {
		return &FactCheckResult{
			NeedsFactCheck: false,
			PolicyDefault:  "empty_text",
			Label:          FactCheckLabelNotNeeded,
		}, nil
	}

	result, err := admitModelInference(ctx, c.gate, admissionDeploymentFactCheckClassifier, func() (tasks.ClassResultWithProbs, error) {
		distribution, err := c.backend.Classify(ctx, text)
		if err != nil {
			return tasks.ClassResultWithProbs{}, err
		}
		class, confidence := deriveArgmax(distribution.Probabilities)
		return tasks.ClassResultWithProbs{Class: class, Confidence: confidence, Probabilities: distribution.Probabilities, NumClasses: len(distribution.Probabilities)}, nil
	})
	if err != nil {
		return nil, fmt.Errorf("fact-check ML classification failed: %w", err)
	}

	// Model outputs: 0=NO_FACT_CHECK_NEEDED, 1=FACT_CHECK_NEEDED
	needsFactCheck := result.Class == 1
	confidence := result.Confidence

	var label string
	if needsFactCheck {
		label = FactCheckLabelNeeded
	} else {
		label = FactCheckLabelNotNeeded
	}

	// Apply threshold check
	threshold := c.config.Threshold
	if threshold <= 0 {
		threshold = 0.7 // Default threshold
	}

	// Only mark as needing fact-check if confidence exceeds threshold
	if needsFactCheck && confidence < threshold {
		// Below threshold, flip decision
		needsFactCheck = false
		label = FactCheckLabelNotNeeded
		confidence = result.Probabilities[0] // Report the actual probability of the policy-selected label.
	}

	logging.Debugf("Fact-check ML classification: text_len=%d, needs_fact_check=%v, confidence=%.3f",
		len(text), needsFactCheck, confidence)

	return &FactCheckResult{
		NeedsFactCheck:      needsFactCheck,
		Confidence:          confidence,
		ConfidenceAvailable: true,
		Label:               label,
	}, nil
}

// IsInitialized returns whether the classifier is initialized
func (c *FactCheckClassifier) IsInitialized() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.initialized
}

// GetMapping returns the fact-check mapping
func (c *FactCheckClassifier) GetMapping() *FactCheckMapping {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.mapping
}

func (c *FactCheckClassifier) Close() error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.initialized = false
	if c.backend != nil {
		return c.backend.Close()
	}
	return nil
}
