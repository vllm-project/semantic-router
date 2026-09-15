package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Default feedback type labels (used as fallback if config.json doesn't have id2label)
const (
	FeedbackLabelSatisfied         = "satisfied"
	FeedbackLabelNeedClarification = "need_clarification"
	FeedbackLabelWrongAnswer       = "wrong_answer"
	FeedbackLabelWantDifferent     = "want_different"
)

// FeedbackResult represents the result of user feedback classification
type FeedbackResult struct {
	FeedbackType        string  `json:"feedback_type"` // feedback type label from model's id2label
	Confidence          float32 `json:"confidence"`
	ConfidenceAvailable bool    `json:"confidence_available"`
	PolicyDefault       string  `json:"policy_default,omitempty"`
	Class               int     `json:"class"` // class index from model
}

// FeedbackMapping maps feedback types to class indices
type FeedbackMapping struct {
	LabelToIdx map[string]int
	IdxToLabel map[string]string
}

// FeedbackDetector handles user feedback classification from follow-up messages
type FeedbackDetector struct {
	backend     *ownedSequenceBackend
	config      *config.FeedbackDetectorConfig
	mapping     *FeedbackMapping
	initialized bool
	gate        admission.Admissioner
	mu          sync.RWMutex
}

// SetAdmissioner installs the deployment's admission gate for model inference.
func (d *FeedbackDetector) SetAdmissioner(gate admission.Admissioner) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.gate = gate
}

// NewFeedbackDetector creates a new feedback detector
func NewFeedbackDetector(cfg *config.FeedbackDetectorConfig, models ...*classifierModelRuntime) (*FeedbackDetector, error) {
	if cfg == nil {
		return nil, nil // Disabled
	}

	runtime := consumerModelRuntime(models)
	adapter := "modernbert"
	if cfg.UseMmBERT32K {
		adapter = "mmbert32k"
	}
	spec := runtime.localSpec("feedback_detector", cfg.ModelID, adapter, config.RemoteClassifierContractLabelDistribution, cfg.UseCPU)
	detector := &FeedbackDetector{
		backend: &ownedSequenceBackend{runtime: runtime.runtime, spec: spec},
		config:  cfg,
	}

	return detector, nil
}

type feedbackMappingFile struct {
	IdxToLabel map[string]string `json:"idx_to_label"`
	LabelToIdx map[string]int    `json:"label_to_idx"`
	ID2Label   map[string]string `json:"id2label"`
	Label2ID   map[string]int    `json:"label2id"`
}

func (d *FeedbackDetector) loadMapping(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read feedback mapping %s: %w", path, err)
	}
	var file feedbackMappingFile
	if err := json.Unmarshal(data, &file); err != nil {
		return fmt.Errorf("failed to parse feedback mapping %s: %w", path, err)
	}
	idxToLabel, labelToIdx := file.IdxToLabel, file.LabelToIdx
	if len(idxToLabel) == 0 {
		idxToLabel, labelToIdx = file.ID2Label, file.Label2ID
	}
	if len(idxToLabel) == 0 {
		return fmt.Errorf("feedback mapping %s declares no labels", path)
	}
	if err := ValidateLabelMappingAgainstModelConfig(path, d.config.ModelID, idxToLabel); err != nil {
		return err
	}
	d.mapping = &FeedbackMapping{
		LabelToIdx: make(map[string]int, len(labelToIdx)),
		IdxToLabel: make(map[string]string, len(idxToLabel)),
	}
	for idx, label := range idxToLabel {
		d.mapping.IdxToLabel[idx] = normalizeFeedbackLabel(label)
	}
	for label, idx := range labelToIdx {
		d.mapping.LabelToIdx[normalizeFeedbackLabel(label)] = idx
	}
	logging.ComponentEvent("classifier", "feedback_mapping_loaded", map[string]interface{}{
		"labels":       len(d.mapping.IdxToLabel),
		"mapping_path": path,
	})
	return nil
}

// normalizeFeedbackLabel converts model labels (e.g., "SAT", "NEED_CLARIFICATION") to standard form
func normalizeFeedbackLabel(label string) string {
	switch strings.ToUpper(label) {
	case "SAT", "SATISFIED":
		return FeedbackLabelSatisfied
	case "NEED_CLARIFICATION":
		return FeedbackLabelNeedClarification
	case "WRONG_ANSWER":
		return FeedbackLabelWrongAnswer
	case "WANT_DIFFERENT":
		return FeedbackLabelWantDifferent
	default:
		return strings.ToLower(label)
	}
}

// Initialize initializes the feedback detector with the ModernBERT model
func (d *FeedbackDetector) Initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.initialized {
		return nil
	}

	// Initialize ML model - ModelID is required
	if d.config.ModelID == "" {
		return fmt.Errorf("feedback detector requires ModelID to be configured")
	}

	mappingPath := d.config.FeedbackMappingPath
	if mappingPath == "" {
		mappingPath = filepath.Join(d.config.ModelID, "config.json")
	}
	if err := d.loadMapping(mappingPath); err != nil {
		return err
	}

	d.backend.labels = indexedNativeLabels(d.mapping.IdxToLabel)
	d.backend.normalizeLabel = normalizeFeedbackLabel
	if err := d.backend.Init(d.config.ModelID, d.config.UseCPU, len(d.mapping.IdxToLabel)); err != nil {
		return err
	}

	d.initialized = true
	logging.ComponentEvent("classifier", "feedback_detector_initialized", map[string]interface{}{
		"backend":   d.backend.spec.Deployment.Provider,
		"model_ref": d.config.ModelID,
	})

	return nil
}

// Classify determines user feedback type from follow-up message using the ML model
func (d *FeedbackDetector) Classify(ctx context.Context, text string) (*FeedbackResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if !d.initialized {
		return nil, fmt.Errorf("feedback detector not initialized")
	}

	if text == "" {
		return &FeedbackResult{
			FeedbackType:  FeedbackLabelSatisfied,
			PolicyDefault: "empty_text",
			Class:         0,
		}, nil
	}

	result, err := admitModelInference(ctx, d.gate, admissionDeploymentFeedbackDetector, func() (tasks.ClassResultWithProbs, error) {
		distribution, err := d.backend.Classify(ctx, text)
		if err != nil {
			return tasks.ClassResultWithProbs{}, err
		}
		class, confidence := deriveArgmax(distribution.Probabilities)
		return tasks.ClassResultWithProbs{Class: class, Confidence: confidence, Probabilities: distribution.Probabilities, NumClasses: len(distribution.Probabilities)}, nil
	})
	if err != nil {
		return nil, fmt.Errorf("feedback detection failed: %w", err)
	}

	// Get feedback type from mapping loaded from config.json
	feedbackType := d.mapping.IdxToLabel[fmt.Sprintf("%d", result.Class)]
	if feedbackType == "" {
		feedbackType = FeedbackLabelSatisfied // Default fallback
	}

	// Apply threshold check
	threshold := d.config.Threshold
	if threshold <= 0 {
		threshold = 0.5 // Default threshold
	}
	feedbackType, confidence := d.applyThreshold(feedbackType, result, threshold)

	logging.Debugf("Feedback detection: text_len=%d, feedback_type=%s, confidence=%.3f",
		len(text), feedbackType, confidence)

	return &FeedbackResult{
		FeedbackType:        feedbackType,
		Confidence:          confidence,
		ConfidenceAvailable: true,
		Class:               result.Class,
	}, nil
}

// applyThreshold decides what a prediction below the configured threshold is
// reported as.
//
// A prediction the model is not confident about is treated as uncertain and
// reported as satisfied, so the confidence beside that label has to be
// P(satisfied). The detector has four classes, so 1 - confidence is the mass on
// the other three and overstates satisfaction by whatever the two rejected
// classes hold. The satisfied index comes from the same loaded mapping the label
// above is read through. When that mapping names no satisfied class, or the model
// returned no probability for it, the model's own prediction is kept, since a
// satisfied reading nothing supports is the defect this replaces.
func (d *FeedbackDetector) applyThreshold(
	feedbackType string, result tasks.ClassResultWithProbs, threshold float32,
) (string, float32) {
	if result.Confidence >= threshold {
		return feedbackType, result.Confidence
	}
	for idx, label := range d.mapping.IdxToLabel {
		if label != FeedbackLabelSatisfied {
			continue
		}
		parsed, err := strconv.Atoi(idx)
		if err != nil || parsed < 0 || parsed >= len(result.Probabilities) {
			continue
		}
		return FeedbackLabelSatisfied, result.Probabilities[parsed]
	}
	return feedbackType, result.Confidence
}

// IsInitialized returns whether the detector is initialized
func (d *FeedbackDetector) IsInitialized() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.initialized
}

// GetMapping returns the feedback mapping
func (d *FeedbackDetector) GetMapping() *FeedbackMapping {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.mapping
}

func (d *FeedbackDetector) Close() error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.initialized = false
	if d.backend != nil {
		return d.backend.Close()
	}
	return nil
}
