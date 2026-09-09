package classification

import (
	"fmt"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ComplexityInitializer loads the trained complexity classifier model.
type ComplexityInitializer interface {
	Init(modelID string, useCPU bool) error
}

// ComplexityInference runs inference against the trained complexity classifier and
// returns the predicted difficulty class index and confidence.
type ComplexityInference interface {
	Classify(text string) (candle_binding.ClassResult, error)
}

// ComplexityInitializerImpl initializes the Candle ModernBERT complexity classifier.
type ComplexityInitializerImpl struct{}

func (c *ComplexityInitializerImpl) Init(modelID string, useCPU bool) error {
	if err := candle_binding.InitComplexityClassifier(modelID, useCPU); err != nil {
		return fmt.Errorf("failed to initialize complexity classifier: %w", err)
	}
	logging.ComponentEvent("classifier", "complexity_classifier_backend_initialized", map[string]interface{}{
		"backend":   "candle_modernbert",
		"model_ref": modelID,
	})
	return nil
}

// createComplexityInitializer creates the Candle complexity initializer.
func createComplexityInitializer() ComplexityInitializer {
	return &ComplexityInitializerImpl{}
}

// ComplexityInferenceImpl runs Candle ModernBERT complexity inference.
type ComplexityInferenceImpl struct{}

func (c *ComplexityInferenceImpl) Classify(text string) (candle_binding.ClassResult, error) {
	return candle_binding.ClassifyComplexityText(text)
}

// createComplexityInference creates the Candle complexity inference.
func createComplexityInference() ComplexityInference {
	return &ComplexityInferenceImpl{}
}
