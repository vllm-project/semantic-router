package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// extractUserAndNonUserContent extracts user and non-user messages from the request

// performClassificationAndModelSelection performs classification and model selection once
// Returns (categoryName, confidence, reasoningDecision, selectedModel)
func (r *OpenAIRouter) performClassificationAndModelSelection(originalModel string, userContent string, nonUserMessages []string) (string, float64, entropy.ReasoningDecision, string) {
	var categoryName string
	var classificationConfidence float64
	var reasoningDecision entropy.ReasoningDecision
	var selectedModel string

	// Only perform classification for auto models with content
	if !r.Config.IsAutoModelName(originalModel) {
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	if len(nonUserMessages) == 0 && userContent == "" {
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// Determine text to use for classification
	classificationText := userContent
	if classificationText == "" && len(nonUserMessages) > 0 {
		classificationText = strings.Join(nonUserMessages, " ")
	}

	if classificationText == "" {
		return "", 0.0, entropy.ReasoningDecision{}, ""
	}

	// Perform entropy-based classification once
	catName, confidence, reasoningDec, err := r.Classifier.ClassifyCategoryWithEntropy(classificationText)
	if err != nil {
		logging.Errorf("Entropy-based classification error: %v, using empty category", err)
		categoryName = ""
		classificationConfidence = 0.0
		reasoningDecision = entropy.ReasoningDecision{}
	} else {
		categoryName = catName
		classificationConfidence = confidence
		reasoningDecision = reasoningDec
		logging.Infof("Classification Result: category=%s, confidence=%.3f, reasoning=%v",
			categoryName, classificationConfidence, reasoningDecision.UseReasoning)
	}

	// Select best model for this category
	if categoryName != "" {
		selectedModel = r.Classifier.SelectBestModelForCategory(categoryName)
		logging.Infof("Selected model for category %s: %s", categoryName, selectedModel)
	} else {
		// No category found, use default model
		selectedModel = r.Config.DefaultModel
		logging.Infof("No category classified, using default model: %s", selectedModel)
	}

	return categoryName, classificationConfidence, reasoningDecision, selectedModel
}
