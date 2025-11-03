package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"

// classifyAndSelectBestModel chooses best models based on category classification and model quality and expected TTFT
func (r *OpenAIRouter) classifyAndSelectBestModel(query string) string {
	return r.Classifier.ClassifyAndSelectBestModel(query)
}

// findCategoryForClassification determines the category for the given text using classification
func (r *OpenAIRouter) findCategoryForClassification(query string) string {
	if len(r.CategoryDescriptions) == 0 {
		return ""
	}

	categoryName, _, err := r.Classifier.ClassifyCategory(query)
	if err != nil {
		logging.Errorf("Category classification error: %v", err)
		return ""
	}

	return categoryName
}
