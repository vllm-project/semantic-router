package classification

import (
	"context"
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func (uc *UnifiedClassifier) classifyBatchRecipe(ctx context.Context, texts []string) (*UnifiedBatchResults, error) {
	if uc.testClassifyBatchLegacy != nil {
		return uc.testClassifyBatchLegacy(texts)
	}
	c := uc.recipeClassifier
	if c == nil {
		return nil, fmt.Errorf("%w: real recipe task bindings are required for unified classification", binding.ErrCapability)
	}
	result := &UnifiedBatchResults{BatchSize: len(texts), IntentResults: make([]IntentResult, len(texts)), PIIResults: make([]PIIResult, len(texts)), SecurityResults: make([]SecurityResult, len(texts))}
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		intent, err := c.categoryInference.ClassifyWithProbabilities(ctx, text)
		if err != nil {
			return nil, err
		}
		label, ok := c.CategoryMapping.GetCategoryFromIndex(intent.Class)
		if !ok {
			return nil, fmt.Errorf("unknown intent label index %d", intent.Class)
		}
		result.IntentResults[i] = IntentResult{Category: label, Confidence: intent.Confidence, Probabilities: append([]float32(nil), intent.Probabilities...)}
		pii, err := c.ClassifyPIIWithDetails(ctx, text)
		if err != nil {
			return nil, err
		}
		types := map[string]bool{}
		score := float32(0)
		hasScore := len(pii) > 0
		for _, d := range pii {
			types[d.EntityType] = true
			if d.Confidence > score {
				score = d.Confidence
			}
		}
		detected := make([]string, 0, len(types))
		for label := range types {
			detected = append(detected, label)
		}
		sort.Strings(detected)
		result.PIIResults[i] = PIIResult{HasPII: len(detected) > 0, PIITypes: detected, Confidence: score, ScoresAvailable: &hasScore}
		verdict, err := c.CheckForJailbreakVerdict(ctx, text, c.Config.PromptGuard.Threshold)
		if err != nil {
			return nil, err
		}
		available := verdict.Confidence != nil
		security := SecurityResult{IsJailbreak: verdict.Detected, ThreatType: verdict.Label, Decision: verdict.Decision, ScoresAvailable: &available}
		if available {
			security.Confidence = *verdict.Confidence
		}
		result.SecurityResults[i] = security
	}
	return result, nil
}
