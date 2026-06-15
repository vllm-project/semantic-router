package classification

import (
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func (c *ComplexityClassifier) classifyRuleWithEmbeddings(
	rule config.ComplexityRule,
	queryEmbeddings complexityQueryEmbeddings,
	scoreOptions prototypeScoreOptions,
) ComplexityRuleResult {
	textHardScore, textEasyScore, textSignal := c.scoreTextSignal(rule.Name, queryEmbeddings.text, scoreOptions)
	imageHardScore, imageEasyScore, imageSignal, hasImage := c.scoreImageSignal(rule.Name, queryEmbeddings, scoreOptions)
	fusedSignal, signalSource := selectComplexitySignal(textSignal, imageSignal, hasImage)

	return ComplexityRuleResult{
		RuleName:       rule.Name,
		Difficulty:     classifyComplexityDifficulty(rule.Threshold, fusedSignal),
		TextHardScore:  textHardScore.Score,
		TextEasyScore:  textEasyScore.Score,
		TextMargin:     textSignal,
		ImageHardScore: imageHardScore.Score,
		ImageEasyScore: imageEasyScore.Score,
		ImageMargin:    imageSignal,
		FusedMargin:    fusedSignal,
		Confidence:     math.Abs(fusedSignal),
		SignalSource:   signalSource,
	}
}

func (c *ComplexityClassifier) scoreTextSignal(
	ruleName string,
	queryEmbedding []float32,
	scoreOptions prototypeScoreOptions,
) (prototypeBankScore, prototypeBankScore, float64) {
	hardScore := c.hardPrototypeBanks[ruleName].score(queryEmbedding, scoreOptions)
	easyScore := c.easyPrototypeBanks[ruleName].score(queryEmbedding, scoreOptions)
	return hardScore, easyScore, hardScore.Score - easyScore.Score
}

func (c *ComplexityClassifier) scoreImageSignal(
	ruleName string,
	queryEmbeddings complexityQueryEmbeddings,
	scoreOptions prototypeScoreOptions,
) (prototypeBankScore, prototypeBankScore, float64, bool) {
	queryEmbedding := queryEmbeddings.image
	if queryEmbedding == nil {
		queryEmbedding = queryEmbeddings.mmText
	}
	if queryEmbedding == nil {
		return prototypeBankScore{}, prototypeBankScore{}, 0, false
	}

	hardScore := c.imageHardPrototypeBanks[ruleName].score(queryEmbedding, scoreOptions)
	easyScore := c.imageEasyPrototypeBanks[ruleName].score(queryEmbedding, scoreOptions)
	if hardScore.PrototypeCount == 0 || easyScore.PrototypeCount == 0 {
		return hardScore, easyScore, 0, false
	}

	return hardScore, easyScore, hardScore.Score - easyScore.Score, true
}

func selectComplexitySignal(textSignal float64, imageSignal float64, hasImage bool) (float64, string) {
	if hasImage && math.Abs(imageSignal) > math.Abs(textSignal) {
		return imageSignal, "image"
	}
	return textSignal, "text"
}

func classifyComplexityDifficulty(threshold float32, signal float64) string {
	if signal > float64(threshold) {
		return "hard"
	}
	if signal < -float64(threshold) {
		return "easy"
	}
	return "medium"
}

func logComplexityRuleResult(
	rule config.ComplexityRule,
	result ComplexityRuleResult,
	requestImageProvided bool,
) {
	if result.SignalSource == "image" || result.ImageHardScore > 0 || result.ImageEasyScore > 0 {
		logging.Infof(
			"Complexity rule '%s': text_signal=%.3f, image_signal=%.3f (src=%s), fused=%s(%.3f), difficulty=%s",
			rule.Name,
			result.TextMargin,
			result.ImageMargin,
			complexityImageSourceLabel(requestImageProvided),
			result.SignalSource,
			result.FusedMargin,
			result.Difficulty,
		)
		return
	}
	logging.Infof(
		"Complexity rule '%s': hard_score=%.3f, easy_score=%.3f, signal=%.3f, difficulty=%s",
		rule.Name,
		result.TextHardScore,
		result.TextEasyScore,
		result.FusedMargin,
		result.Difficulty,
	)
}

func complexityImageSourceLabel(requestImageProvided bool) string {
	if requestImageProvided {
		return "screenshot"
	}
	return "mm_text"
}
