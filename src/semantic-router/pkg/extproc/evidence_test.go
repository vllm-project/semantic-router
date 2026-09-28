package extproc

import (
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const testIntelligenceIndex = "vllm-sr/test-intelligence@1.0.0"

func modelParamsWithTestQuality(score float64) config.ModelParams {
	scaled := score * 100
	return config.ModelParams{
		QualityIndex: testIntelligenceIndex,
		IndexResults: map[string]modelcatalog.IndexResult{
			testIntelligenceIndex: {
				Model:    "test-model",
				Index:    testIntelligenceIndex,
				Status:   "available",
				Score:    &scaled,
				Coverage: 1,
			},
		},
	}
}

func addTestQuality(params config.ModelParams, score float64) config.ModelParams {
	quality := modelParamsWithTestQuality(score)
	params.QualityIndex = quality.QualityIndex
	params.IndexResults = quality.IndexResults
	return params
}
