package extproc

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestShortRequestUsageDoesNotInflateLongContextSignal(t *testing.T) {
	classifier, err := classification.BuildClassifier(&config.RouterConfig{
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				ContextRules: []config.ContextRule{{
					Name:      "beyond_window",
					MinTokens: config.TokenCount("200K"),
				}},
			},
		},
	}, nil, nil, nil)
	require.NoError(t, err)
	router := &OpenAIRouter{Classifier: classifier}

	observe := func(textBytes, promptTokens int) {
		ctx := &RequestContext{VSRContextTextBytes: textBytes}
		ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: config.DefaultRecipeName})
		router.calibrateTokenEstimator(ctx, promptTokens)
	}
	// Chat traffic: provider prompt usage is mostly chat-template overhead.
	for i := 0; i < 200; i++ {
		observe(len("What is the capital of France?"), 25)
	}

	// About 55K real tokens: a production backend counted 41,827 prompt
	// tokens for 204,643 bytes of English prose.
	sentence := "The committee reviewed the quarterly budget and agreed to fund the new library wing next spring. "
	prompt := strings.Repeat(sentence, 55_000*204_643/41_827/len(sentence))
	results := classifier.EvaluateAllSignalsWithForceOption(prompt, true)

	assert.Empty(t, results.MatchedContextRules,
		"a ~55K-token prompt must not match a 200K context rule")
	assert.Less(t, results.TokenCount, 70_000)
}
