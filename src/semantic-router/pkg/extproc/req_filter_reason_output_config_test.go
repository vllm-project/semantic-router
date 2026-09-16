package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestOutputConfigEffortReasoningTransport(t *testing.T) {
	router := newReasoningRouter(
		config.ReasoningConfig{
			ReasoningFamilies: map[string]config.ReasoningFamilyConfig{
				"claude": {
					Type: config.ReasoningFamilyTypeReasoningEffort, Parameter: "effort",
					Levels: []string{"low", "medium", "high", "xhigh", "max"}, Default: "high",
				},
			},
		},
		[]config.Decision{reasoningDecision("deep", "", 0, "claude-model", boolPtr(true), "xhigh")},
		map[string]config.ModelParams{"claude-model": {ReasoningFamily: "claude"}},
	)
	profile := &config.ProviderProfile{ReasoningTransport: modelcatalog.ReasoningTransportOutputConfig}

	t.Run("projects effort into output config and preserves siblings", func(t *testing.T) {
		body, err := json.Marshal(map[string]interface{}{
			"model": "claude-model", "messages": []map[string]string{{"role": "user", "content": "hello"}},
			"output_config": map[string]interface{}{"format": map[string]interface{}{"type": "json_schema"}},
		})
		require.NoError(t, err)
		modified, err := router.setReasoningModeToRequestBodyForProvider(
			body, true, router.Config.GetDecisionByName("deep"), profile,
		)
		require.NoError(t, err)
		request := unmarshalReasoningRequest(t, modified)
		outputConfig := request["output_config"].(map[string]interface{})
		assert.Equal(t, "xhigh", outputConfig["effort"])
		assert.NotNil(t, outputConfig["format"])
		_, hasTopLevel := request["reasoning_effort"]
		assert.False(t, hasTopLevel)
	})

	t.Run("disabled routing preserves an explicitly authored effort", func(t *testing.T) {
		body := []byte(`{"model":"claude-model","messages":[],"reasoning_effort":"medium"}`)
		modified, err := router.setReasoningModeToRequestBodyForProvider(body, false, nil, profile)
		require.NoError(t, err)
		request := unmarshalReasoningRequest(t, modified)
		outputConfig := request["output_config"].(map[string]interface{})
		assert.Equal(t, "medium", outputConfig["effort"])
	})
}
