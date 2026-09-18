package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
)

func TestMergeRouterExtensions(t *testing.T) {
	t.Run("nil extensions returns original body", func(t *testing.T) {
		body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4"}`)
		result, err := mergeRouterExtensions(body, nil)
		require.NoError(t, err)
		assert.Equal(t, body, result)
	})

	t.Run("empty extensions returns original body", func(t *testing.T) {
		body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4"}`)
		ext := &looper.RouterExtensions{}
		result, err := mergeRouterExtensions(body, ext)
		require.NoError(t, err)
		assert.Equal(t, body, result)
	})

	t.Run("merges fusion extension into body", func(t *testing.T) {
		body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4","choices":[]}`)
		fusionTrace := map[string]interface{}{
			"analysis":       map[string]interface{}{"mode": "consensus"},
			"failed_models":  []interface{}{"model-b"},
			"judge_model":    "gpt-4",
			"analysis_models": []interface{}{"gpt-4", "model-a", "model-b"},
		}
		ext := &looper.RouterExtensions{
			Fusion: fusionTrace,
		}
		result, err := mergeRouterExtensions(body, ext)
		require.NoError(t, err)

		var parsed map[string]interface{}
		require.NoError(t, json.Unmarshal(result, &parsed))
		assert.Contains(t, parsed, "fusion")
		assert.NotContains(t, parsed, "flow")
		assert.NotContains(t, parsed, "reasoning_mom_responses")

		fusion := parsed["fusion"].(map[string]interface{})
		assert.Equal(t, "consensus", fusion["analysis"].(map[string]interface{})["mode"])
	})

	t.Run("merges flow extension into body", func(t *testing.T) {
		body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4","choices":[]}`)
		flowTrace := map[string]interface{}{
			"steps": []interface{}{map[string]interface{}{"step": 1}},
		}
		ext := &looper.RouterExtensions{
			Flow: flowTrace,
		}
		result, err := mergeRouterExtensions(body, ext)
		require.NoError(t, err)

		var parsed map[string]interface{}
		require.NoError(t, json.Unmarshal(result, &parsed))
		assert.Contains(t, parsed, "flow")
		assert.NotContains(t, parsed, "fusion")
	})

	t.Run("merges reasoning_mom_responses extension into body", func(t *testing.T) {
		body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4","choices":[]}`)
		rounds := []interface{}{map[string]interface{}{"round": 1}}
		ext := &looper.RouterExtensions{
			ReMoM: rounds,
		}
		result, err := mergeRouterExtensions(body, ext)
		require.NoError(t, err)

		var parsed map[string]interface{}
		require.NoError(t, json.Unmarshal(result, &parsed))
		assert.Contains(t, parsed, "reasoning_mom_responses")
		assert.NotContains(t, parsed, "fusion")
		assert.NotContains(t, parsed, "flow")
	})

	t.Run("merges all three extensions", func(t *testing.T) {
		body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4","choices":[]}`)
		ext := &looper.RouterExtensions{
			Fusion: map[string]interface{}{"analysis": "x"},
			Flow:   map[string]interface{}{"steps": []interface{}{}},
			ReMoM:  []interface{}{map[string]interface{}{"round": 1}},
		}
		result, err := mergeRouterExtensions(body, ext)
		require.NoError(t, err)

		var parsed map[string]interface{}
		require.NoError(t, json.Unmarshal(result, &parsed))
		assert.Contains(t, parsed, "fusion")
		assert.Contains(t, parsed, "flow")
		assert.Contains(t, parsed, "reasoning_mom_responses")
	})

	t.Run("preserves existing body fields", func(t *testing.T) {
		body := []byte(`{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-4","usage":{"prompt_tokens":10,"completion_tokens":20}}`)
		ext := &looper.RouterExtensions{
			Fusion: map[string]interface{}{"analysis": "x"},
		}
		result, err := mergeRouterExtensions(body, ext)
		require.NoError(t, err)

		var parsed map[string]interface{}
		require.NoError(t, json.Unmarshal(result, &parsed))
		assert.Equal(t, "chatcmpl-1", parsed["id"])
		assert.Equal(t, "gpt-4", parsed["model"])
		assert.Contains(t, parsed, "usage")
	})

	t.Run("returns error on invalid body JSON", func(t *testing.T) {
		body := []byte(`{invalid json`)
		ext := &looper.RouterExtensions{
			Fusion: map[string]interface{}{"x": 1},
		}
		_, err := mergeRouterExtensions(body, ext)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to unmarshal")
	})
}
