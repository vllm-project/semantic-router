//go:build !windows

package extproc

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/png"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// imageContextRouter routes requests of 200K tokens or more to long-model and
// everything else to vision-model.
func imageContextRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	return dispatchPluginsRouter(t, `
version: v0.5
providers:
  defaults:
    model: vision-model
  models:
    - name: vision-model
      provider_model_id: vision-model
      api_format: chat_completions
      backend_refs:
        - name: vision-backend
          provider: vllm
          endpoint: http://127.0.0.1:8000/v1
    - name: long-model
      provider_model_id: long-model
      api_format: chat_completions
      backend_refs:
        - name: long-backend
          provider: vllm
          endpoint: http://127.0.0.1:8001/v1
recipes:
  - name: default
    signals:
      context:
        - name: beyond_window
          min_tokens: 200K
    decisions:
      - name: long_context
        priority: 900
        rules:
          operator: AND
          conditions:
            - type: context
              name: beyond_window
        modelRefs:
          - model: long-model
      - name: vision
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: vision-model
`)
}

// Twenty-five small inline pictures are estimated from their size, a few
// hundred tokens in all, so they stay on the vision lane instead of passing
// a 200K-token context band at 8K tokens each. Pictures whose size the
// Router cannot read still count 8K each.
func TestSmallInlinePicturesStayBelowALongContextBand(t *testing.T) {
	var picture bytes.Buffer
	require.NoError(t, png.Encode(&picture, image.NewGray(image.Rect(0, 0, 120, 90))))
	small := base64.StdEncoding.EncodeToString(picture.Bytes())
	request := func(data string) string {
		parts := []string{`{"type":"text","text":"Which of these icons shows a cat?"}`}
		for range 25 {
			parts = append(parts, fmt.Sprintf(`{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}`, data))
		}
		return `{"model":"vllm-sr/auto","messages":[{"role":"user","content":[` + strings.Join(parts, ",") + `]}]}`
	}
	router := imageContextRouter(t)

	_, ctx := sendThroughPlugins(t, router, "/v1/chat/completions", request(small))
	require.Equal(t, "vision-model", ctx.VSRSelectedModel)
	require.Less(t, ctx.VSRContextTokenCount, 2_000)

	_, ctx = sendThroughPlugins(t, router, "/v1/chat/completions", request("AAAA"))
	require.Equal(t, "long-model", ctx.VSRSelectedModel, "25 pictures of unknown size count 8K tokens each")
}
