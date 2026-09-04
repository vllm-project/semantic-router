//go:build !onnx

package cache

import (
	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

func queryTokensExceed(query string, window int) bool {
	tokens, err := candle_binding.TokenizeText(query, window+1)
	return err == nil && len(tokens.TokenIDs) > window
}
