package protocolcodec

import (
	"bytes"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func decodeTextVerbosity(raw json.RawMessage, target *string) error {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil
	}
	var value string
	if err := json.Unmarshal(raw, &value); err != nil ||
		value != "low" && value != "medium" && value != "high" {
		return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest,
			"invalid_text_verbosity", "text verbosity must be low, medium or high", err)
	}
	*target = value
	return nil
}
