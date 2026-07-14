package candle_binding

import (
	"errors"
	"fmt"
)

// ErrEmbeddingInputTooLong marks an input whose true tokenizer output exceeds
// the selected embedding model's context window.
var ErrEmbeddingInputTooLong = errors.New("embedding input exceeds model context")

const embeddingInputTooLongStatus = -3

func embeddingStatusError(operation string, status int) error {
	if status == embeddingInputTooLongStatus {
		return fmt.Errorf("%w: %s", ErrEmbeddingInputTooLong, operation)
	}
	return fmt.Errorf("%s (status: %d)", operation, status)
}
