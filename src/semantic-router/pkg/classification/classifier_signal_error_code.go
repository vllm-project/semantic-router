package classification

import (
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

const signalInputLimitCode = "input_limit"

// Publish only a typed, bounded subtype; provider messages may contain paths or input.
func boundedSignalErrorCode(err error, fallback string) string {
	if errors.Is(err, binding.ErrInputLimit) {
		return signalInputLimitCode
	}
	return fallback
}

// A rule can read several pieces: preserve a known input limit independently of order.
func mergeSignalErrorCode(current, next string) string {
	if current == "" || next == signalInputLimitCode {
		return next
	}
	return current
}
