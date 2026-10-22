package classification

import (
	"context"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

const (
	signalInputLimitCode = "input_limit"
	signalScanBudgetCode = "scan_budget"
	signalDeadlineCode   = "deadline"
)

// Publish only a typed, bounded subtype; provider messages may contain paths or input.
func boundedSignalErrorCode(err error, fallback string) string {
	switch {
	case errors.Is(err, binding.ErrScanBudget):
		return signalScanBudgetCode
	case errors.Is(err, binding.ErrInputLimit), errors.Is(err, tasks.ErrTokenSpansTruncated):
		return signalInputLimitCode
	case errors.Is(err, errSignalDeadline):
		return signalDeadlineCode
	}
	return fallback
}

// A rule can read several pieces: preserve a known input limit independently of order.
func mergeSignalErrorCode(current, next string) string {
	if current == "" || (lengthCode(next) && !lengthCode(current)) {
		return next
	}
	return current
}

func lengthCode(code string) bool {
	return code == signalInputLimitCode || code == signalScanBudgetCode || code == signalDeadlineCode
}

// UnscannedInput reports whether err says a model did not read all of an
// input: longer than the model's input, than the scan budget of a model that
// reads it in windows, truncated, or not scanned by the signals' deadline.
func UnscannedInput(err error) bool {
	return errors.Is(err, binding.ErrInputLimit) || errors.Is(err, binding.ErrScanBudget) ||
		errors.Is(err, tasks.ErrTokenSpansTruncated) || errors.Is(err, errSignalDeadline)
}

// errSignalDeadline marks a scan the signals' own deadline cut short: content
// the model did not read in time. A backend's own timeout, before that
// deadline, stays a backend failure that on_error decides.
var errSignalDeadline = errors.New("the signals' deadline passed before the scan finished")

// signalDeadline marks err when the signals' deadline (ctx) has passed.
func signalDeadline(ctx context.Context, err error) error {
	if err != nil && errors.Is(err, context.DeadlineExceeded) && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return fmt.Errorf("%w: %w", errSignalDeadline, err)
	}
	return err
}
