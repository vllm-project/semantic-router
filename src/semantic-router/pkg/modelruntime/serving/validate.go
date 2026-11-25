package serving

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func validateText(text string) error {
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("model input text must not be empty")
	}
	return nil
}

// itemError is the error of one item a model runtime call answered with an
// error code.
func itemError(code string) error {
	switch code {
	case "max_length_exceeded":
		return fmt.Errorf("%w: %s", binding.ErrInputLimit, code)
	case "scan_budget_exceeded":
		return fmt.Errorf("%w: %s", binding.ErrScanBudget, code)
	case "invalid_input":
		return fmt.Errorf("%w: %s", binding.ErrInvalidInput, code)
	case "invalid_model_output":
		return fmt.Errorf("%w: %s", binding.ErrInvalidResult, code)
	case "deadline_exceeded":
		return context.DeadlineExceeded
	case "unavailable":
		return modelservice.ErrUnavailable
	default:
		return fmt.Errorf("%w: %s", modelservice.ErrFailed, code)
	}
}

func inputUsage(usage *modelservice.InputUsage) *tasks.InputUsage {
	if usage == nil {
		return nil
	}
	return &tasks.InputUsage{OriginalTokens: usage.Tokens, ProcessedTokens: usage.ProcessedTokens, Truncated: usage.Truncated}
}
