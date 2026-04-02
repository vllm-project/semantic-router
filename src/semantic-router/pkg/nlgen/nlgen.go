package nlgen

import (
	"context"

	internal "github.com/vllm-project/semantic-router/src/semantic-router/internal/nlgen"
)

type (
	LLMClient             = internal.LLMClient
	ChatCompletionRequest = internal.ChatCompletionRequest
	ChatMessage           = internal.ChatMessage
	NLOption              = internal.NLOption
	NLResult              = internal.NLResult
	ProgressEvent         = internal.ProgressEvent
	ProgressReporter      = internal.ProgressReporter
)

const (
	SchemaReference = internal.SchemaReference
	FewShotExamples = internal.FewShotExamples
	SystemPrompt    = internal.SystemPrompt
)

var (
	NewOpenAIClient      = internal.NewOpenAIClient
	WithTemperature      = internal.WithTemperature
	WithMaxTokens        = internal.WithMaxTokens
	WithMaxRetries       = internal.WithMaxRetries
	WithValidation       = internal.WithValidation
	WithFormat           = internal.WithFormat
	WithTaskContext      = internal.WithTaskContext
	WithAttemptOffset    = internal.WithAttemptOffset
	WithProgressReporter = internal.WithProgressReporter
	SanitizeLLMOutput    = internal.SanitizeLLMOutput
)

func GenerateFromNL(ctx context.Context, client LLMClient, instruction string, opts ...NLOption) (*NLResult, error) {
	return internal.GenerateFromNL(ctx, client, instruction, opts...)
}

func RepairFromFeedback(ctx context.Context, client LLMClient, instruction string, badCode string, feedback string, opts ...NLOption) (*NLResult, error) {
	return internal.RepairFromFeedback(ctx, client, instruction, badCode, feedback, opts...)
}

func BuildNLPrompt(instruction string) string {
	return internal.BuildNLPrompt(instruction)
}

func BuildNLPromptWithContext(instruction string, taskContext string) string {
	return internal.BuildNLPromptWithContext(instruction, taskContext)
}

func BuildRepairPrompt(badCode string, parseErr string) string {
	return internal.BuildRepairPrompt(badCode, parseErr)
}

func BuildRepairPromptWithContext(instruction string, taskContext string, badCode string, feedback string) string {
	return internal.BuildRepairPromptWithContext(instruction, taskContext, badCode, feedback)
}
