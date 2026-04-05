package nlgen

import (
	"context"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

// LLMClient is the interface for calling a language model.
// Implementations can wrap OpenAI, vLLM, Ollama, or any other backend.
type LLMClient interface {
	ChatCompletion(ctx context.Context, req ChatCompletionRequest) (string, error)
}

// ChatCompletionRequest is a minimal chat completion request.
type ChatCompletionRequest struct {
	Messages    []ChatMessage
	Temperature float64
	MaxTokens   int
}

// ChatMessage represents a single message in a chat conversation.
type ChatMessage struct {
	Role    string // "system" or "user"
	Content string
}

// ProgressEvent reports stage-level progress from the NL-to-DSL pipeline.
type ProgressEvent struct {
	Phase   string
	Message string
	Attempt int
}

// ProgressReporter receives stage-level progress updates from GenerateFromNL or
// RepairFromFeedback.
type ProgressReporter func(ProgressEvent)

// NLOption configures the NL-to-DSL generation pipeline.
type NLOption func(*nlConfig)

type nlConfig struct {
	temperature float64
	maxTokens   int
	maxRetries  int
	validate    bool
	format      bool
	taskContext string
	attemptBase int
	reporter    ProgressReporter
}

func defaultNLConfig() nlConfig {
	return nlConfig{
		temperature: 0.1,
		maxTokens:   4096,
		maxRetries:  2,
		validate:    true,
		format:      true,
	}
}

// WithTemperature sets the LLM sampling temperature (default 0.1).
func WithTemperature(t float64) NLOption {
	return func(c *nlConfig) { c.temperature = t }
}

// WithMaxTokens sets the maximum output tokens (default 4096).
func WithMaxTokens(n int) NLOption {
	return func(c *nlConfig) { c.maxTokens = n }
}

// WithMaxRetries sets the maximum number of repair retries (default 2).
func WithMaxRetries(n int) NLOption {
	return func(c *nlConfig) { c.maxRetries = n }
}

// WithValidation enables/disables parse validation (default true).
func WithValidation(v bool) NLOption {
	return func(c *nlConfig) { c.validate = v }
}

// WithFormat enables/disables canonical formatting of the output (default true).
func WithFormat(f bool) NLOption {
	return func(c *nlConfig) { c.format = f }
}

// WithTaskContext injects additional task context into the shared prompt and
// repair prompt without changing the canonical schema/few-shot prompt body.
func WithTaskContext(taskContext string) NLOption {
	return func(c *nlConfig) { c.taskContext = strings.TrimSpace(taskContext) }
}

// WithAttemptOffset shifts reported attempt numbers while leaving result.Attempts
// relative to the current invocation.
func WithAttemptOffset(attemptBase int) NLOption {
	return func(c *nlConfig) {
		if attemptBase > 0 {
			c.attemptBase = attemptBase
		}
	}
}

// WithProgressReporter attaches a progress callback to the NL pipeline.
func WithProgressReporter(reporter ProgressReporter) NLOption {
	return func(c *nlConfig) { c.reporter = reporter }
}

// NLResult contains the output of an NL-to-DSL generation attempt.
type NLResult struct {
	DSL        string   // The generated (and optionally formatted) DSL program
	RawOutput  string   // The raw LLM output before sanitization
	Attempts   int      // Number of LLM calls made (1 = first try succeeded)
	ParseError string   // Non-empty if the final output still has parse errors
	Warnings   []string // Validation warnings (non-fatal)
}

// GenerateFromNL generates a DSL program from a natural language description.
// It uses the schema-as-supervision approach: the system prompt contains the
// full DSL reference + few-shot examples, and the user prompt contains the
// natural language instruction plus optional task context.
//
// On parse failure, it retries with a repair prompt that includes the bad code,
// the error message, and the shared schema reference (up to maxRetries times).
func GenerateFromNL(ctx context.Context, client LLMClient, instruction string, opts ...NLOption) (*NLResult, error) {
	cfg := applyOptions(opts...)
	return runGeneration(
		ctx,
		client,
		instruction,
		"generate",
		[]ChatMessage{
			{Role: "system", Content: SystemPrompt},
			{Role: "user", Content: BuildNLPromptWithContext(instruction, cfg.taskContext)},
		},
		&cfg,
	)
}

// RepairFromFeedback repairs an existing DSL program using shared schema-aware
// prompting plus the provided feedback text.
func RepairFromFeedback(
	ctx context.Context,
	client LLMClient,
	instruction string,
	badCode string,
	feedback string,
	opts ...NLOption,
) (*NLResult, error) {
	cfg := applyOptions(opts...)
	return runGeneration(
		ctx,
		client,
		instruction,
		"repair",
		[]ChatMessage{
			{Role: "system", Content: SystemPrompt},
			{Role: "user", Content: BuildRepairPromptWithContext(instruction, cfg.taskContext, badCode, feedback)},
		},
		&cfg,
	)
}

func applyOptions(opts ...NLOption) nlConfig {
	cfg := defaultNLConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func runGeneration(
	ctx context.Context,
	client LLMClient,
	instruction string,
	initialPhase string,
	initialMessages []ChatMessage,
	cfg *nlConfig,
) (*NLResult, error) {
	result := &NLResult{}
	messages := initialMessages

	for attempt := 0; attempt <= cfg.maxRetries; attempt++ {
		attemptNumber := attempt + 1
		result.Attempts = attemptNumber
		reportProgress(cfg, phaseForAttempt(initialPhase, attempt), attemptNumber, fmt.Sprintf("%s attempt %d/%d.", attemptLabel(initialPhase, attempt), attemptNumber, cfg.maxRetries+1))

		raw, err := client.ChatCompletion(ctx, ChatCompletionRequest{
			Messages:    messages,
			Temperature: cfg.temperature,
			MaxTokens:   cfg.maxTokens,
		})
		if err != nil {
			return nil, fmt.Errorf("LLM call failed (attempt %d): %w", attemptNumber, err)
		}
		result.RawOutput = raw

		sanitized := SanitizeLLMOutput(raw)

		if !cfg.validate {
			result.DSL = formatIfNeeded(cfg, sanitized)
			reportProgress(cfg, "complete", attemptNumber, "Returned a generated DSL draft without parse validation.")
			return result, nil
		}

		done, next := validateAndFormat(result, cfg, instruction, sanitized, attemptNumber)
		if done {
			return result, nil
		}
		messages = next
	}

	return result, nil
}

// validateAndFormat parses and validates the sanitized DSL output.
// Returns (true, nil) if the result is final, or (false, repairMessages) to retry.
func validateAndFormat(result *NLResult, cfg *nlConfig, instruction string, sanitized string, attemptNumber int) (bool, []ChatMessage) {
	prog, parseErrors := dsl.Parse(sanitized)
	if len(parseErrors) > 0 {
		errMsg := formatErrors(parseErrors)
		result.ParseError = errMsg
		reportProgress(cfg, "parse", attemptNumber, fmt.Sprintf("Parse failed: %s", summarizeProgressMessage(errMsg)))

		if attemptNumber <= cfg.maxRetries {
			reportProgress(cfg, "repair", attemptNumber, "Preparing a repair prompt from parser feedback.")
			return false, []ChatMessage{
				{Role: "system", Content: SystemPrompt},
				{Role: "user", Content: BuildRepairPromptWithContext(instruction, cfg.taskContext, sanitized, errMsg)},
			}
		}
		result.DSL = sanitized
		reportProgress(cfg, "complete", attemptNumber, "Exhausted retries and returned the latest DSL draft with remaining parse errors.")
		return true, nil
	}

	reportProgress(cfg, "parse", attemptNumber, "Parsed the generated DSL successfully.")
	diags := dsl.ValidateAST(prog)
	for _, d := range diags {
		result.Warnings = appendUniqueWarnings(result.Warnings, d.String())
	}
	result.ParseError = ""
	result.DSL = formatIfNeeded(cfg, sanitized)
	reportProgress(cfg, "complete", attemptNumber, "Generated a parse-valid DSL draft.")
	return true, nil
}

func formatIfNeeded(cfg *nlConfig, sanitized string) string {
	if cfg.format && !strings.Contains(sanitized, "DECISION_TREE") {
		if formatted, err := dsl.Format(sanitized); err == nil {
			return formatted
		}
	}
	return sanitized
}

func formatErrors(errs []error) string {
	msgs := make([]string, len(errs))
	for i, e := range errs {
		msgs[i] = e.Error()
	}
	return strings.Join(msgs, "; ")
}

func reportProgress(cfg *nlConfig, phase string, attempt int, message string) {
	if cfg == nil || cfg.reporter == nil {
		return
	}
	cfg.reporter(ProgressEvent{
		Phase:   phase,
		Message: strings.TrimSpace(message),
		Attempt: cfg.attemptBase + attempt,
	})
}

func phaseForAttempt(initialPhase string, attempt int) string {
	if attempt == 0 {
		if strings.TrimSpace(initialPhase) != "" {
			return initialPhase
		}
		return "generate"
	}
	return "repair"
}

func attemptLabel(initialPhase string, attempt int) string {
	if attempt == 0 {
		if strings.EqualFold(strings.TrimSpace(initialPhase), "repair") {
			return "Repair"
		}
		return "Generation"
	}
	return "Repair"
}

func summarizeProgressMessage(raw string) string {
	line := strings.TrimSpace(raw)
	if line == "" {
		return "unknown parse error"
	}
	if idx := strings.Index(line, "\n"); idx >= 0 {
		line = strings.TrimSpace(line[:idx])
	}
	if len(line) > 160 {
		return line[:157] + "..."
	}
	return line
}

func appendUniqueWarnings(existing []string, warnings ...string) []string {
	if len(warnings) == 0 {
		return existing
	}
	seen := make(map[string]struct{}, len(existing)+len(warnings))
	result := make([]string, 0, len(existing)+len(warnings))
	for _, warning := range existing {
		warning = strings.TrimSpace(warning)
		if warning == "" {
			continue
		}
		if _, ok := seen[warning]; ok {
			continue
		}
		seen[warning] = struct{}{}
		result = append(result, warning)
	}
	for _, warning := range warnings {
		warning = strings.TrimSpace(warning)
		if warning == "" {
			continue
		}
		if _, ok := seen[warning]; ok {
			continue
		}
		seen[warning] = struct{}{}
		result = append(result, warning)
	}
	return result
}
