package dsl

import (
	"context"
	"fmt"
	"strings"
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

// NLOption configures the NL-to-DSL generation pipeline.
type NLOption func(*nlConfig)

type nlConfig struct {
	temperature float64
	maxTokens   int
	maxRetries  int
	validate    bool
	format      bool
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
// full DSL reference + few-shot examples, and the user prompt contains only
// the natural language instruction.
//
// On parse failure, it retries with a repair prompt that includes the bad code,
// the error message, and the full schema reference (up to maxRetries times).
func GenerateFromNL(ctx context.Context, client LLMClient, instruction string, opts ...NLOption) (*NLResult, error) {
	cfg := defaultNLConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	result := &NLResult{}

	// Initial generation
	messages := []ChatMessage{
		{Role: "system", Content: SystemPrompt},
		{Role: "user", Content: BuildNLPrompt(instruction)},
	}

	for attempt := 0; attempt <= cfg.maxRetries; attempt++ {
		result.Attempts = attempt + 1

		raw, err := client.ChatCompletion(ctx, ChatCompletionRequest{
			Messages:    messages,
			Temperature: cfg.temperature,
			MaxTokens:   cfg.maxTokens,
		})
		if err != nil {
			return nil, fmt.Errorf("LLM call failed (attempt %d): %w", attempt+1, err)
		}

		result.RawOutput = raw
		sanitized := SanitizeLLMOutput(raw)

		if !cfg.validate {
			result.DSL = sanitized
			return result, nil
		}

		prog, parseErrors := Parse(sanitized)
		if len(parseErrors) > 0 {
			errMsg := formatErrors(parseErrors)
			result.ParseError = errMsg

			if attempt < cfg.maxRetries {
				messages = []ChatMessage{
					{Role: "system", Content: SystemPrompt},
					{Role: "user", Content: BuildRepairPrompt(sanitized, errMsg)},
				}
				continue
			}

			result.DSL = sanitized
			return result, nil
		}

		// Parse succeeded -- run validation for warnings
		diags := ValidateAST(prog)
		for _, d := range diags {
			result.Warnings = append(result.Warnings, d.String())
		}
		result.ParseError = ""

		if cfg.format && !containsDecisionTree(sanitized) {
			formatted, fmtErr := Format(sanitized)
			if fmtErr == nil {
				result.DSL = formatted
			} else {
				result.DSL = sanitized
			}
		} else {
			result.DSL = sanitized
		}

		return result, nil
	}

	return result, nil
}

// containsDecisionTree checks if the DSL source contains a DECISION_TREE block.
// Format() compiles decision trees into equivalent ROUTE blocks, which destroys
// the user's intended structure. We skip formatting in this case.
func containsDecisionTree(source string) bool {
	return strings.Contains(source, "DECISION_TREE")
}

func formatErrors(errs []error) string {
	msgs := make([]string, len(errs))
	for i, e := range errs {
		msgs[i] = e.Error()
	}
	return strings.Join(msgs, "; ")
}
