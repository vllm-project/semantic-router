// Package shadow implements bounded multi-arm shadow comparison for routing
// evaluation (issue #3376). Each arm replays the same normalized request to an
// OpenAI-compatible backend so normalized inputs stay byte-identical across the
// primary route and every shadow arm. Dispatch is synchronous over the arms but
// is designed to run inside a caller-owned goroutine; all arm failures are
// reported in the returned results and never surfaced to the primary path.
//
// The arm client is intentionally a small stdlib OpenAI-compatible POST: it
// keeps this package free of the native binding dependency chain carried by the
// looper client, so shadow logic is testable on any host.
package shadow

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// maxResponseBytes bounds a single arm response read (32 MiB, matching the
// router's per-upstream ceiling) so a huge or malicious arm cannot exhaust
// router memory.
const maxResponseBytes int64 = 32 << 20

// defaultArmTimeout bounds a single arm attempt when the arm does not set one.
const defaultArmTimeout = 30 * time.Second

// ArmResult is the outcome of one shadow arm evaluation.
type ArmResult struct {
	// Arm identifies the configured arm (stable name, opaque to observers).
	Arm string
	// Model is what the arm actually invoked (kept for operator debugging,
	// excluded from judge input by the M3 blinding layer).
	Model string
	// OK is true only when the arm returned a non-streaming completion.
	OK bool
	// LatencyMS is the observed round-trip on success.
	LatencyMS int64
	// Content is the extracted text of the completion. Raw hidden reasoning
	// from the candidate is intentionally never captured (issue #3376).
	Content string
	// Err is set for configuration, timeout, cancellation or HTTP failures.
	Err string
}

// AccessKey resolves the per-arm Authorization bearer key (empty disables the
// Authorization header). Returning an error fails that arm only.
type AccessKey func(armName string) (string, error)

// Dispatch evaluates the normalized params against every configured arm and
// returns one result per arm, in configuration order. Nil params is a
// fail-closed result for every arm. ctx cancellation (client disconnect,
// aggregate shadow window, router shutdown) stops all in-flight arms.
func Dispatch(ctx context.Context, cfg config.ShadowComparisonConfig, params *openai.ChatCompletionNewParams, key AccessKey) []ArmResult {
	results := make([]ArmResult, 0, len(cfg.Arms))
	for _, arm := range cfg.Arms {
		results = append(results, runArm(ctx, arm, params, key))
	}
	return results
}

func runArm(ctx context.Context, arm config.ShadowArmConfig, params *openai.ChatCompletionNewParams, key AccessKey) ArmResult {
	if params == nil {
		return armFailure(arm, "nil params")
	}
	if arm.Model == "" || arm.Endpoint == "" {
		return armFailure(arm, "arm missing model or endpoint")
	}
	if ctx == nil {
		ctx = context.Background()
	}

	// Each arm gets its own bounded context so one slow arm cannot outlive the
	// aggregate shadow window.
	// ponytail: per-arm timeout only; per-arm concurrency and token/cost
	// budgets land in the M2 aggregate budget controller (issue #3376).
	armCtx, cancel, armTimeout := boundedContext(ctx, arm)
	defer cancel()

	req, reqErr := buildArmRequest(armCtx, arm, params, key)
	if reqErr != "" {
		return armFailure(arm, reqErr)
	}

	client := &http.Client{Timeout: armTimeout}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		return armFailure(arm, fmt.Sprintf("request: %v", err))
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		return armFailure(arm, fmt.Sprintf("read response: %v", err))
	}
	if resp.StatusCode != http.StatusOK {
		return armFailure(arm, fmt.Sprintf("status %d: %s", resp.StatusCode, truncate(respBody)))
	}
	content, parseErr := parseCompletion(respBody)
	if parseErr != "" {
		return armFailure(arm, parseErr)
	}
	return ArmResult{
		Arm:       arm.Name,
		Model:     arm.Model,
		OK:        true,
		LatencyMS: time.Since(start).Milliseconds(),
		Content:   content,
	}
}

// boundedContext applies an arm timeout as its own cancellation scope. Positive
// Arm.TimeoutSeconds overrides the package default.
func boundedContext(ctx context.Context, arm config.ShadowArmConfig) (context.Context, context.CancelFunc, time.Duration) {
	armTimeout := defaultArmTimeout
	if arm.TimeoutSeconds <= 0 {
		return ctx, func() {}, armTimeout
	}
	armTimeout = time.Duration(arm.TimeoutSeconds) * time.Second
	sub, cancel := context.WithTimeout(ctx, armTimeout)
	return sub, cancel, armTimeout
}

// buildArmRequest clones the caller's normalized params so it is never muted,
// swaps in the arm model, and assembles the POST request with the resolved
// bearer key (when an AccessKey is provided).
func buildArmRequest(ctx context.Context, arm config.ShadowArmConfig, params *openai.ChatCompletionNewParams, key AccessKey) (*http.Request, string) {
	payload := cloneParams(params)
	payload.Model = arm.Model
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Sprintf("marshal params: %v", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, arm.Endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Sprintf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if key != nil {
		k, err := key(arm.Name)
		if err != nil {
			return nil, fmt.Sprintf("resolve key: %v", err)
		}
		if k != "" {
			req.Header.Set("Authorization", "Bearer "+k)
		}
	}
	return req, ""
}

// parseCompletion extracts the single text completion from a non-streaming
// OpenAI-compatible body under a closed response shape.
func parseCompletion(respBody []byte) (string, string) {
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &completion); err != nil {
		return "", fmt.Sprintf("parse response: %v", err)
	}
	if len(completion.Choices) == 0 {
		return "", "empty choices"
	}
	return completion.Choices[0].Message.Content, ""
}

// cloneParams shallow-copies params so the caller's normalized request is never
// mutated (each arm only swaps the model name).
func cloneParams(params *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	copied := *params
	return &copied
}

func armFailure(arm config.ShadowArmConfig, err string) ArmResult {
	return ArmResult{Arm: arm.Name, Model: arm.Model, Err: err}
}

func truncate(b []byte) string {
	const max = 512
	if len(b) <= max {
		return string(b)
	}
	return string(b[:max])
}
