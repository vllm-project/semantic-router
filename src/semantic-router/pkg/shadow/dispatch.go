// Package shadow implements bounded multi-arm shadow comparison for routing
// evaluation (issue #3376). Each arm replays the same normalized request to an
// OpenAI-compatible backend so normalized inputs stay byte-identical across the
// primary route and every shadow arm. Dispatch runs arms concurrently under a
// per-request aggregate budget; all arm failures are reported in the returned
// results and never surfaced to the primary path.
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
	"sync"
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
	// Outcome is the normalized lifecycle result, reconciled into the
	// aggregate budget and persisted to Replay evidence (C4).
	Outcome Outcome
	// LatencyMS is the observed round-trip on success.
	LatencyMS int64
	// Content is the extracted text of the completion. Raw hidden reasoning
	// from the candidate is intentionally never captured (issue #3376).
	Content string
	// PromptTokens / CompletionTokens are parsed from the arm's usage block
	// (0 when the response carried none).
	PromptTokens     int64
	CompletionTokens int64
	// Err is set for configuration, timeout, cancellation, budget or HTTP
	// failures.
	Err string
}

// AccessKey resolves the per-arm Authorization bearer key (empty disables the
// Authorization header). Returning an error fails that arm only.
type AccessKey func(armName string) (string, error)

// Dispatch evaluates the normalized params against every configured arm
// concurrently under the aggregate budget and returns one result per arm in
// configuration order. Arms rejected before admission carry OutcomeSkipped.
// Nil params is a fail-closed result for every arm. ctx cancellation (client
// disconnect, aggregate shadow window, router shutdown) stops all in-flight
// arms.
func Dispatch(ctx context.Context, cfg config.ShadowComparisonConfig, params *openai.ChatCompletionNewParams, key AccessKey) []ArmResult {
	results := make([]ArmResult, len(cfg.Arms))
	if len(cfg.Arms) == 0 {
		return results
	}
	budget := newBudget(cfg.Budget)
	var wg sync.WaitGroup
	for i := range cfg.Arms {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i] = runArm(ctx, budget, cfg.Arms[i], params, key)
		}(i)
	}
	wg.Wait()
	return results
}

func runArm(ctx context.Context, budget *Budget, arm config.ShadowArmConfig, params *openai.ChatCompletionNewParams, key AccessKey) ArmResult {
	if params == nil {
		return ArmResult{Arm: arm.Name, Model: arm.Model, Outcome: OutcomeFailed, Err: "nil params"}
	}
	if arm.Model == "" || arm.Endpoint == "" {
		return ArmResult{Arm: arm.Name, Model: arm.Model, Outcome: OutcomeFailed, Err: "arm missing model or endpoint"}
	}
	if ctx == nil {
		ctx = context.Background()
	}

	// Aggregate budget gates admission deterministically. The request's
	// declared output cap is reserved so concurrent arms cannot jointly
	// overshoot token/cost budgets (issue #3376, review feedback).
	reserve := outputReserve(params)
	rejected, ok := budget.tryEnter(arm.Name, arm.Model, reserve)
	if !ok {
		return rejected
	}
	outcome := OutcomeFailed
	pTokens, cTokens := int64(0), int64(0)
	defer func() { budget.settle(outcome, pTokens, cTokens, reserve) }()

	// Each arm gets its own bounded context so one slow arm cannot outlive the
	// aggregate shadow window.
	// ponytail: per-arm timeout only; token/cost without a declared output cap
	// are accounted on completion and enforced for later arms; wall time is
	// bounded by the caller's aggregate context.
	armCtx, cancel, armTimeout := boundedContext(ctx, arm)
	defer cancel()

	req, reqErr := buildArmRequest(armCtx, arm, params, key)
	if reqErr != "" {
		outcome = outcomeFor(armCtx)
		return budgetedFailure(arm, outcome, reqErr)
	}

	client := &http.Client{Timeout: armTimeout}
	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		outcome = outcomeFor(armCtx)
		return budgetedFailure(arm, outcome, fmt.Sprintf("request: %v", err))
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxResponseBytes))
	if err != nil {
		outcome = OutcomeFailed
		return budgetedFailure(arm, outcome, fmt.Sprintf("read response: %v", err))
	}
	if resp.StatusCode != http.StatusOK {
		outcome = OutcomeFailed
		return budgetedFailure(arm, outcome,
			fmt.Sprintf("status %d: %s", resp.StatusCode, truncate(respBody)))
	}

	completion, parseErr := parseCompletion(respBody)
	if parseErr != "" {
		outcome = OutcomeFailed
		return budgetedFailure(arm, outcome, parseErr)
	}
	pTokens, cTokens = completion.PromptTokens, completion.CompletionTokens
	outcome = OutcomeCompleted
	return ArmResult{
		Arm:              arm.Name,
		Model:            arm.Model,
		Outcome:          outcome,
		LatencyMS:        time.Since(start).Milliseconds(),
		Content:          completion.Content,
		PromptTokens:     completion.PromptTokens,
		CompletionTokens: completion.CompletionTokens,
	}
}

// outputReserve returns the request's declared maximum output tokens, used to
// reserve token/cost headroom at arm admission. MaxCompletionTokens takes
// precedence over the legacy MaxTokens; 0 means the request declares no output
// cap, so token/cost can only be reconciled after completion.
func outputReserve(params *openai.ChatCompletionNewParams) int64 {
	if params == nil {
		return 0
	}
	if v := params.MaxCompletionTokens; v.Valid() {
		return v.Value
	}
	if v := params.MaxTokens; v.Valid() {
		return v.Value
	}
	return 0
}

// budgetedFailure returns the ArmResult for a non-completed outcome; only
// completed arms ever consume token/cost in the budget.
func budgetedFailure(arm config.ShadowArmConfig, outcome Outcome, err string) ArmResult {
	return ArmResult{Arm: arm.Name, Model: arm.Model, Outcome: outcome, Err: err}
}

// outcomeFor maps an arm failure to its normalized outcome: a deadline first,
// then cancellation, then generic failure.
func outcomeFor(armCtx context.Context) Outcome {
	switch {
	case armCtx.Err() == context.DeadlineExceeded:
		return OutcomeTimedOut
	case armCtx.Err() == context.Canceled:
		return OutcomeCancelled
	default:
		return OutcomeFailed
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

type armCompletion struct {
	Content          string
	PromptTokens     int64
	CompletionTokens int64
}

// parseCompletion extracts the single text completion plus usage accounting
// from a non-streaming OpenAI-compatible body under a closed response shape.
func parseCompletion(respBody []byte) (*armCompletion, string) {
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int64 `json:"prompt_tokens"`
			CompletionTokens int64 `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(respBody, &completion); err != nil {
		return nil, fmt.Sprintf("parse response: %v", err)
	}
	if len(completion.Choices) == 0 {
		return nil, "empty choices"
	}
	return &armCompletion{
		Content:          completion.Choices[0].Message.Content,
		PromptTokens:     completion.Usage.PromptTokens,
		CompletionTokens: completion.Usage.CompletionTokens,
	}, ""
}

// cloneParams shallow-copies params so the caller's normalized request is never
// mutated (each arm only swaps the model name).
func cloneParams(params *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	copied := *params
	return &copied
}

func truncate(b []byte) string {
	const max = 512
	if len(b) <= max {
		return string(b)
	}
	return string(b[:max])
}
