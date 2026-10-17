/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package looper

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

const (
	CallStageGenerate   = "generate"
	CallStageVerify     = "verify"
	CallStageJudge      = "judge"
	CallStageSynthesize = "synthesize"
)

// CallInfo contains metadata known before an upstream model request is sent.
// EstimatedPromptTokens is a deterministic approximation based on the encoded
// request body; it is a reservation guard, not a replacement for provider
// usage. RequestBytes is exposed so an external benchmark can audit the
// estimator without retaining request contents.
type CallInfo struct {
	Model                 string
	Stage                 string
	Role                  string
	DecisionName          string
	Iteration             int
	Streaming             bool
	RequestBytes          int64
	EstimatedPromptTokens int64
	ReservedOutputTokens  int64
	EstimatedTotalTokens  int64
}

// CallResult contains terminal information for one upstream call. Response
// may be non-nil even when a later algorithm step rejects it as unusable.
type CallResult struct {
	Response  *ModelResponse
	Err       error
	LatencyMs int64
}

// CallReservation is returned by an observer that admits a call. ID is
// observer-owned and can be used to join accounting rows to raw artifacts.
// EstimatedTokens is copied from CallInfo for simple observers.
type CallReservation struct {
	ID              string
	EstimatedTokens int64
	Opaque          any
}

// CallObserver is the runtime seam used by fixed-budget benchmark runners.
// BeforeCall may reject a dispatch. AfterCall is invoked for every admitted
// call, including transport and response-parse failures.
type CallObserver interface {
	BeforeCall(context.Context, CallInfo) (*CallReservation, error)
	AfterCall(context.Context, CallInfo, *CallReservation, CallResult)
}

type callObserverContextKey struct{}

// WithCallObserver attaches an observer to all Looper client calls made with
// ctx, including calls launched by ReMoM and Fusion goroutines.
func WithCallObserver(ctx context.Context, observer CallObserver) context.Context {
	if observer == nil {
		return ctx
	}
	return context.WithValue(ctx, callObserverContextKey{}, observer)
}

// CallObserverFromContext returns the observer attached to ctx, if any.
func CallObserverFromContext(ctx context.Context) CallObserver {
	if ctx == nil {
		return nil
	}
	observer, _ := ctx.Value(callObserverContextKey{}).(CallObserver)
	return observer
}

func callObserverFor(ctx context.Context, options CallOptions) CallObserver {
	if options.Observer != nil {
		return options.Observer
	}
	return CallObserverFromContext(ctx)
}

func normalizedCallStage(stage string) string {
	switch stage {
	case CallStageGenerate, CallStageVerify, CallStageJudge, CallStageSynthesize:
		return stage
	default:
		return CallStageGenerate
	}
}

func normalizedCallRole(role string) string {
	if role == "" {
		return "candidate"
	}
	return role
}

// estimateCallInfo creates a bounded, body-only token estimate. A tokenizer
// is deliberately not required by the client package; benchmark reports keep
// provider-reported usage as the authoritative value when it is available.
func estimateCallInfo(body []byte, reqOutputTokens int64, target ModelTarget, options CallOptions) CallInfo {
	reqOutputTokens = effectiveOutputTokens(body, reqOutputTokens)
	promptTokens := int64((len(body) + 3) / 4)
	if promptTokens < 1 {
		promptTokens = 1
	}
	if reqOutputTokens < 0 {
		reqOutputTokens = 0
	}
	return CallInfo{
		Model:                 target.Name,
		Stage:                 normalizedCallStage(options.Stage),
		Role:                  normalizedCallRole(options.Role),
		DecisionName:          options.DecisionName,
		Iteration:             options.Iteration,
		Streaming:             options.Mode == ResponseSSE,
		RequestBytes:          int64(len(body)),
		EstimatedPromptTokens: promptTokens,
		ReservedOutputTokens:  reqOutputTokens,
		EstimatedTotalTokens:  promptTokens + reqOutputTokens,
	}
}

func effectiveOutputTokens(body []byte, fallback int64) int64 {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return fallback
	}
	for _, key := range []string{"max_completion_tokens", "max_tokens"} {
		raw, ok := payload[key]
		if !ok {
			continue
		}
		var value int64
		if err := json.Unmarshal(raw, &value); err == nil && value >= 0 {
			return value
		}
	}
	return fallback
}

// BudgetLimits describes a per-execution admission envelope. A zero limit is
// treated as unlimited so the type is safe to embed in optional integrations;
// benchmark manifests always provide positive limits.
type BudgetLimits struct {
	MaxCalls       int64
	MaxTotalTokens int64
}

// BudgetSnapshot is a point-in-time view of admitted calls and token charges.
// ReservedTokens are in-flight reservations and therefore are included in
// admission checks but not in Tokens until settlement.
type BudgetSnapshot struct {
	Calls             int64 `json:"calls"`
	ActiveCalls       int64 `json:"active_calls"`
	Tokens            int64 `json:"tokens"`
	ReservedTokens    int64 `json:"reserved_tokens"`
	UnknownUsageCalls int64 `json:"unknown_usage_calls"`
	Exhausted         bool  `json:"exhausted"`
}

// BudgetExhaustedError is returned before an upstream call when admitting it
// would violate either budget ceiling.
type BudgetExhaustedError struct {
	Reason   string
	Limits   BudgetLimits
	Snapshot BudgetSnapshot
}

func (e *BudgetExhaustedError) Error() string {
	if e == nil {
		return "looper benchmark budget exhausted"
	}
	if e.Reason == "" {
		return "looper benchmark budget exhausted"
	}
	return fmt.Sprintf("looper benchmark budget exhausted: %s", e.Reason)
}

// BudgetController reserves the complete estimated request before dispatch
// and settles it against provider usage afterward. Unknown usage is charged at
// the reservation so a missing usage block cannot silently make an algorithm
// appear cheaper.
type BudgetController struct {
	mu           sync.Mutex
	limits       BudgetLimits
	calls        int64
	activeCalls  int64
	tokens       int64
	reserved     int64
	unknownCalls int64
	nextID       int64
	exhausted    bool
	reservations map[string]budgetReservation
}

type budgetReservation struct {
	estimated int64
}

// NewBudgetController creates a concurrency-safe per-execution controller.
func NewBudgetController(limits BudgetLimits) *BudgetController {
	return &BudgetController{
		limits:       limits,
		reservations: make(map[string]budgetReservation),
	}
}

func (b *BudgetController) BeforeCall(_ context.Context, info CallInfo) (*CallReservation, error) {
	if b == nil {
		return nil, nil
	}
	estimated := info.EstimatedTotalTokens
	if estimated <= 0 {
		estimated = 1
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.limits.MaxCalls > 0 && b.calls >= b.limits.MaxCalls {
		b.exhausted = true
		return nil, b.exhaustedErrorLocked("maximum calls reached")
	}
	if b.limits.MaxTotalTokens > 0 && b.tokens+b.reserved+estimated > b.limits.MaxTotalTokens {
		b.exhausted = true
		return nil, b.exhaustedErrorLocked("estimated token reservation exceeds limit")
	}
	b.nextID++
	id := fmt.Sprintf("budget-call-%d", b.nextID)
	b.reservations[id] = budgetReservation{estimated: estimated}
	b.calls++
	b.activeCalls++
	b.reserved += estimated
	return &CallReservation{ID: id, EstimatedTokens: estimated}, nil
}

func (b *BudgetController) AfterCall(_ context.Context, _ CallInfo, reservation *CallReservation, result CallResult) {
	if b == nil || reservation == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	state, ok := b.reservations[reservation.ID]
	if !ok {
		return
	}
	delete(b.reservations, reservation.ID)
	if b.activeCalls > 0 {
		b.activeCalls--
	}
	if b.reserved >= state.estimated {
		b.reserved -= state.estimated
	} else {
		b.reserved = 0
	}
	charged, known := usageTokensForBudget(result.Response)
	if !known {
		charged = state.estimated
		b.unknownCalls++
	}
	if charged < 0 {
		charged = 0
	}
	b.tokens += charged
	if b.limits.MaxTotalTokens > 0 && b.tokens > b.limits.MaxTotalTokens {
		b.exhausted = true
	}
}

// Snapshot returns a race-free budget view.
func (b *BudgetController) Snapshot() BudgetSnapshot {
	if b == nil {
		return BudgetSnapshot{}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.snapshotLocked()
}

func (b *BudgetController) snapshotLocked() BudgetSnapshot {
	return BudgetSnapshot{
		Calls:             b.calls,
		ActiveCalls:       b.activeCalls,
		Tokens:            b.tokens,
		ReservedTokens:    b.reserved,
		UnknownUsageCalls: b.unknownCalls,
		Exhausted:         b.exhausted,
	}
}

func (b *BudgetController) exhaustedErrorLocked(reason string) *BudgetExhaustedError {
	return &BudgetExhaustedError{Reason: reason, Limits: b.limits, Snapshot: b.snapshotLocked()}
}

func usageTokensForBudget(response *ModelResponse) (int64, bool) {
	if response == nil {
		return 0, false
	}
	presence := response.UsagePresent
	if presence.TotalTokens {
		return response.Usage.TotalTokens, true
	}
	if presence.PromptTokens && presence.CompletionTokens {
		return response.Usage.PromptTokens + response.Usage.CompletionTokens, true
	}
	if presence.Any() {
		return 0, false
	}
	// Responses assembled by existing unit tests predate UsagePresent. A
	// non-zero usage value is still treated as known for compatibility; an all
	// zero value remains unknown because zero is also the omitted-usage value.
	if !response.Usage.Unreported && (response.Usage.PromptTokens != 0 || response.Usage.CompletionTokens != 0 || response.Usage.TotalTokens != 0) {
		return response.Usage.TotalTokens, true
	}
	return 0, false
}

var _ CallObserver = (*BudgetController)(nil)

// IsBudgetExhausted reports whether err is an admission failure from this
// controller or a wrapped one.
func IsBudgetExhausted(err error) bool {
	var exhausted *BudgetExhaustedError
	return errors.As(err, &exhausted)
}
