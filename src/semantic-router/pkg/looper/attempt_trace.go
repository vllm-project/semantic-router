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
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/openai/openai-go"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	oteltrace "go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelpricing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outputtokens"
)

const (
	ExecutionTraceVersion = 1
	maxTraceAttempts      = 100
	maxTraceBytes         = 64 * 1024
	maxTraceStringBytes   = 256
)

// AttemptStatus is the bounded terminal state of one dispatched Looper attempt.
type AttemptStatus string

// AttemptReason explains the bounded outcome of one Looper attempt.
type AttemptReason string

const (
	AttemptStatusSucceeded AttemptStatus = "succeeded"
	AttemptStatusFailed    AttemptStatus = "failed"
	AttemptStatusCancelled AttemptStatus = "cancelled"
	AttemptStatusTimedOut  AttemptStatus = "timed_out"

	AttemptReasonThresholdMet    AttemptReason = "threshold_met"
	AttemptReasonThresholdNotMet AttemptReason = "threshold_not_met"
	AttemptReasonUnusable        AttemptReason = "unusable_response"
	AttemptReasonUpstreamError   AttemptReason = "upstream_error"
	AttemptReasonInvalidResponse AttemptReason = "invalid_response"
	AttemptReasonCancelled       AttemptReason = "request_cancelled"
	AttemptReasonDeadline        AttemptReason = "deadline_exceeded"
)

// AttemptTrace contains bounded, content-free evidence for one Looper attempt.
type AttemptTrace struct {
	Ordinal int    `json:"ordinal"`
	Stage   string `json:"stage"`
	Role    string `json:"role,omitempty"`
	Model   string `json:"model,omitempty"`

	Status      AttemptStatus `json:"status"`
	Reason      AttemptReason `json:"reason,omitempty"`
	Usable      *bool         `json:"usable,omitempty"`
	Accepted    *bool         `json:"accepted,omitempty"`
	Selected    bool          `json:"selected,omitempty"`
	Synthesized bool          `json:"synthesized,omitempty"`
	Discarded   bool          `json:"discarded,omitempty"`

	VerifierType    string   `json:"verifier_type,omitempty"`
	VerifierVersion string   `json:"verifier_version,omitempty"`
	Score           *float64 `json:"score,omitempty"`
	Threshold       *float64 `json:"threshold,omitempty"`

	ReservedTokens  *int64     `json:"reserved_tokens,omitempty"`
	EstimatedTokens *int64     `json:"estimated_tokens,omitempty"`
	Usage           TokenUsage `json:"usage,omitempty"`

	EffectiveMaxOutputTokens         *int64 `json:"effective_max_output_tokens,omitempty"`
	EffectiveMaxOutputTokensSource   string `json:"effective_max_output_tokens_source,omitempty"`
	EffectiveMaxOutputTokensFallback string `json:"effective_max_output_tokens_fallback,omitempty"`

	EstimatedCost *float64 `json:"estimated_cost,omitempty"`
	ActualCost    *float64 `json:"actual_cost,omitempty"`
	Currency      string   `json:"currency,omitempty"`

	QueueLatencyMs     *int64 `json:"queue_latency_ms,omitempty"`
	FirstByteLatencyMs *int64 `json:"first_byte_latency_ms,omitempty"`
	TotalLatencyMs     int64  `json:"total_latency_ms"`
}

// ExecutionTrace groups bounded attempt evidence for one Looper execution.
type ExecutionTrace struct {
	Version             int            `json:"version"`
	TraceID             string         `json:"trace_id,omitempty"`
	Algorithm           string         `json:"algorithm"`
	Attempts            []AttemptTrace `json:"attempts,omitempty"`
	FinalAttemptOrdinal int            `json:"final_attempt_ordinal,omitempty"`
	AttemptsTruncated   bool           `json:"attempts_truncated,omitempty"`
	DroppedAttemptCount int            `json:"dropped_attempt_count,omitempty"`
	DroppedUsage        TokenUsage     `json:"dropped_usage,omitempty"`
}

// attemptSpec contains metadata known before an upstream dispatch starts.
type attemptSpec struct {
	stage           string
	role            string
	model           string
	reservedTokens  *int64
	estimatedTokens *int64
	pricing         modelpricing.Rates

	effectiveMaxOutputTokens         *int64
	effectiveMaxOutputTokensSource   string
	effectiveMaxOutputTokensFallback string
}

// attemptResult contains terminal observations used to finish an attempt.
type attemptResult struct {
	response        *ModelResponse
	err             error
	reason          AttemptReason
	usable          *bool
	accepted        *bool
	score           *float64
	threshold       *float64
	verifierType    string
	verifierVersion string
}

type attemptTrackerKey struct{}

// attemptTracker owns ordered attempt evidence for one Looper execution.
type attemptTracker struct {
	mu          sync.Mutex
	trace       ExecutionTrace
	nextOrdinal int
}

// attemptHandle coordinates timing, span completion, and trace recording.
type attemptHandle struct {
	tracker   *attemptTracker
	ordinal   int
	startedAt time.Time
	span      oteltrace.Span
	pricing   modelpricing.Rates
	stored    bool
	trace     AttemptTrace

	mu          sync.Mutex
	firstByteAt time.Time
	once        sync.Once
}

// newAttemptTracker starts the parent execution span and attaches its tracker to context.
func newAttemptTracker(ctx context.Context, algorithm string) (*attemptTracker, context.Context, oteltrace.Span) {
	algorithm = boundedTraceString(algorithm)
	ctx, span := tracing.StartSpan(ctx, "looper.execute", oteltrace.WithSpanKind(oteltrace.SpanKindInternal))
	traceID := ""
	if span.SpanContext().IsValid() {
		traceID = span.SpanContext().TraceID().String()
	}
	tracker := &attemptTracker{trace: ExecutionTrace{
		Version: ExecutionTraceVersion, TraceID: traceID, Algorithm: algorithm,
	}}
	return tracker, context.WithValue(ctx, attemptTrackerKey{}, tracker), span
}

func attemptTrackerFromContext(ctx context.Context) *attemptTracker {
	tracker, _ := ctx.Value(attemptTrackerKey{}).(*attemptTracker)
	return tracker
}

// modelAttemptSpec derives bounded dispatch metadata and optional accounting estimates.
func modelAttemptSpec(req *Request, stageReq *openai.ChatCompletionNewParams, stage, role, model string) attemptSpec {
	spec := attemptSpec{stage: stage, role: role, model: model}
	if reserve := looperOutputTokenReserve(stageReq); reserve > 0 {
		value := int64(reserve)
		spec.reservedTokens = &value
	}
	if req != nil && req.BaseContextTokens > 0 {
		if added, err := looperStageAddedMessageTokens(req.OriginalRequest, stageReq); err == nil {
			estimated := saturatingAddContextTokens(req.BaseContextTokens, added)
			if reserve := looperOutputTokenReserve(stageReq) - looperOutputTokenReserve(req.OriginalRequest); reserve > 0 {
				estimated = saturatingAddContextTokens(estimated, reserve)
			}
			value := int64(estimated)
			spec.estimatedTokens = &value
		}
	}
	spec.pricing = attemptPricing(req, model)
	limit := composeAttemptOutputTokenLimit(req, stageReq, model)
	spec.effectiveMaxOutputTokens = cloneInt64Ptr(limit.Effective)
	spec.effectiveMaxOutputTokensSource = limit.Source
	spec.effectiveMaxOutputTokensFallback = limit.Fallback
	return spec
}

func attemptPricing(req *Request, model string) modelpricing.Rates {
	if req == nil {
		return modelpricing.Rates{}
	}
	if params, ok := req.ModelParams[model]; ok {
		return modelPricingRates(params.Pricing)
	}
	for _, ref := range req.ModelRefs {
		if ref.LoRAName == model {
			return modelPricingRates(req.ModelParams[ref.Model].Pricing)
		}
	}
	return modelpricing.Rates{}
}

func modelPricingRates(pricing config.ModelPricing) modelpricing.Rates {
	return modelpricing.Rates{
		Currency: pricing.Currency, PromptPer1M: pricing.PromptPer1M,
		CachedInputPer1M: pricing.CachedInputPer1M, CacheWritePer1M: pricing.CacheWritePer1M,
		CompletionPer1M: pricing.CompletionPer1M,
	}
}

// startAttempt allocates an ordinal, starts a child span, and reserves trace storage.
func startAttempt(ctx context.Context, spec attemptSpec) (context.Context, *attemptHandle) {
	tracker := attemptTrackerFromContext(ctx)
	if tracker == nil {
		return ctx, nil
	}
	tracker.mu.Lock()
	tracker.nextOrdinal++
	ordinal := tracker.nextOrdinal
	tracker.mu.Unlock()

	spec.stage = boundedTraceString(spec.stage)
	spec.role = boundedTraceString(spec.role)
	spec.model = boundedTraceString(spec.model)
	ctx, span := tracing.StartSpan(ctx, "looper.attempt",
		oteltrace.WithSpanKind(oteltrace.SpanKindClient),
		oteltrace.WithAttributes(
			attribute.String("looper.algorithm", tracker.trace.Algorithm),
			attribute.Int("looper.attempt.ordinal", ordinal),
			attribute.String("looper.attempt.stage", spec.stage),
			attribute.String("looper.attempt.role", spec.role),
			attribute.String("looper.attempt.model", spec.model),
		),
	)
	record := AttemptTrace{
		Ordinal: ordinal, Stage: spec.stage, Role: spec.role, Model: spec.model,
		ReservedTokens: cloneInt64Ptr(spec.reservedTokens), EstimatedTokens: cloneInt64Ptr(spec.estimatedTokens),
		EffectiveMaxOutputTokens:         cloneInt64Ptr(spec.effectiveMaxOutputTokens),
		EffectiveMaxOutputTokensSource:   spec.effectiveMaxOutputTokensSource,
		EffectiveMaxOutputTokensFallback: spec.effectiveMaxOutputTokensFallback,
		EstimatedCost:                    estimatedAttemptCost(spec), Currency: attemptCurrency(spec),
	}
	handle := &attemptHandle{
		tracker: tracker, ordinal: ordinal, startedAt: time.Now(), span: span,
		pricing: spec.pricing, trace: record,
	}
	tracker.mu.Lock()
	if len(tracker.trace.Attempts) < maxTraceAttempts {
		tracker.trace.Attempts = append(tracker.trace.Attempts, record)
		handle.stored = true
	} else {
		tracker.trace.AttemptsTruncated = true
		tracker.trace.DroppedAttemptCount++
	}
	tracker.mu.Unlock()
	return context.WithValue(ctx, attemptTimingKey{}, handle), handle
}

// finish records one terminal result, emits metrics, and ends the child span exactly once.
func (a *attemptHandle) finish(result attemptResult) {
	if a == nil {
		return
	}
	a.once.Do(func() {
		status, reason := classifyAttemptError(result.err)
		if result.err == nil {
			status = AttemptStatusSucceeded
		}
		if result.reason != "" {
			reason = result.reason
		}
		a.mu.Lock()
		firstByteAt := a.firstByteAt
		a.mu.Unlock()
		totalLatency := time.Since(a.startedAt).Milliseconds()

		a.tracker.mu.Lock()
		attempt := &a.trace
		if a.stored {
			attempt = a.tracker.attemptLocked(a.ordinal)
		}
		if attempt != nil {
			attempt.Status = status
			attempt.Reason = reason
			attempt.Usable = cloneBoolPtr(result.usable)
			attempt.Accepted = cloneBoolPtr(result.accepted)
			attempt.Score = cloneFloat64Ptr(result.score)
			attempt.Threshold = cloneFloat64Ptr(result.threshold)
			attempt.VerifierType = boundedTraceString(result.verifierType)
			attempt.VerifierVersion = boundedTraceString(result.verifierVersion)
			attempt.TotalLatencyMs = totalLatency
			if !firstByteAt.IsZero() {
				latency := firstByteAt.Sub(a.startedAt).Milliseconds()
				attempt.FirstByteLatencyMs = &latency
			}
			if result.response != nil {
				attempt.Usage = result.response.Usage
				attempt.ActualCost = actualAttemptCost(a.pricing, result.response.Usage)
			}
			attempt.Discarded = (result.usable != nil && !*result.usable) ||
				(result.accepted != nil && !*result.accepted)
			if !a.stored {
				a.tracker.trace.DroppedUsage = a.tracker.trace.DroppedUsage.Add(&ModelResponse{Usage: attempt.Usage})
			}
			metrics.RecordLooperAttempt(
				a.tracker.trace.Algorithm, attempt.Stage, string(status), string(reason),
				attempt.TotalLatencyMs, attempt.FirstByteLatencyMs,
				attempt.Usage.PromptTokens, attempt.Usage.CompletionTokens,
				attempt.ActualCost, attempt.Currency,
			)
		}
		a.tracker.mu.Unlock()

		attrs := []attribute.KeyValue{
			attribute.String("looper.attempt.status", string(status)),
			attribute.String("looper.attempt.reason", string(reason)),
			attribute.Int64("looper.attempt.total_latency_ms", totalLatency),
		}
		if !firstByteAt.IsZero() {
			attrs = append(attrs, attribute.Int64("looper.attempt.first_byte_latency_ms", firstByteAt.Sub(a.startedAt).Milliseconds()))
		}
		if result.response != nil {
			attrs = append(attrs,
				attribute.Int64("looper.attempt.prompt_tokens", result.response.Usage.PromptTokens),
				attribute.Int64("looper.attempt.completion_tokens", result.response.Usage.CompletionTokens),
			)
		}
		if result.usable != nil {
			attrs = append(attrs, attribute.Bool("looper.attempt.usable", *result.usable))
		}
		if result.accepted != nil {
			attrs = append(attrs, attribute.Bool("looper.attempt.accepted", *result.accepted))
		}
		if result.score != nil {
			attrs = append(attrs, attribute.Float64("looper.attempt.score", *result.score))
		}
		if result.threshold != nil {
			attrs = append(attrs, attribute.Float64("looper.attempt.threshold", *result.threshold))
		}
		tracing.SetSpanAttributes(a.span, attrs...)
		if result.err != nil {
			tracing.SetSpanAttributes(a.span, attribute.String("error.type", string(reason)))
			a.span.SetStatus(codes.Error, string(reason))
		}
		a.span.End()
	})
}

func (t *attemptTracker) markSelected(ordinal int) {
	if t == nil || ordinal <= 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if attempt := t.attemptLocked(ordinal); attempt != nil {
		attempt.Selected = true
		attempt.Discarded = false
	}
	t.trace.FinalAttemptOrdinal = ordinal
}

func (t *attemptTracker) attemptCount() int {
	if t == nil {
		return 0
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.nextOrdinal
}

// snapshot returns an ordered, bounded copy suitable for Replay persistence.
func (t *attemptTracker) snapshot() ExecutionTrace {
	if t == nil {
		return ExecutionTrace{Version: ExecutionTraceVersion}
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	result := t.trace
	result.Attempts = append([]AttemptTrace(nil), t.trace.Attempts...)
	sort.Slice(result.Attempts, func(i, j int) bool { return result.Attempts[i].Ordinal < result.Attempts[j].Ordinal })
	return boundExecutionTrace(result)
}

func (t *attemptTracker) attemptLocked(ordinal int) *AttemptTrace {
	for i := range t.trace.Attempts {
		if t.trace.Attempts[i].Ordinal == ordinal {
			return &t.trace.Attempts[i]
		}
	}
	return nil
}

type attemptTimingKey struct{}

// recordAttemptFirstByte records the first response byte once for the active attempt.
func recordAttemptFirstByte(ctx context.Context) {
	handle, _ := ctx.Value(attemptTimingKey{}).(*attemptHandle)
	if handle == nil {
		return
	}
	handle.mu.Lock()
	if handle.firstByteAt.IsZero() {
		handle.firstByteAt = time.Now()
	}
	handle.mu.Unlock()
}

// classifyAttemptError maps transport failures to bounded status and reason values.
func classifyAttemptError(err error) (AttemptStatus, AttemptReason) {
	switch {
	case errors.Is(err, context.Canceled):
		return AttemptStatusCancelled, AttemptReasonCancelled
	case errors.Is(err, context.DeadlineExceeded):
		return AttemptStatusTimedOut, AttemptReasonDeadline
	default:
		return AttemptStatusFailed, attemptReasonFromError(err)
	}
}

// attemptReasonFromError maps an error to a bounded reason without storing its text.
func attemptReasonFromError(err error) AttemptReason {
	if err == nil {
		return ""
	}
	switch {
	case errors.Is(err, context.Canceled):
		return AttemptReasonCancelled
	case errors.Is(err, context.DeadlineExceeded):
		return AttemptReasonDeadline
	case strings.Contains(err.Error(), "parse response"):
		return AttemptReasonInvalidResponse
	default:
		return AttemptReasonUpstreamError
	}
}

// boundExecutionTrace enforces attempt count and serialized-size limits.
func boundExecutionTrace(trace ExecutionTrace) ExecutionTrace {
	bounded := trace
	bounded.Attempts = nil
	bounded.DroppedUsage = trace.DroppedUsage
	attemptBytes := 2 // JSON array brackets.
	for _, attempt := range trace.Attempts {
		encoded, _ := json.Marshal(attempt)
		additional := len(encoded)
		if len(bounded.Attempts) > 0 {
			additional++ // comma
		}
		if len(bounded.Attempts) >= maxTraceAttempts || attemptBytes+additional > maxTraceBytes {
			bounded.AttemptsTruncated = true
			bounded.DroppedAttemptCount++
			bounded.DroppedUsage = bounded.DroppedUsage.Add(&ModelResponse{Usage: attempt.Usage})
			continue
		}
		bounded.Attempts = append(bounded.Attempts, attempt)
		attemptBytes += additional
	}
	return bounded
}

func boundedTraceString(value string) string {
	value = strings.TrimSpace(value)
	if len(value) > maxTraceStringBytes {
		return value[:maxTraceStringBytes]
	}
	return value
}

func cloneInt64Ptr(value *int64) *int64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneBoolPtr(value *bool) *bool {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func cloneFloat64Ptr(value *float64) *float64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func attemptCurrency(spec attemptSpec) string {
	if !spec.pricing.IsConfigured() {
		return ""
	}
	return spec.pricing.Currency
}

func composeAttemptOutputTokenLimit(
	req *Request,
	stageReq *openai.ChatCompletionNewParams,
	model string,
) outputtokens.Result {
	sources := outputtokens.Sources{
		AlgorithmStage: stageOverrideMaxOutputTokens(nil, stageReq),
	}
	if req != nil {
		sources.Client = chatParamsMaxOutputTokens(req.OriginalRequest)
		sources.Plugin = outputtokens.Clone(req.PluginMaxOutputTokens)
		sources.ModelRef = modelRefMaxCompletionTokens(req.ModelRefs, model)
		sources.AlgorithmStage = stageOverrideMaxOutputTokens(req.OriginalRequest, stageReq)
	}
	return outputtokens.Compose(sources)
}

func modelRefMaxCompletionTokens(refs []config.ModelRef, model string) *int64 {
	for _, ref := range refs {
		if ref.Model == model || ref.LoRAName == model {
			return outputtokens.FromInt(ref.MaxCompletionTokens)
		}
	}
	return nil
}

// estimatedAttemptCost prices the estimated input and reserved output tokens.
func estimatedAttemptCost(spec attemptSpec) *float64 {
	if !spec.pricing.IsConfigured() || spec.estimatedTokens == nil {
		return nil
	}
	reserved := int64(0)
	if spec.reservedTokens != nil {
		reserved = *spec.reservedTokens
	}
	input := max(*spec.estimatedTokens-reserved, 0)
	cost := modelpricing.Cost(modelpricing.Usage{
		PromptTokens: saturatingInt64ToInt(input), CompletionTokens: saturatingInt64ToInt(reserved),
	}, spec.pricing)
	return &cost
}

func actualAttemptCost(pricing modelpricing.Rates, usage TokenUsage) *float64 {
	if !pricing.IsConfigured() {
		return nil
	}
	cost := modelpricing.Cost(modelpricing.Usage{
		PromptTokens:     saturatingInt64ToInt(usage.PromptTokens),
		CompletionTokens: saturatingInt64ToInt(usage.CompletionTokens),
	}, pricing)
	return &cost
}
