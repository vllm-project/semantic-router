package routerreplay

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const (
	DefaultMaxRecords        = 200
	DefaultMaxBodyBytes      = 4096 // 4KB
	DefaultMaxToolTraceBytes = 0    // No limit — structured fields are typically small
)

type (
	Signal        = store.Signal
	RoutingRecord = store.Record
	ToolTrace     = store.ToolTrace
	ToolTraceStep = store.ToolTraceStep
	UsageCost     = store.UsageCost
)

type Recorder struct {
	storage store.Storage

	maxBodyBytes      int
	maxToolTraceBytes int // 0 = no limit

	captureRequestBody  bool
	captureResponseBody bool
}

// NewRecorder creates a new Recorder with the specified storage backend.
func NewRecorder(storage store.Storage) *Recorder {
	return &Recorder{
		storage:           storage,
		maxBodyBytes:      DefaultMaxBodyBytes,
		maxToolTraceBytes: DefaultMaxToolTraceBytes,
	}
}

func (r *Recorder) SetCapturePolicy(captureRequest, captureResponse bool, maxBodyBytes int) {
	r.captureRequestBody = captureRequest
	r.captureResponseBody = captureResponse

	if maxBodyBytes > 0 {
		r.maxBodyBytes = maxBodyBytes
	} else {
		r.maxBodyBytes = DefaultMaxBodyBytes
	}
}

// SetMaxToolTraceBytes sets the per-field byte limit for structured tool-trace
// fields (Prompt, ToolDefinitions, ToolTraceStep.Arguments, ToolTraceStep.Output).
// A value of 0 disables truncation for those fields.
func (r *Recorder) SetMaxToolTraceBytes(max int) {
	if max >= 0 {
		r.maxToolTraceBytes = max
	}
}

func (r *Recorder) ShouldCaptureRequest() bool {
	return r.captureRequestBody
}

func (r *Recorder) ShouldCaptureResponse() bool {
	return r.captureResponseBody
}

func (r *Recorder) SetMaxRecords(max int) {
	if memStore, ok := r.storage.(*store.MemoryStore); ok {
		memStore.SetMaxRecords(max)
	}
}

func (r *Recorder) AddRecord(rec RoutingRecord) (string, error) {
	if rec.Timestamp.IsZero() {
		rec.Timestamp = time.Now().UTC()
	}

	if r.captureRequestBody && len(rec.RequestBody) > r.maxBodyBytes {
		rec.RequestBody = rec.RequestBody[:r.maxBodyBytes]
		rec.RequestBodyTruncated = true
	}

	if r.captureResponseBody && len(rec.ResponseBody) > r.maxBodyBytes {
		rec.ResponseBody = rec.ResponseBody[:r.maxBodyBytes]
		rec.ResponseBodyTruncated = true
	}

	// Apply MaxToolTraceBytes to structured tool-trace fields.
	// These are truncated independently of the raw body fields so that
	// callers can keep MaxBodyBytes small without losing structured data.
	applyMaxToolTraceBytes(&rec, r.maxToolTraceBytes)

	ctx := context.Background()
	return r.storage.Add(ctx, rec)
}

// applyMaxToolTraceBytes truncates structured tool-trace text fields to max
// bytes. A non-positive max disables truncation.
func applyMaxToolTraceBytes(rec *RoutingRecord, max int) {
	if max <= 0 {
		return
	}
	if len(rec.Prompt) > max {
		rec.Prompt = rec.Prompt[:max]
		rec.PromptTruncated = true
	}
	if len(rec.ToolDefinitions) > max {
		rec.ToolDefinitions = rec.ToolDefinitions[:max]
	}
	truncateToolTraceSteps(rec.ToolTrace, max)
}

// truncateToolTraceSteps applies the byte limit to each step's Arguments and
// Output fields.
func truncateToolTraceSteps(trace *ToolTrace, max int) {
	if trace == nil {
		return
	}
	for i := range trace.Steps {
		truncateToolTraceStep(&trace.Steps[i], max)
	}
}

// truncateToolTraceStep applies the byte limit to a single step's fields.
func truncateToolTraceStep(step *ToolTraceStep, max int) {
	if len(step.Arguments) > max {
		step.Arguments = step.Arguments[:max]
		step.RawArguments = step.Arguments
		step.Truncated = true
	}
	if len(step.Output) > max {
		step.Output = step.Output[:max]
		step.RawOutput = step.Output
		step.Truncated = true
	}
}

func (r *Recorder) UpdateStatus(id string, status int, fromCache bool, streaming bool) error {
	ctx := context.Background()
	return r.storage.UpdateStatus(ctx, id, status, fromCache, streaming)
}

func (r *Recorder) AttachRequest(id string, requestBody []byte) error {
	if !r.captureRequestBody {
		return nil
	}

	body, truncated := truncateBody(requestBody, r.maxBodyBytes)
	ctx := context.Background()
	return r.storage.AttachRequest(ctx, id, body, truncated)
}

func (r *Recorder) AttachResponse(id string, responseBody []byte) error {
	if !r.captureResponseBody {
		return nil
	}

	body, truncated := truncateBody(responseBody, r.maxBodyBytes)
	ctx := context.Background()
	return r.storage.AttachResponse(ctx, id, body, truncated)
}

// UpdateHallucinationStatus updates hallucination detection results for a record.
func (r *Recorder) UpdateHallucinationStatus(id string, detected bool, confidence float32, spans []string) error {
	ctx := context.Background()
	return r.storage.UpdateHallucinationStatus(ctx, id, detected, confidence, spans)
}

func (r *Recorder) UpdateUsageCost(id string, usage UsageCost) error {
	ctx := context.Background()
	return r.storage.UpdateUsageCost(ctx, id, usage)
}

func (r *Recorder) UpdateToolTrace(id string, trace ToolTrace) error {
	ctx := context.Background()
	return r.storage.UpdateToolTrace(ctx, id, trace)
}

// Reader returns the underlying store.Reader so that read-only consumers (e.g.
// lookup table builders) can query historical records without needing write access.
func (r *Recorder) Reader() store.Reader {
	return r.storage
}

// GetRecord returns a copy of the record with the given ID.
func (r *Recorder) GetRecord(id string) (RoutingRecord, bool) {
	ctx := context.Background()
	rec, found, err := r.storage.Get(ctx, id)
	if err != nil {
		return RoutingRecord{}, false
	}
	return rec, found
}

func (r *Recorder) ListAllRecords() []RoutingRecord {
	ctx := context.Background()
	records, err := r.storage.List(ctx)
	if err != nil {
		return []RoutingRecord{}
	}
	return records
}

// Releases resources held by the storage backend.
func (r *Recorder) Close() error {
	return r.storage.Close()
}

func truncateBody(body []byte, maxBytes int) (string, bool) {
	if maxBytes <= 0 || len(body) <= maxBytes {
		return string(body), false
	}
	return string(body[:maxBytes]), true
}

func logSignalFields(signals Signal) map[string]interface{} {
	return map[string]interface{}{
		"keyword":       signals.Keyword,
		"embedding":     signals.Embedding,
		"domain":        signals.Domain,
		"fact_check":    signals.FactCheck,
		"user_feedback": signals.UserFeedback,
		"reask":         signals.Reask,
		"preference":    signals.Preference,
		"language":      signals.Language,
		"context":       signals.Context,
		"structure":     signals.Structure,
		"complexity":    signals.Complexity,
		"modality":      signals.Modality,
		"authz":         signals.Authz,
		"jailbreak":     signals.Jailbreak,
		"pii":           signals.PII,
		"kb":            signals.KB,
	}
}

func appendGuardrailLogFields(fields map[string]interface{}, r RoutingRecord) {
	if !r.GuardrailsEnabled && !r.JailbreakEnabled && !r.PIIEnabled {
		return
	}

	fields["guardrails_enabled"] = r.GuardrailsEnabled
	fields["jailbreak_enabled"] = r.JailbreakEnabled
	fields["pii_enabled"] = r.PIIEnabled

	if r.JailbreakDetected {
		fields["jailbreak_detected"] = r.JailbreakDetected
		fields["jailbreak_type"] = r.JailbreakType
		fields["jailbreak_confidence"] = r.JailbreakConfidence
	}
	if r.ResponseJailbreakDetected {
		fields["response_jailbreak_detected"] = r.ResponseJailbreakDetected
		fields["response_jailbreak_type"] = r.ResponseJailbreakType
		fields["response_jailbreak_confidence"] = r.ResponseJailbreakConfidence
	}
	if r.PIIDetected {
		fields["pii_detected"] = r.PIIDetected
		fields["pii_entities"] = r.PIIEntities
		fields["pii_blocked"] = r.PIIBlocked
	}
}

func appendRAGLogFields(fields map[string]interface{}, r RoutingRecord) {
	if !r.RAGEnabled {
		return
	}

	fields["rag_enabled"] = r.RAGEnabled
	fields["rag_backend"] = r.RAGBackend
	fields["rag_context_length"] = r.RAGContextLength
	fields["rag_similarity_score"] = r.RAGSimilarityScore
}

func appendHallucinationLogFields(fields map[string]interface{}, r RoutingRecord) {
	if !r.HallucinationEnabled {
		return
	}

	fields["hallucination_enabled"] = r.HallucinationEnabled
	fields["hallucination_detected"] = r.HallucinationDetected
	fields["hallucination_confidence"] = r.HallucinationConfidence
	if len(r.HallucinationSpans) > 0 {
		fields["hallucination_spans"] = r.HallucinationSpans
	}
}

func appendUsageCostLogFields(fields map[string]interface{}, r RoutingRecord) {
	if r.PromptTokens != nil {
		fields["prompt_tokens"] = *r.PromptTokens
	}
	if r.CompletionTokens != nil {
		fields["completion_tokens"] = *r.CompletionTokens
	}
	if r.TotalTokens != nil {
		fields["total_tokens"] = *r.TotalTokens
	}
	if r.ActualCost != nil {
		fields["actual_cost"] = *r.ActualCost
	}
	if r.BaselineCost != nil {
		fields["baseline_cost"] = *r.BaselineCost
	}
	if r.CostSavings != nil {
		fields["cost_savings"] = *r.CostSavings
	}
	if r.Currency != nil {
		fields["currency"] = *r.Currency
	}
	if r.BaselineModel != nil {
		fields["baseline_model"] = *r.BaselineModel
	}
}

func LogFields(r RoutingRecord, event string) map[string]interface{} {
	fields := map[string]interface{}{
		"event":             event,
		"replay_id":         r.ID,
		"decision":          r.Decision,
		"decision_tier":     r.DecisionTier,
		"decision_priority": r.DecisionPriority,
		"category":          r.Category,
		"original_model":    r.OriginalModel,
		"selected_model":    r.SelectedModel,
		"reasoning_mode":    r.ReasoningMode,
		"confidence_score":  r.ConfidenceScore,
		"selection_method":  r.SelectionMethod,
		"request_id":        r.RequestID,
		"timestamp":         r.Timestamp,
		"turn_index":        r.TurnIndex,
		"from_cache":        r.FromCache,
		"streaming":         r.Streaming,
		"response_status":   r.ResponseStatus,
		"signals":           logSignalFields(r.Signals),
	}
	if r.SessionID != "" {
		fields["session_id"] = r.SessionID
	}
	if r.ConversationID != "" {
		fields["conversation_id"] = r.ConversationID
	}
	if r.PreviousResponseID != "" {
		fields["previous_response_id"] = r.PreviousResponseID
	}
	if len(r.Projections) > 0 {
		fields["projections"] = r.Projections
	}
	if len(r.ProjectionScores) > 0 {
		fields["projection_scores"] = r.ProjectionScores
	}
	if len(r.SignalConfidences) > 0 {
		fields["signal_confidences"] = r.SignalConfidences
	}
	if len(r.SignalValues) > 0 {
		fields["signal_values"] = r.SignalValues
	}
	appendToolTraceLogFields(fields, r.ToolTrace)

	appendGuardrailLogFields(fields, r)
	appendRAGLogFields(fields, r)
	appendHallucinationLogFields(fields, r)
	appendUsageCostLogFields(fields, r)
	return fields
}

func appendToolTraceLogFields(fields map[string]interface{}, trace *ToolTrace) {
	if trace == nil {
		return
	}
	if trace.Flow != "" {
		fields["tool_trace_flow"] = trace.Flow
	}
	if trace.Stage != "" {
		fields["tool_trace_stage"] = trace.Stage
	}
	if len(trace.ToolNames) > 0 {
		fields["tool_names"] = trace.ToolNames
	}
	if len(trace.Steps) > 0 {
		fields["tool_trace_step_count"] = len(trace.Steps)
	}
}
