package routerreplay

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const (
	DefaultMaxRecords   = 200
	DefaultMaxBodyBytes = 4096 // 4KB
)

type (
	Signal        = store.Signal
	RoutingRecord = store.Record
)

type Recorder struct {
	storage store.Storage

	maxBodyBytes int

	captureRequestBody  bool
	captureResponseBody bool
}

// NewRecorder creates a new Recorder with the specified storage backend.
func NewRecorder(storage store.Storage) *Recorder {
	return &Recorder{
		storage:      storage,
		maxBodyBytes: DefaultMaxBodyBytes,
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

	ctx := context.Background()
	return r.storage.Add(ctx, rec)
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

func LogFields(r RoutingRecord, event string) map[string]interface{} {
	fields := map[string]interface{}{
		"event":           event,
		"replay_id":       r.ID,
		"decision":        r.Decision,
		"category":        r.Category,
		"original_model":  r.OriginalModel,
		"selected_model":  r.SelectedModel,
		"reasoning_mode":  r.ReasoningMode,
		"request_id":      r.RequestID,
		"timestamp":       r.Timestamp,
		"from_cache":      r.FromCache,
		"streaming":       r.Streaming,
		"response_status": r.ResponseStatus,
		"signals": map[string]interface{}{
			"keyword":       r.Signals.Keyword,
			"embedding":     r.Signals.Embedding,
			"domain":        r.Signals.Domain,
			"fact_check":    r.Signals.FactCheck,
			"user_feedback": r.Signals.UserFeedback,
			"preference":    r.Signals.Preference,
			"language":      r.Signals.Language,
			"latency":       r.Signals.Latency,
			"context":       r.Signals.Context,
			"complexity":    r.Signals.Complexity,
		},
	}

	// Guardrails
	if r.GuardrailsEnabled || r.JailbreakEnabled || r.PIIEnabled {
		fields["guardrails_enabled"] = r.GuardrailsEnabled
		fields["jailbreak_enabled"] = r.JailbreakEnabled
		fields["pii_enabled"] = r.PIIEnabled

		// Jailbreak detection results
		if r.JailbreakDetected {
			fields["jailbreak_detected"] = r.JailbreakDetected
			fields["jailbreak_type"] = r.JailbreakType
			fields["jailbreak_confidence"] = r.JailbreakConfidence
		}

		// PII detection results
		if r.PIIDetected {
			fields["pii_detected"] = r.PIIDetected
			fields["pii_entities"] = r.PIIEntities
			fields["pii_blocked"] = r.PIIBlocked
		}
	}

	// RAG
	if r.RAGEnabled {
		fields["rag_enabled"] = r.RAGEnabled
		fields["rag_backend"] = r.RAGBackend
		fields["rag_context_length"] = r.RAGContextLength
		fields["rag_similarity_score"] = r.RAGSimilarityScore
	}

	// Hallucination detection
	if r.HallucinationEnabled {
		fields["hallucination_enabled"] = r.HallucinationEnabled
		fields["hallucination_detected"] = r.HallucinationDetected
		fields["hallucination_confidence"] = r.HallucinationConfidence
		if len(r.HallucinationSpans) > 0 {
			fields["hallucination_spans"] = r.HallucinationSpans
		}
	}

	return fields
}
