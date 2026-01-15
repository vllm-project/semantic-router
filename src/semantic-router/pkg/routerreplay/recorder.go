package routerreplay

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

const (
	DefaultMaxRecords   = 200
	DefaultMaxBodyBytes = 4096 // 4KB
)

type Signal struct {
	Keyword      []string `json:"keyword,omitempty"`
	Embedding    []string `json:"embedding,omitempty"`
	Domain       []string `json:"domain,omitempty"`
	FactCheck    []string `json:"fact_check,omitempty"`
	UserFeedback []string `json:"user_feedback,omitempty"`
	Preference   []string `json:"preference,omitempty"`
}

type RoutingRecord struct {
	ID                    string    `json:"id"`
	Timestamp             time.Time `json:"timestamp"`
	RequestID             string    `json:"request_id,omitempty"`
	Decision              string    `json:"decision,omitempty"`
	Category              string    `json:"category,omitempty"`
	OriginalModel         string    `json:"original_model,omitempty"`
	SelectedModel         string    `json:"selected_model,omitempty"`
	ReasoningMode         string    `json:"reasoning_mode,omitempty"`
	Signals               Signal    `json:"signals"`
	RequestBody           string    `json:"request_body,omitempty"`
	ResponseBody          string    `json:"response_body,omitempty"`
	ResponseStatus        int       `json:"response_status,omitempty"`
	FromCache             bool      `json:"from_cache,omitempty"`
	Streaming             bool      `json:"streaming,omitempty"`
	RequestBodyTruncated  bool      `json:"request_body_truncated,omitempty"`
	ResponseBodyTruncated bool      `json:"response_body_truncated,omitempty"`
}

type Recorder struct {
	mu sync.Mutex

	records []*RoutingRecord
	byID    map[string]*RoutingRecord

	maxRecords   int
	maxBodyBytes int

	captureRequestBody  bool
	captureResponseBody bool
}

func NewRecorder(maxRecords int) *Recorder {
	if maxRecords <= 0 {
		maxRecords = DefaultMaxRecords
	}

	return &Recorder{
		records:      make([]*RoutingRecord, 0, maxRecords),
		byID:         make(map[string]*RoutingRecord),
		maxRecords:   maxRecords,
		maxBodyBytes: DefaultMaxBodyBytes,
	}
}

func (r *Recorder) SetCapturePolicy(captureRequest, captureResponse bool, maxBodyBytes int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.captureRequestBody = captureRequest
	r.captureResponseBody = captureResponse

	if maxBodyBytes > 0 {
		r.maxBodyBytes = maxBodyBytes
	} else {
		r.maxBodyBytes = DefaultMaxBodyBytes
	}
}

func (r *Recorder) ShouldCaptureRequest() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.captureRequestBody
}

func (r *Recorder) ShouldCaptureResponse() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.captureResponseBody
}

func (r *Recorder) SetMaxRecords(max int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if max <= 0 {
		max = DefaultMaxRecords
	}
	r.maxRecords = max

	for len(r.records) > r.maxRecords {
		oldest := r.records[0]
		delete(r.byID, oldest.ID)
		r.records = r.records[1:]
	}
}

func (r *Recorder) AddRecord(rec RoutingRecord) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if rec.ID == "" {
		id, err := generateID()
		if err != nil {
			return "", err
		}
		rec.ID = id
	}

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

	if len(r.records) >= r.maxRecords {
		oldest := r.records[0]
		delete(r.byID, oldest.ID)
		r.records = r.records[1:]
	}

	copyRec := rec
	r.records = append(r.records, &copyRec)
	r.byID[copyRec.ID] = &copyRec

	return copyRec.ID, nil
}

func (r *Recorder) UpdateStatus(id string, status int, fromCache bool, streaming bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rec, ok := r.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	if status != 0 {
		rec.ResponseStatus = status
	}
	rec.FromCache = rec.FromCache || fromCache
	rec.Streaming = rec.Streaming || streaming

	return nil
}

func (r *Recorder) AttachRequest(id string, requestBody []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rec, ok := r.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	if !r.captureRequestBody {
		return nil
	}

	body, truncated := truncateBody(requestBody, r.maxBodyBytes)
	rec.RequestBody = body
	rec.RequestBodyTruncated = rec.RequestBodyTruncated || truncated

	return nil
}

func (r *Recorder) AttachResponse(id string, responseBody []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rec, ok := r.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}

	if !r.captureResponseBody {
		return nil
	}

	body, truncated := truncateBody(responseBody, r.maxBodyBytes)
	rec.ResponseBody = body
	rec.ResponseBodyTruncated = rec.ResponseBodyTruncated || truncated

	return nil
}

// GetRecord returns a copy of the record with the given ID.
func (r *Recorder) GetRecord(id string) (RoutingRecord, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	rec, ok := r.byID[id]
	if !ok {
		return RoutingRecord{}, false
	}

	return *rec, true
}

func (r *Recorder) ListAllRecords() []RoutingRecord {
	r.mu.Lock()
	defer r.mu.Unlock()

	out := make([]RoutingRecord, 0, len(r.records))
	for _, rec := range r.records {
		out = append(out, *rec)
	}
	return out
}

func generateID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func truncateBody(body []byte, maxBytes int) (string, bool) {
	if maxBytes <= 0 || len(body) <= maxBytes {
		return string(body), false
	}
	return string(body[:maxBytes]), true
}

func (r *RoutingRecord) LogFields(event string) map[string]interface{} {
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
		},
	}

	if r.RequestBody != "" {
		fields["request_body"] = r.RequestBody
		fields["request_body_truncated"] = r.RequestBodyTruncated
	}
	if r.ResponseBody != "" {
		fields["response_body"] = r.ResponseBody
		fields["response_body_truncated"] = r.ResponseBodyTruncated
	}

	return fields
}
