package routerreplay

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
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

// RecorderConfig contains configuration for creating a Recorder with store support.
type RecorderConfig struct {
	// StoreConfig is the configuration for the storage backend.
	StoreConfig store.StoreConfig

	// AsyncConfig is the configuration for async writes.
	AsyncConfig store.AsyncWriterConfig

	// UseAsyncWrites enables asynchronous writes to the store.
	UseAsyncWrites bool

	// MaxBodyBytes caps how many bytes of request/response body are recorded.
	MaxBodyBytes int

	// CaptureRequestBody controls whether request bodies are captured.
	CaptureRequestBody bool

	// CaptureResponseBody controls whether response bodies are captured.
	CaptureResponseBody bool
}

type Recorder struct {
	mu sync.Mutex

	// Legacy in-memory storage (used when store is nil)
	records []*RoutingRecord
	byID    map[string]*RoutingRecord

	// Store backend (when configured)
	store       store.ReplayStore
	asyncWriter *store.AsyncWriter
	useAsync    bool

	maxRecords   int
	maxBodyBytes int

	captureRequestBody  bool
	captureResponseBody bool
}

// NewRecorder creates a new Recorder with in-memory storage.
// This maintains backward compatibility with the original implementation.
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

// NewRecorderWithStore creates a new Recorder with pluggable storage backend.
func NewRecorderWithStore(config RecorderConfig) (*Recorder, error) {
	replayStore, err := store.NewStore(config.StoreConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create replay store: %w", err)
	}

	maxBodyBytes := config.MaxBodyBytes
	if maxBodyBytes <= 0 {
		maxBodyBytes = DefaultMaxBodyBytes
	}

	r := &Recorder{
		store:               replayStore,
		useAsync:            config.UseAsyncWrites,
		maxBodyBytes:        maxBodyBytes,
		captureRequestBody:  config.CaptureRequestBody,
		captureResponseBody: config.CaptureResponseBody,
		maxRecords:          config.StoreConfig.MaxRecords,
		// Initialize legacy fields for fallback
		records: make([]*RoutingRecord, 0),
		byID:    make(map[string]*RoutingRecord),
	}

	if config.UseAsyncWrites && replayStore.IsEnabled() {
		r.asyncWriter = store.NewAsyncWriter(replayStore, config.AsyncConfig)
		r.asyncWriter.Start()
	}

	return r, nil
}

// Close releases resources held by the recorder.
func (r *Recorder) Close() error {
	if r.asyncWriter != nil {
		r.asyncWriter.Stop()
	}
	if r.store != nil {
		return r.store.Close()
	}
	return nil
}

// HasStore returns true if the recorder is using a storage backend.
func (r *Recorder) HasStore() bool {
	return r.store != nil && r.store.IsEnabled()
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

	// Use store backend if available
	if r.store != nil && r.store.IsEnabled() {
		storeRec := toStoreRecord(&rec)
		if r.useAsync && r.asyncWriter != nil {
			r.asyncWriter.EnqueueStore(storeRec)
		} else {
			if err := r.store.StoreRecord(context.Background(), storeRec); err != nil {
				return "", err
			}
		}
		// Also keep in local cache for fast lookups during request processing
		r.byID[rec.ID] = &rec
		return rec.ID, nil
	}

	// Legacy in-memory storage
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
		// Try to get from store if not in local cache
		if r.store != nil && r.store.IsEnabled() {
			storeRec, err := r.store.GetRecord(context.Background(), id)
			if err != nil {
				return fmt.Errorf("record with ID %s not found", id)
			}
			rec = fromStoreRecord(storeRec)
			r.byID[id] = rec
		} else {
			return fmt.Errorf("record with ID %s not found", id)
		}
	}

	if status != 0 {
		rec.ResponseStatus = status
	}
	rec.FromCache = rec.FromCache || fromCache
	rec.Streaming = rec.Streaming || streaming

	// Update in store if available
	if r.store != nil && r.store.IsEnabled() {
		storeRec := toStoreRecord(rec)
		if r.useAsync && r.asyncWriter != nil {
			r.asyncWriter.EnqueueUpdate(storeRec)
		} else {
			_ = r.store.UpdateRecord(context.Background(), storeRec)
		}
	}

	return nil
}

func (r *Recorder) AttachRequest(id string, requestBody []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rec, ok := r.byID[id]
	if !ok {
		// Try to get from store if not in local cache
		if r.store != nil && r.store.IsEnabled() {
			storeRec, err := r.store.GetRecord(context.Background(), id)
			if err != nil {
				return fmt.Errorf("record with ID %s not found", id)
			}
			rec = fromStoreRecord(storeRec)
			r.byID[id] = rec
		} else {
			return fmt.Errorf("record with ID %s not found", id)
		}
	}

	if !r.captureRequestBody {
		return nil
	}

	body, truncated := truncateBody(requestBody, r.maxBodyBytes)
	rec.RequestBody = body
	rec.RequestBodyTruncated = rec.RequestBodyTruncated || truncated

	// Update in store if available
	if r.store != nil && r.store.IsEnabled() {
		storeRec := toStoreRecord(rec)
		if r.useAsync && r.asyncWriter != nil {
			r.asyncWriter.EnqueueUpdate(storeRec)
		} else {
			_ = r.store.UpdateRecord(context.Background(), storeRec)
		}
	}

	return nil
}

func (r *Recorder) AttachResponse(id string, responseBody []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	rec, ok := r.byID[id]
	if !ok {
		// Try to get from store if not in local cache
		if r.store != nil && r.store.IsEnabled() {
			storeRec, err := r.store.GetRecord(context.Background(), id)
			if err != nil {
				return fmt.Errorf("record with ID %s not found", id)
			}
			rec = fromStoreRecord(storeRec)
			r.byID[id] = rec
		} else {
			return fmt.Errorf("record with ID %s not found", id)
		}
	}

	if !r.captureResponseBody {
		return nil
	}

	body, truncated := truncateBody(responseBody, r.maxBodyBytes)
	rec.ResponseBody = body
	rec.ResponseBodyTruncated = rec.ResponseBodyTruncated || truncated

	// Update in store if available
	if r.store != nil && r.store.IsEnabled() {
		storeRec := toStoreRecord(rec)
		if r.useAsync && r.asyncWriter != nil {
			r.asyncWriter.EnqueueUpdate(storeRec)
		} else {
			_ = r.store.UpdateRecord(context.Background(), storeRec)
		}
	}

	return nil
}

// GetRecord returns a copy of the record with the given ID.
func (r *Recorder) GetRecord(id string) (RoutingRecord, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// First check local cache
	rec, ok := r.byID[id]
	if ok {
		return *rec, true
	}

	// Try store if available
	if r.store != nil && r.store.IsEnabled() {
		storeRec, err := r.store.GetRecord(context.Background(), id)
		if err == nil {
			return *fromStoreRecord(storeRec), true
		}
	}

	return RoutingRecord{}, false
}

// ListAllRecords returns all records (legacy method for backward compatibility).
// For large datasets, prefer using ListRecords with pagination.
func (r *Recorder) ListAllRecords() []RoutingRecord {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Use store if available
	if r.store != nil && r.store.IsEnabled() {
		result, err := r.store.ListRecords(context.Background(), store.ListOptions{
			Limit: store.MaxListLimit,
			Order: "desc",
		})
		if err == nil && result != nil {
			out := make([]RoutingRecord, 0, len(result.Records))
			for _, rec := range result.Records {
				out = append(out, *fromStoreRecord(rec))
			}
			return out
		}
	}

	// Legacy in-memory storage
	out := make([]RoutingRecord, 0, len(r.records))
	for _, rec := range r.records {
		out = append(out, *rec)
	}
	return out
}

// ListRecords returns records with pagination and filtering support.
func (r *Recorder) ListRecords(opts store.ListOptions) (*store.ListResult, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Use store if available
	if r.store != nil && r.store.IsEnabled() {
		return r.store.ListRecords(context.Background(), opts)
	}

	// Legacy in-memory storage - convert to store format
	limit := opts.Limit
	if limit <= 0 {
		limit = store.DefaultListLimit
	}
	if limit > store.MaxListLimit {
		limit = store.MaxListLimit
	}

	// Convert local records to store records
	storeRecords := make([]*store.RoutingRecord, 0, len(r.records))
	for _, rec := range r.records {
		storeRecords = append(storeRecords, toStoreRecord(rec))
	}

	// Apply basic pagination
	startIdx := 0
	if opts.After != "" {
		for i, rec := range storeRecords {
			if rec.ID == opts.After {
				startIdx = i + 1
				break
			}
		}
	}

	if startIdx >= len(storeRecords) {
		return &store.ListResult{Records: nil, HasMore: false}, nil
	}

	// Slice and check for more
	remaining := storeRecords[startIdx:]
	hasMore := len(remaining) > limit
	if hasMore {
		remaining = remaining[:limit]
	}

	result := &store.ListResult{
		Records: remaining,
		HasMore: hasMore,
	}
	if len(remaining) > 0 {
		result.FirstID = remaining[0].ID
		result.LastID = remaining[len(remaining)-1].ID
	}

	return result, nil
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

// toStoreRecord converts a RoutingRecord to a store.RoutingRecord.
func toStoreRecord(rec *RoutingRecord) *store.RoutingRecord {
	return &store.RoutingRecord{
		ID:            rec.ID,
		Timestamp:     rec.Timestamp,
		RequestID:     rec.RequestID,
		Decision:      rec.Decision,
		Category:      rec.Category,
		OriginalModel: rec.OriginalModel,
		SelectedModel: rec.SelectedModel,
		ReasoningMode: rec.ReasoningMode,
		Signals: store.Signal{
			Keyword:      rec.Signals.Keyword,
			Embedding:    rec.Signals.Embedding,
			Domain:       rec.Signals.Domain,
			FactCheck:    rec.Signals.FactCheck,
			UserFeedback: rec.Signals.UserFeedback,
			Preference:   rec.Signals.Preference,
		},
		RequestBody:           rec.RequestBody,
		ResponseBody:          rec.ResponseBody,
		ResponseStatus:        rec.ResponseStatus,
		FromCache:             rec.FromCache,
		Streaming:             rec.Streaming,
		RequestBodyTruncated:  rec.RequestBodyTruncated,
		ResponseBodyTruncated: rec.ResponseBodyTruncated,
	}
}

// fromStoreRecord converts a store.RoutingRecord to a RoutingRecord.
func fromStoreRecord(rec *store.RoutingRecord) *RoutingRecord {
	return &RoutingRecord{
		ID:            rec.ID,
		Timestamp:     rec.Timestamp,
		RequestID:     rec.RequestID,
		Decision:      rec.Decision,
		Category:      rec.Category,
		OriginalModel: rec.OriginalModel,
		SelectedModel: rec.SelectedModel,
		ReasoningMode: rec.ReasoningMode,
		Signals: Signal{
			Keyword:      rec.Signals.Keyword,
			Embedding:    rec.Signals.Embedding,
			Domain:       rec.Signals.Domain,
			FactCheck:    rec.Signals.FactCheck,
			UserFeedback: rec.Signals.UserFeedback,
			Preference:   rec.Signals.Preference,
		},
		RequestBody:           rec.RequestBody,
		ResponseBody:          rec.ResponseBody,
		ResponseStatus:        rec.ResponseStatus,
		FromCache:             rec.FromCache,
		Streaming:             rec.Streaming,
		RequestBodyTruncated:  rec.RequestBodyTruncated,
		ResponseBodyTruncated: rec.ResponseBodyTruncated,
	}
}
