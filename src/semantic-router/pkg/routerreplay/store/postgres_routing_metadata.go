package store

import "encoding/json"

// Keep routing and cache diagnostics separate from safety evidence. A nullable
// confidence distinguishes an actual zero score from an unavailable score.
type postgresRoutingMetadata struct {
	ConfidenceScore      *float64 `json:"confidence_score,omitempty"`
	SelectionMethod      string   `json:"selection_method,omitempty"`
	CacheHitKind         string   `json:"cache_hit_kind,omitempty"`
	CacheSource          string   `json:"cache_source,omitempty"`
	CacheEntryAgeSeconds float64  `json:"cache_entry_age_seconds,omitempty"`
	CacheTTLSeconds      int      `json:"cache_ttl_seconds,omitempty"`
	// The record's place in its agent run, the name it gave the run's
	// session and the effort it was sent with.
	Agent           *AgentIdentity `json:"agent,omitempty"`
	ParentRequestID string         `json:"parent_request_id,omitempty"`
	SessionName     string         `json:"session_name,omitempty"`
	ReasoningEffort string         `json:"reasoning_effort,omitempty"`
	// The images in the request's newest turn and in its history.
	ImageCount        int `json:"image_count,omitempty"`
	HistoryImageCount int `json:"history_image_count,omitempty"`
	// The model the client named when its listener routed the request as
	// another.
	RequestedModel string `json:"requested_model,omitempty"`
}

func marshalPostgresRoutingMetadata(record Record) ([]byte, error) {
	metadata := postgresRoutingMetadata{
		SelectionMethod: record.SelectionMethod,
		CacheHitKind:    record.CacheHitKind, CacheSource: record.CacheSource,
		CacheEntryAgeSeconds: record.CacheEntryAgeSeconds, CacheTTLSeconds: record.CacheTTLSeconds,
		Agent: record.Agent, ParentRequestID: record.ParentRequestID, SessionName: record.SessionName,
		ReasoningEffort: record.ReasoningEffort, RequestedModel: record.RequestedModel,
		ImageCount: record.ImageCount, HistoryImageCount: record.HistoryImageCount,
	}
	if record.ConfidenceScoreAvailable {
		metadata.ConfidenceScore = &record.ConfidenceScore
	}
	return json.Marshal(metadata)
}

func unmarshalPostgresRoutingMetadata(encoded []byte, record *Record) error {
	var metadata postgresRoutingMetadata
	if err := unmarshalReplayOptionalJSON(encoded, &metadata); err != nil {
		return err
	}
	// Older rows persisted availability without the numeric confidence. Do not
	// present their missing score as a measured zero after a schema upgrade.
	record.ConfidenceScore = 0
	record.ConfidenceScoreAvailable = metadata.ConfidenceScore != nil
	if metadata.ConfidenceScore != nil {
		record.ConfidenceScore = *metadata.ConfidenceScore
	}
	record.SelectionMethod = metadata.SelectionMethod
	record.CacheHitKind = metadata.CacheHitKind
	record.CacheSource = metadata.CacheSource
	record.CacheEntryAgeSeconds = metadata.CacheEntryAgeSeconds
	record.CacheTTLSeconds = metadata.CacheTTLSeconds
	record.Agent, record.ParentRequestID, record.SessionName = metadata.Agent, metadata.ParentRequestID, metadata.SessionName
	record.ReasoningEffort = metadata.ReasoningEffort
	record.ImageCount, record.HistoryImageCount = metadata.ImageCount, metadata.HistoryImageCount
	record.RequestedModel = metadata.RequestedModel
	return nil
}
