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
}

func marshalPostgresRoutingMetadata(record Record) ([]byte, error) {
	metadata := postgresRoutingMetadata{
		SelectionMethod: record.SelectionMethod,
		CacheHitKind:    record.CacheHitKind, CacheSource: record.CacheSource,
		CacheEntryAgeSeconds: record.CacheEntryAgeSeconds, CacheTTLSeconds: record.CacheTTLSeconds,
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
	return nil
}
