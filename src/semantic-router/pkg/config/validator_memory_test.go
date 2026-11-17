package config

import (
	"strings"
	"testing"
)

func TestValidateMemoryGlobalThresholdRange(t *testing.T) {
	cases := []struct {
		name      string
		threshold float32
		wantErr   bool
	}{
		{"unset_zero_ok", 0, false},
		{"mid_ok", 0.74, false},
		{"one_ok", 1.0, false},
		{"above_one_rejected", 1.5, true},
		{"negative_rejected", -0.1, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.Memory.DefaultSimilarityThreshold = tc.threshold
			err := validateMemoryContracts(cfg)
			if tc.wantErr != (err != nil) {
				t.Fatalf("threshold=%v: wantErr=%v, got err=%v", tc.threshold, tc.wantErr, err)
			}
		})
	}
}

func TestValidateMemoryNilConfig(t *testing.T) {
	if err := validateMemoryContracts(nil); err != nil {
		t.Fatalf("nil config must be valid, got: %v", err)
	}
}

func TestValidateMemoryBackend(t *testing.T) {
	tests := []struct {
		name    string
		backend string
		setup   func(*MemoryConfig)
		wantErr string
	}{
		{name: "default backend", backend: ""},
		{name: "milvus", backend: "milvus"},
		{name: "qdrant", backend: "qdrant", setup: func(cfg *MemoryConfig) {
			cfg.Qdrant = &MemoryQdrantConfig{Host: "localhost"}
		}},
		{name: "valkey", backend: "valkey", setup: func(cfg *MemoryConfig) {
			cfg.Valkey = &MemoryValkeyConfig{Host: "localhost"}
		}},
		{name: "unknown backend", backend: "qdrnt", wantErr: "not supported"},
		{name: "qdrant block missing", backend: "qdrant", wantErr: "qdrant must be configured"},
		{name: "valkey block missing", backend: "valkey", wantErr: "valkey must be configured"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &MemoryConfig{Backend: tc.backend}
			if tc.setup != nil {
				tc.setup(cfg)
			}
			err := validateMemoryContracts(&RouterConfig{Memory: *cfg})
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("valid backend rejected: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidateMemoryGlobalRetrievalLimit(t *testing.T) {
	cases := []struct {
		name    string
		limit   int
		wantErr bool
	}{
		{"unset_zero_ok", 0, false},
		{"positive_ok", 5, false},
		{"negative_rejected", -1, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.Memory.DefaultRetrievalLimit = tc.limit
			err := validateMemoryContracts(cfg)
			if tc.wantErr != (err != nil) {
				t.Fatalf("default_retrieval_limit=%d: wantErr=%v, got err=%v", tc.limit, tc.wantErr, err)
			}
		})
	}
}
