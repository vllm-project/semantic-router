package cache

import (
	"context"
	"testing"
	"time"
)

func TestLexicalPolarityConflict(t *testing.T) {
	tests := []struct {
		name     string
		cached   string
		incoming string
		want     bool
	}{
		{
			name:     "not versus positive",
			cached:   "enable caching",
			incoming: "do not enable caching",
			want:     true,
		},
		{
			name:     "contraction versus positive",
			cached:   "enable caching",
			incoming: "don't enable caching",
			want:     true,
		},
		{
			name:     "can't contraction",
			cached:   "enable caching",
			incoming: "can't enable caching",
			want:     true,
		},
		{
			name:     "won't contraction",
			cached:   "enable caching",
			incoming: "won't enable caching",
			want:     true,
		},
		{
			name:     "isn't contraction",
			cached:   "enable caching",
			incoming: "isn't enable caching",
			want:     true,
		},
		{
			name:     "doesn't contraction",
			cached:   "enable caching",
			incoming: "doesn't enable caching",
			want:     true,
		},

		{
			name:     "enable versus disable",
			cached:   "enable caching",
			incoming: "disable caching",
			want:     true,
		},
		{
			name:     "on versus off",
			cached:   "turn feature on",
			incoming: "turn feature off",
			want:     true,
		},
		{
			name:     "open versus closed",
			cached:   "keep the connection open",
			incoming: "keep the connection closed",
			want:     true,
		},
		{
			name:     "genuine paraphrase",
			cached:   "enable caching",
			incoming: "turn caching on",
			want:     false,
		},
		{
			name:     "same query",
			cached:   "enable caching",
			incoming: "enable caching",
			want:     false,
		},
		{
			name:     "different meaning",
			cached:   "enable caching",
			incoming: "configure the database",
			want:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := lexicalPolarityConflict(tt.cached, tt.incoming)
			if got != tt.want {
				t.Fatalf(
					"lexicalPolarityConflict(%q, %q) = %v, want %v",
					tt.cached,
					tt.incoming,
					got,
					tt.want,
				)
			}
		})
	}
}

func TestLexicalPolarityGuardIntegration(t *testing.T) {
	c := NewInMemoryCache(InMemoryCacheOptions{
		SimilarityThreshold: 0.80,
		MaxEntries:          16,
		TTLSeconds:          0,
		Enabled:             true,
		EvictionPolicy:      FIFOEvictionPolicyType,
	})
	t.Cleanup(func() { _ = c.Close() })

	entry := CacheEntry{
		RequestID:    "lexical-1",
		Model:        "model-x",
		Query:        "enable caching",
		ResponseBody: []byte("ENABLE-ANSWER"),
		Embedding:    []float32{1, 0, 0},
		Timestamp:    time.Now(),
	}

	c.entries = append(c.entries, entry)
	c.entryMap[entry.RequestID] = 0

	tests := []struct {
		name    string
		query   string
		wantHit bool
	}{
		{
			name:    "explicit negation",
			query:   "do not enable caching",
			wantHit: false,
		},
		{
			name:    "known antonym",
			query:   "disable caching",
			wantHit: false,
		},
		{
			name:    "genuine paraphrase",
			query:   "turn caching on",
			wantHit: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := c.finishFindSimilarSearch(
				context.Background(),
				time.Now(),
				entry.Model,
				tt.query,
				0.80,
				0,
				entry,
				1.0,
				1,
				0,
			)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			if result.Found != tt.wantHit {
				t.Fatalf("query %q: Found=%v, want %v", tt.query, result.Found, tt.wantHit)
			}
		})
	}
}
