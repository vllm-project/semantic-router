package cache

import (
	"context"
	"strings"
	"testing"
)

type partitionRecordingBackend struct {
	entries map[string][]byte
}

func (b *partitionRecordingBackend) IsEnabled() bool                       { return true }
func (b *partitionRecordingBackend) CheckConnection(context.Context) error { return nil }
func (b *partitionRecordingBackend) Close() error                          { return nil }
func (b *partitionRecordingBackend) GetStats() CacheStats                  { return CacheStats{} }

func (b *partitionRecordingBackend) AddEntry(
	_ context.Context, _ string, model string, _ string, _, responseBody []byte, _ int,
) error {
	b.entries[model] = responseBody
	return nil
}

func (b *partitionRecordingBackend) LookupSimilarWithThreshold(
	_ context.Context, model string, _ string, _ float32,
) (LookupResult, error) {
	body, ok := b.entries[model]
	return LookupResult{Found: ok, ResponseBody: body}, nil
}

func TestLegacyAdapterPartitionsSemanticEntriesByCompatibilityFingerprint(t *testing.T) {
	backend := &partitionRecordingBackend{entries: make(map[string][]byte)}
	adapter := NewLegacyBackendAdapter(backend, InMemoryCacheType)
	identity := func(fingerprint string) CacheIdentity {
		return CacheIdentity{
			Partition:                CachePartition{RequestModel: "model"},
			CompatibilityFingerprint: fingerprint,
			SemanticQuery:            "what is the capital of france",
		}
	}
	if err := adapter.StoreSemantic(context.Background(), CacheWrite{
		Identity:     identity("system-a"),
		ResponseBody: []byte("paris"),
		TTL:          DefaultTTL(),
	}); err != nil {
		t.Fatal(err)
	}
	lookup := func(fingerprint string) bool {
		result, err := adapter.LookupSemantic(context.Background(), SemanticLookup{
			Identity: identity(fingerprint),
		})
		if err != nil {
			t.Fatal(err)
		}
		return result.Found
	}
	if !lookup("system-a") {
		t.Fatal("same compatibility fingerprint must hit")
	}
	if lookup("system-b") {
		t.Fatal("different compatibility fingerprint must miss")
	}
	if lookup("") {
		t.Fatal("missing compatibility fingerprint must miss")
	}
}

func TestSemanticPartitionKeyStaysWithinMilvusFieldWidth(t *testing.T) {
	identity := CacheIdentity{
		Partition: CachePartition{
			Recipe:    strings.Repeat("r", 200),
			Decision:  strings.Repeat("d", 200),
			Namespace: strings.Repeat("n", 200),
		},
		CompatibilityFingerprint: strings.Repeat("f", 64),
	}
	if got := len(identity.SemanticPartitionKey()); got != 64 {
		t.Fatalf("semantic partition key length = %d, want 64", got)
	}
}
