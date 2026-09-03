package cache

import (
	"strings"
	"testing"
)

func TestExactCacheRecordIDIsDeterministicAndBackendSafe(t *testing.T) {
	first := exactCacheRecordID("partition-a", "fingerprint")
	second := exactCacheRecordID("partition-a", "fingerprint")
	otherPartition := exactCacheRecordID("partition-b", "fingerprint")

	if first != second {
		t.Fatal("exact record ID must be deterministic")
	}
	if len(first) != 64 || !isHexString(first) {
		t.Fatalf("exact record ID must be 64-char hex, got %q", first)
	}
	if first == otherPartition {
		t.Fatal("partition must participate in exact record ID")
	}
	if exactCacheStorageKey("partition-a", "fingerprint") != exactCacheKeyPrefix+first {
		t.Fatal("KV storage key and shared-backend record ID diverged")
	}
}

func TestMilvusSemanticFilterExcludesExactEntries(t *testing.T) {
	filter := milvusActiveEntryFilterExpr("partition")
	if !strings.Contains(filter, "query !=") ||
		!strings.Contains(filter, exactCacheQueryMarker) {
		t.Fatalf("semantic filter does not exclude exact entries: %s", filter)
	}
}

func TestParseValkeyHashFields(t *testing.T) {
	t.Run("map[string]interface{}", func(t *testing.T) {
		raw := map[string]interface{}{
			"response_body": `{"ok":true}`,
			"timestamp":     1700000000,
			"expires_at":    1700003600,
			"ttl_seconds":   3600,
		}
		fields := parseValkeyHashFields(raw)
		if fields["response_body"] != `{"ok":true}` || fields["timestamp"] != "1700000000" {
			t.Fatalf("unexpected fields parsed: %#v", fields)
		}
	})

	t.Run("[]interface{}", func(t *testing.T) {
		raw := []interface{}{
			"response_body", `{"ok":true}`,
			"timestamp", int64(1700000000),
			"expires_at", int64(1700003600),
		}
		fields := parseValkeyHashFields(raw)
		if fields["response_body"] != `{"ok":true}` || fields["timestamp"] != "1700000000" {
			t.Fatalf("unexpected fields parsed: %#v", fields)
		}
	})
}
