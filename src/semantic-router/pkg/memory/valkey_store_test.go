package memory

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Verify ValkeyStore satisfies the Store interface at compile time.
var _ Store = (*ValkeyStore)(nil)

func TestValkeyClusterInfoIndicatesMode(t *testing.T) {
	t.Parallel()

	assert.True(t, valkeyClusterInfoIndicatesMode("cluster_state:ok\ncluster_slots_assigned:16384"))
	assert.True(t, valkeyClusterInfoIndicatesMode([]byte("cluster_state:fail")))
	assert.False(t, valkeyClusterInfoIndicatesMode("role:master"))
}

func TestValkeyClusterModeDisablesAtomicConsolidation(t *testing.T) {
	store := &ValkeyStore{enabled: true}
	assert.True(t, store.supportsAtomicGroupReplacement())

	store.clusterMode = true
	assert.False(t, store.supportsAtomicGroupReplacement())
}

func TestValkeyReplaceCurrentGroupArgs(t *testing.T) {
	createdAt := time.Unix(1_700_000_000, 0).UTC()
	updatedAt := createdAt.Add(time.Second)
	versions := []memoryVersion{{
		id:         "a",
		userID:     "user-1",
		projectID:  "project-1",
		typ:        MemoryTypeSemantic,
		content:    "alpha beta",
		createdAt:  createdAt,
		updatedAt:  updatedAt,
		importance: 0.5,
	}}
	args := valkeyReplaceCurrentGroupArgs(versions, map[string]string{
		"content": "summary",
		"id":      "summary-id",
	})

	require.Equal(t, "1", args[0])
	require.Equal(t, valkeySourceVersionArgs(versions[0]), args[1:9])
	require.Equal(t, "2", args[9])
	require.Equal(t, []string{"content", "summary", "id", "summary-id"}, args[10:])
}

func TestValkeySearchProjectionsIncludeAccessFields(t *testing.T) {
	t.Parallel()

	store := &ValkeyStore{indexName: "mem_idx"}
	retrieve := store.buildRetrieveSearchCmd(RetrieveOptions{UserID: "user-1"}, []float32{0.1}, 5)
	assertSearchReturnFields(t, retrieve, []string{
		"id", "content", "memory_type", "metadata", "vector_distance", "last_accessed", "access_count",
	})

	list := store.buildListSearchCmd("@user_id:{user-1}", 0, 10)
	assertSearchReturnFields(t, list, []string{
		"id", "content", "user_id", "memory_type", "metadata", "created_at", "updated_at", "last_accessed", "access_count",
	})
}

func assertSearchReturnFields(t *testing.T, cmd []string, want []string) {
	t.Helper()
	for i := 0; i < len(cmd)-1; i++ {
		if cmd[i] != "RETURN" {
			continue
		}
		require.Equal(t, strconv.Itoa(len(want)), cmd[i+1])
		require.Equal(t, want, cmd[i+2:i+2+len(want)])
		return
	}
	t.Fatal("FT.SEARCH command has no RETURN clause")
}

func TestValkeyConditionalMutationScripts(t *testing.T) {
	t.Parallel()

	assert.Contains(t, valkeyUpdateIfCurrentScriptSource, "redis.call('HGET', KEYS[1], 'id') ~= ARGV[1]")
	assert.Contains(t, valkeyUpdateIfCurrentScriptSource, "redis.call('HSET', KEYS[1], unpack(ARGV, 2))")
	assert.Contains(t, valkeyTrackRetrievalScriptSource, "redis.call('HGET', KEYS[1], 'id') ~= ARGV[1]")
	assert.Contains(t, valkeyTrackRetrievalScriptSource, "redis.call('HINCRBY', KEYS[1], 'access_count', 1)")
	assert.Contains(t, valkeyTrackRetrievalScriptSource, "redis.call('HSET', KEYS[1], 'last_accessed', ARGV[2])")
	assert.Equal(t, []string{"content", "summary", "id", "memory-id"}, valkeyHashFieldArgs(map[string]string{
		"id":      "memory-id",
		"content": "summary",
	}))
}

// ---------------------------------------------------------------------------
// TLS configuration (config struct tests — no live Valkey required)
// ---------------------------------------------------------------------------

func TestMemoryValkeyConfig_TLSFields(t *testing.T) {
	t.Parallel()

	t.Run("defaults are zero-values", func(t *testing.T) {
		t.Parallel()
		cfg := config.MemoryValkeyConfig{}
		assert.False(t, cfg.TLSEnabled)
		assert.Empty(t, cfg.TLSCAPath)
		assert.False(t, cfg.TLSInsecureSkipVerify)
	})

	t.Run("all fields populated", func(t *testing.T) {
		t.Parallel()
		cfg := config.MemoryValkeyConfig{
			TLSEnabled:            true,
			TLSCAPath:             "/etc/certs/ca.pem",
			TLSInsecureSkipVerify: false,
		}
		assert.True(t, cfg.TLSEnabled)
		assert.Equal(t, "/etc/certs/ca.pem", cfg.TLSCAPath)
		assert.False(t, cfg.TLSInsecureSkipVerify)
	})

	t.Run("insecure skip verify", func(t *testing.T) {
		t.Parallel()
		cfg := config.MemoryValkeyConfig{
			TLSEnabled:            true,
			TLSInsecureSkipVerify: true,
		}
		assert.True(t, cfg.TLSEnabled)
		assert.True(t, cfg.TLSInsecureSkipVerify)
	})
}

func TestNewValkeyStore_DisabledWithTLS(t *testing.T) {
	t.Parallel()
	// TLS fields should not matter when the store is disabled.
	store, err := NewValkeyStore(ValkeyStoreOptions{
		Enabled: false,
		ValkeyConfig: &config.MemoryValkeyConfig{
			TLSEnabled: true,
			TLSCAPath:  "/nonexistent/ca.pem",
		},
	})
	require.NoError(t, err)
	assert.False(t, store.IsEnabled())
}

func TestNewValkeyStore_RequiresClient(t *testing.T) {
	t.Parallel()
	// Enabled store with TLS but no client should fail with clear error.
	_, err := NewValkeyStore(ValkeyStoreOptions{
		Enabled: true,
		ValkeyConfig: &config.MemoryValkeyConfig{
			TLSEnabled: true,
		},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "valkey client is required")
}

// ---------------------------------------------------------------------------
// valkeyFloat32ToBytes / valkeyBytesToFloat32
// ---------------------------------------------------------------------------

func TestValkeyFloat32ToBytes_Roundtrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		input []float32
	}{
		{"basic values", []float32{1.0, 2.0, 3.0}},
		{"negative and special", []float32{0.0, -1.5, 3.14, math.MaxFloat32, math.SmallestNonzeroFloat32}},
		{"empty", []float32{}},
		{"single value", []float32{42.0}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			b := valkeyFloat32ToBytes(tc.input)
			assert.Len(t, b, len(tc.input)*4)

			// Verify little-endian encoding
			for i, expected := range tc.input {
				bits := binary.LittleEndian.Uint32(b[i*4:])
				assert.Equal(t, expected, math.Float32frombits(bits))
			}

			// Roundtrip
			result := valkeyBytesToFloat32(b)
			assert.Equal(t, tc.input, result)
		})
	}
}

func TestValkeyBytesToFloat32_InvalidLength(t *testing.T) {
	t.Parallel()
	// Not a multiple of 4
	assert.Nil(t, valkeyBytesToFloat32([]byte{1, 2, 3}))
	assert.Nil(t, valkeyBytesToFloat32([]byte{1}))
}

// ---------------------------------------------------------------------------
// valkeyEscapeTagValue
// ---------------------------------------------------------------------------

func TestValkeyEscapeTagValue(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"hyphens", "file-123", "file\\-123"},
		{"dots", "doc.txt", "doc\\.txt"},
		{"colons", "ns:val", "ns\\:val"},
		{"slashes", "path/to", "path\\/to"},
		{"spaces", "hello world", "hello\\ world"},
		{"multiple specials", "a-b.c:d/e f", "a\\-b\\.c\\:d\\/e\\ f"},
		{"safe string", "abc123", "abc123"},
		{"empty", "", ""},
		{"braces", "a{b}c", "a\\{b\\}c"},
		{"brackets", "a[b]c", "a\\[b\\]c"},
		{"pipe", "a|b", "a\\|b"},
		{"at sign", "user@host", "user\\@host"},
		{"parens", "f(x)", "f\\(x\\)"},
		{"asterisk", "a*b", "a\\*b"},
		{"exclamation", "no!", "no\\!"},
		{"tilde", "~user", "\\~user"},
		{"caret", "a^b", "a\\^b"},
		{"quotes", `a"b'c`, `a\"b\'c`},
		{"hash", "a#b", "a\\#b"},
		{"dollar", "a$b", "a\\$b"},
		{"percent", "a%b", "a\\%b"},
		{"ampersand", "a&b", "a\\&b"},
		{"plus equals", "a+=b", "a\\+\\=b"},
		{"backslash", "a\\b", "a\\\\b"},
		{"semicolon", "a;b", "a\\;b"},
		{"comma", "a,b", "a\\,b"},
		{"angle brackets", "a<b>c", "a\\<b\\>c"},
		{"tab", "a\tb", "a\\\tb"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, valkeyEscapeTagValue(tc.input))
		})
	}
}

// ---------------------------------------------------------------------------
// valkeyToInt64
// ---------------------------------------------------------------------------

func TestValkeyToInt64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    interface{}
		expected int64
	}{
		{"int64", int64(42), 42},
		{"float64", float64(42.9), 42},
		{"string", "123", 123},
		{"nil", nil, 0},
		{"bool", true, 0},
		{"invalid string", "abc", 0},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, valkeyToInt64(tc.input))
		})
	}
}

// ---------------------------------------------------------------------------
// valkeyParseScoreFromMap
// ---------------------------------------------------------------------------

func TestValkeyParseScoreFromMap(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		fields     map[string]interface{}
		key        string
		metricType string
		expected   float64
		tolerance  float64
	}{
		{
			"valid COSINE distance",
			map[string]interface{}{"vector_distance": "0.2"},
			"vector_distance", "COSINE", 0.9, 0.01,
		},
		{
			"missing key",
			map[string]interface{}{"other": "0.2"},
			"vector_distance", "COSINE", 0.0, 0.001,
		},
		{
			"invalid number",
			map[string]interface{}{"vector_distance": "abc"},
			"vector_distance", "COSINE", 0.0, 0.001,
		},
		{
			"float64 value",
			map[string]interface{}{"vector_distance": float64(0.4)},
			"vector_distance", "COSINE", 0.8, 0.01,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			result := valkeyParseScoreFromMap(tc.fields, tc.key, tc.metricType)
			assert.InDelta(t, tc.expected, result, tc.tolerance)
		})
	}
}

// ---------------------------------------------------------------------------
// valkeyFieldsToMemory
// ---------------------------------------------------------------------------

func TestValkeyFieldsToMemory(t *testing.T) {
	t.Parallel()

	t.Run("full fields", func(t *testing.T) {
		t.Parallel()
		embedding := []float32{0.1, 0.2, 0.3}
		fields := map[string]string{
			"id":            "mem_123",
			"content":       "test content",
			"user_id":       "user1",
			"memory_type":   "semantic",
			"metadata":      `{"project_id":"proj1","source":"conversation","importance":0.8,"access_count":3,"last_accessed":1700000000}`,
			"created_at":    "1700000000",
			"updated_at":    "1700000100",
			"last_accessed": "1700000200123",
			"embedding":     string(valkeyFloat32ToBytes(embedding)),
			"access_count":  "3",
			"importance":    "0.8",
		}

		mem := valkeyFieldsToMemory(fields)
		assert.Equal(t, "mem_123", mem.ID)
		assert.Equal(t, "test content", mem.Content)
		assert.Equal(t, "user1", mem.UserID)
		assert.Equal(t, MemoryType("semantic"), mem.Type)
		assert.Equal(t, "proj1", mem.ProjectID)
		assert.Equal(t, "conversation", mem.Source)
		assert.InDelta(t, float32(0.8), mem.Importance, 0.01)
		assert.Equal(t, 3, mem.AccessCount)
		assert.False(t, mem.CreatedAt.IsZero())
		assert.False(t, mem.UpdatedAt.IsZero())
		assert.Equal(t, time.UnixMilli(1700000200123), mem.LastAccessed)
		require.Len(t, mem.Embedding, 3)
		assert.InDelta(t, float32(0.1), mem.Embedding[0], 0.001)
	})

	t.Run("empty fields", func(t *testing.T) {
		t.Parallel()
		mem := valkeyFieldsToMemory(map[string]string{})
		assert.Empty(t, mem.ID)
		assert.Nil(t, mem.Embedding)
	})

	t.Run("invalid metadata JSON", func(t *testing.T) {
		t.Parallel()
		fields := map[string]string{
			"id":       "mem_456",
			"metadata": "not valid json",
		}
		mem := valkeyFieldsToMemory(fields)
		assert.Equal(t, "mem_456", mem.ID)
		// Should not panic; metadata fields remain zero-value
		assert.Empty(t, mem.ProjectID)
	})

	t.Run("invalid timestamps", func(t *testing.T) {
		t.Parallel()
		fields := map[string]string{
			"id":         "mem_789",
			"created_at": "not_a_number",
			"updated_at": "",
		}
		mem := valkeyFieldsToMemory(fields)
		assert.True(t, mem.CreatedAt.IsZero())
		assert.True(t, mem.UpdatedAt.IsZero())
	})
}

// ---------------------------------------------------------------------------
// valkeyFieldsMapToMemory (FT.SEARCH result format)
// ---------------------------------------------------------------------------

func TestValkeyFieldsMapToMemory(t *testing.T) {
	t.Parallel()

	t.Run("full fields", func(t *testing.T) {
		t.Parallel()
		fields := map[string]interface{}{
			"id":            "mem_100",
			"content":       "search result content",
			"user_id":       "user2",
			"memory_type":   "procedural",
			"metadata":      `{"project_id":"proj2","source":"extraction","importance":0.5}`,
			"created_at":    "1700000000",
			"updated_at":    "1700000200",
			"last_accessed": "1700000300123",
			"access_count":  "4",
		}

		mem := valkeyFieldsMapToMemory(fields)
		assert.Equal(t, "mem_100", mem.ID)
		assert.Equal(t, "search result content", mem.Content)
		assert.Equal(t, "user2", mem.UserID)
		assert.Equal(t, MemoryType("procedural"), mem.Type)
		assert.Equal(t, "proj2", mem.ProjectID)
		assert.Equal(t, "extraction", mem.Source)
		assert.InDelta(t, float32(0.5), mem.Importance, 0.01)
		assert.Equal(t, 4, mem.AccessCount)
		assert.Equal(t, time.UnixMilli(1700000300123), mem.LastAccessed)
	})

	t.Run("empty map", func(t *testing.T) {
		t.Parallel()
		mem := valkeyFieldsMapToMemory(map[string]interface{}{})
		assert.Empty(t, mem.ID)
	})

	t.Run("numeric search fields", func(t *testing.T) {
		t.Parallel()
		fields := map[string]interface{}{
			"id":            "mem_101",
			"content":       "search result content",
			"created_at":    float64(1_700_000_000_000),
			"updated_at":    int64(1_700_000_020_000),
			"last_accessed": float64(1_700_000_030_123),
			"access_count":  float64(4),
		}
		mem := valkeyFieldsMapToMemory(fields)
		assert.Equal(t, time.UnixMilli(1_700_000_000_000), mem.CreatedAt)
		assert.Equal(t, time.UnixMilli(1_700_000_020_000), mem.UpdatedAt)
		assert.Equal(t, time.UnixMilli(1_700_000_030_123), mem.LastAccessed)
		assert.Equal(t, 4, mem.AccessCount)
	})
}

// ---------------------------------------------------------------------------
// parseSearchCandidates
// ---------------------------------------------------------------------------

func TestValkeyStore_ParseSearchCandidates(t *testing.T) {
	t.Parallel()

	store := &ValkeyStore{metricType: "COSINE"}

	t.Run("nil input", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, store.parseSearchCandidates(nil, "user1"))
	})

	t.Run("non-array input", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, store.parseSearchCandidates("not an array", "user1"))
	})

	t.Run("zero total count", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, store.parseSearchCandidates([]interface{}{int64(0)}, "user1"))
	})

	t.Run("valid single result", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(1), map[string]interface{}{
			"mem:1": map[string]interface{}{
				"id": "mem_1", "content": "hello world", "memory_type": "semantic",
				"metadata":        `{"user_id":"user1","project_id":"proj1","source":"conversation"}`,
				"vector_distance": "0.2",
			},
		}}

		candidates := store.parseSearchCandidates(result, "user1")
		require.Len(t, candidates, 1)
		assert.Equal(t, "mem_1", candidates[0].Memory.ID)
		assert.Equal(t, "hello world", candidates[0].Memory.Content)
		assert.Equal(t, "user1", candidates[0].Memory.UserID)
		assert.Equal(t, "proj1", candidates[0].Memory.ProjectID)
		assert.InDelta(t, 0.9, float64(candidates[0].Score), 0.01)
	})

	t.Run("projects access fields from search", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(1), map[string]interface{}{
			"mem:1": map[string]interface{}{
				"id": "mem_1", "content": "hello world", "memory_type": "semantic",
				"metadata":        `{"user_id":"user1"}`,
				"vector_distance": "0.2",
				"last_accessed":   float64(1_700_000_200_123),
				"access_count":    float64(4),
			},
		}}

		candidates := store.parseSearchCandidates(result, "user1")
		require.Len(t, candidates, 1)
		assert.Equal(t, time.UnixMilli(1_700_000_200_123), candidates[0].Memory.LastAccessed)
		assert.Equal(t, 4, candidates[0].Memory.AccessCount)
	})

	t.Run("multiple results sorted by score descending", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(2), map[string]interface{}{
			"mem:1": map[string]interface{}{
				"id": "mem_1", "content": "good", "memory_type": "semantic",
				"metadata": `{"user_id":"u1"}`, "vector_distance": "0.4",
			},
			"mem:2": map[string]interface{}{
				"id": "mem_2", "content": "better", "memory_type": "semantic",
				"metadata": `{"user_id":"u1"}`, "vector_distance": "0.1",
			},
		}}

		candidates := store.parseSearchCandidates(result, "u1")
		require.Len(t, candidates, 2)
		assert.GreaterOrEqual(t, candidates[0].Score, candidates[1].Score, "results should be sorted descending by score")
		assert.Equal(t, "mem_2", candidates[0].Memory.ID) // lower distance = higher similarity
	})

	t.Run("skips entries with missing id", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(1), map[string]interface{}{
			"mem:1": map[string]interface{}{
				"content": "no id", "memory_type": "semantic",
				"metadata": `{}`, "vector_distance": "0.1",
			},
		}}

		candidates := store.parseSearchCandidates(result, "u1")
		assert.Empty(t, candidates)
	})

	t.Run("default user ID from parameter", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(1), map[string]interface{}{
			"mem:1": map[string]interface{}{
				"id": "mem_1", "content": "test", "memory_type": "semantic",
				"metadata": `{}`, "vector_distance": "0.2",
			},
		}}

		candidates := store.parseSearchCandidates(result, "default_user")
		require.Len(t, candidates, 1)
		assert.Equal(t, "default_user", candidates[0].Memory.UserID)
	})
}

// ---------------------------------------------------------------------------
// parseListSearchResults
// ---------------------------------------------------------------------------

func TestValkeyStore_ParseListSearchResults(t *testing.T) {
	t.Parallel()

	store := &ValkeyStore{metricType: "COSINE"}

	t.Run("nil input", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, store.parseListSearchResults(nil))
	})

	t.Run("zero total count", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, store.parseListSearchResults([]interface{}{int64(0)}))
	})

	t.Run("valid results", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(2), map[string]interface{}{
			"mem:1": map[string]interface{}{
				"id": "mem_1", "content": "first", "user_id": "u1",
				"memory_type": "semantic", "metadata": `{"project_id":"p1"}`,
				"created_at": "1700000000", "updated_at": "1700000100",
			},
			"mem:2": map[string]interface{}{
				"id": "mem_2", "content": "second", "user_id": "u1",
				"memory_type": "procedural", "metadata": `{"project_id":"p2"}`,
				"created_at": "1700000200", "updated_at": "1700000300",
			},
		}}

		memories := store.parseListSearchResults(result)
		require.Len(t, memories, 2)
		// Newer created_at first (client-side sort; map iteration order is undefined).
		assert.Equal(t, "mem_2", memories[0].ID)
		assert.Equal(t, "mem_1", memories[1].ID)
	})
}

// ---------------------------------------------------------------------------
// extractIDsFromSearchResult
// ---------------------------------------------------------------------------

func TestValkeyStore_ExtractIDsFromSearchResult(t *testing.T) {
	t.Parallel()

	store := &ValkeyStore{}

	t.Run("no project filter", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(2), map[string]interface{}{
			"mem:1": map[string]interface{}{"id": "mem_1", "metadata": `{"project_id":"proj1"}`},
			"mem:2": map[string]interface{}{"id": "mem_2", "metadata": `{"project_id":"proj2"}`},
		}}

		ids := store.extractIDsFromSearchResult(result, "")
		assert.Len(t, ids, 2)
	})

	t.Run("with project filter", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(2), map[string]interface{}{
			"mem:1": map[string]interface{}{"id": "mem_1", "metadata": `{"project_id":"proj1"}`},
			"mem:2": map[string]interface{}{"id": "mem_2", "metadata": `{"project_id":"proj2"}`},
		}}

		ids := store.extractIDsFromSearchResult(result, "proj1")
		assert.Len(t, ids, 1)
		assert.Equal(t, "mem_1", ids[0])
	})

	t.Run("nil input", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, store.extractIDsFromSearchResult(nil, ""))
	})

	t.Run("zero results", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, store.extractIDsFromSearchResult([]interface{}{int64(0)}, ""))
	})
}

// ---------------------------------------------------------------------------
// hashKey
// ---------------------------------------------------------------------------

func TestValkeyStore_HashKey(t *testing.T) {
	t.Parallel()

	store := &ValkeyStore{collectionPrefix: "mem:"}
	assert.Equal(t, "mem:abc123", store.hashKey("abc123"))

	store2 := &ValkeyStore{collectionPrefix: "custom_prefix:"}
	assert.Equal(t, "custom_prefix:xyz", store2.hashKey("xyz"))
}

// ---------------------------------------------------------------------------
// NewValkeyStore validation
// ---------------------------------------------------------------------------

func TestNewValkeyStore_Disabled(t *testing.T) {
	t.Parallel()
	store, err := NewValkeyStore(ValkeyStoreOptions{Enabled: false})
	require.NoError(t, err)
	assert.False(t, store.IsEnabled())
}

func TestNewValkeyStore_NilClient(t *testing.T) {
	t.Parallel()
	_, err := NewValkeyStore(ValkeyStoreOptions{
		Enabled:      true,
		ValkeyConfig: &config.MemoryValkeyConfig{},
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "valkey client is required")
}

func TestNewValkeyStore_NilConfig(t *testing.T) {
	t.Parallel()
	_, err := NewValkeyStore(ValkeyStoreOptions{
		Enabled: true,
		// Client would be non-nil in a real test, but we'll hit the config check first
		// since we validate config before using client
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "valkey client is required")
}

// ---------------------------------------------------------------------------
// Config defaults
// ---------------------------------------------------------------------------

func TestValkeyStore_ConfigDefaults(t *testing.T) {
	t.Parallel()

	// Verify default values are applied when config fields are zero-valued
	store := &ValkeyStore{}

	// These are tested implicitly through NewValkeyStore, but we also verify
	// the struct fields directly
	assert.Empty(t, store.indexName)
	assert.Empty(t, store.collectionPrefix)
	assert.Empty(t, store.metricType)
	assert.Equal(t, 0, store.dimension)
}

// ---------------------------------------------------------------------------
// extractHashKeysFromSearchResult
// ---------------------------------------------------------------------------

func TestValkeyStore_ExtractHashKeysFromSearchResult(t *testing.T) {
	t.Parallel()

	store := &ValkeyStore{}

	t.Run("nil input", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, store.extractHashKeysFromSearchResult(nil))
	})

	t.Run("empty array", func(t *testing.T) {
		t.Parallel()
		assert.Nil(t, store.extractHashKeysFromSearchResult([]interface{}{int64(0)}))
	})

	t.Run("map format", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(2), map[string]interface{}{
			"mem:key1": map[string]interface{}{"id": "mem_1"},
			"mem:key2": map[string]interface{}{"id": "mem_2"},
		}}

		keys := store.extractHashKeysFromSearchResult(result)
		assert.Len(t, keys, 2)
		assert.Contains(t, keys, "mem:key1")
		assert.Contains(t, keys, "mem:key2")
	})

	t.Run("string format", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(1), "mem:key1"}

		keys := store.extractHashKeysFromSearchResult(result)
		assert.Len(t, keys, 1)
		assert.Equal(t, "mem:key1", keys[0])
	})
}

// ---------------------------------------------------------------------------
// extractTotalCount
// ---------------------------------------------------------------------------

func TestValkeyStore_ExtractTotalCount(t *testing.T) {
	t.Parallel()

	store := &ValkeyStore{}

	t.Run("nil input", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, 0, store.extractTotalCount(nil))
	})

	t.Run("valid count", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(42), map[string]interface{}{}}
		assert.Equal(t, 42, store.extractTotalCount(result))
	})

	t.Run("zero count", func(t *testing.T) {
		t.Parallel()
		result := []interface{}{int64(0)}
		assert.Equal(t, 0, store.extractTotalCount(result))
	})

	t.Run("non-array input", func(t *testing.T) {
		t.Parallel()
		assert.Equal(t, 0, store.extractTotalCount("not an array"))
	})
}

// ---------------------------------------------------------------------------
// Access metadata is stored in top-level fields, not metadata JSON.
// ---------------------------------------------------------------------------

// TestValkeyBuildHashFields_AccessMetadataNotInMetadata verifies that
// valkeyBuildHashFields stores access metadata in top-level HASH fields only.
// This prevents a content update from overwriting newer access metadata through
// a stale JSON blob.
func TestValkeyBuildHashFields_AccessMetadataNotInMetadata(t *testing.T) {
	t.Parallel()

	mem := &Memory{
		ID:           "mem_race_test",
		Content:      "test content",
		UserID:       "u1",
		AccessCount:  5,
		LastAccessed: time.UnixMilli(1700000200123),
		Importance:   0.8,
	}
	embedding := []float32{0.1, 0.2, 0.3}

	fields, err := valkeyBuildHashFields(mem, embedding)
	require.NoError(t, err)

	// access_count must be a top-level HASH field
	assert.Equal(t, "5", fields["access_count"], "access_count should be a top-level HASH field")
	assert.Equal(t, "1700000200123", fields["last_accessed"], "last_accessed should be a top-level HASH field")

	// Access metadata must NOT be in metadata JSON (to avoid concurrent write races).
	metadataStr, ok := fields["metadata"]
	require.True(t, ok, "metadata field must exist")
	var metadata map[string]interface{}
	require.NoError(t, json.Unmarshal([]byte(metadataStr), &metadata))
	_, hasAccessCount := metadata["access_count"]
	assert.False(t, hasAccessCount, "access_count must NOT be stored in metadata JSON to prevent concurrent write races")
	_, hasLastAccessed := metadata["last_accessed"]
	assert.False(t, hasLastAccessed, "last_accessed must NOT be stored in metadata JSON to prevent concurrent write races")
}

func TestValkeyBuildUpdateHashFieldsPreservesAccessMetadata(t *testing.T) {
	mem := &Memory{
		ID:           "mem_update_test",
		Content:      "updated content",
		UserID:       "u1",
		Embedding:    []float32{0.1, 0.2, 0.3},
		AccessCount:  12,
		LastAccessed: time.UnixMilli(1700000200123),
	}

	fields, err := valkeyBuildUpdateHashFields(mem)
	require.NoError(t, err)
	assert.NotContains(t, fields, "access_count")
	assert.NotContains(t, fields, "last_accessed")
	assert.Equal(t, "updated content", fields["content"])
}

// ---------------------------------------------------------------------------
// errValkeyMemoryAlreadyExists sentinel error
// ---------------------------------------------------------------------------

// TestErrValkeyMemoryAlreadyExists verifies that the sentinel error works with
// errors.Is and fmt.Errorf wrapping, so callers can reliably detect duplicates.
func TestErrValkeyMemoryAlreadyExists(t *testing.T) {
	t.Parallel()

	// Wrapped with fmt.Errorf %w
	wrapped := fmt.Errorf("%w: memory id=test123", errValkeyMemoryAlreadyExists)
	assert.ErrorIs(t, wrapped, errValkeyMemoryAlreadyExists)
	assert.Contains(t, wrapped.Error(), "valkey memory already exists")
	assert.Contains(t, wrapped.Error(), "test123")

	// Double-wrapped
	doubleWrapped := fmt.Errorf("valkey store failed: %w", wrapped)
	assert.ErrorIs(t, doubleWrapped, errValkeyMemoryAlreadyExists)
}

func TestValkeySourceVersionArgsUseStoredVersion(t *testing.T) {
	t.Parallel()

	args := valkeySourceVersionArgs(memoryVersion{
		id:         "id-1",
		userID:     "user-1",
		projectID:  "",
		typ:        MemoryTypeSemantic,
		content:    `say "hello"`,
		createdAt:  time.UnixMilli(1_600_000_000_123),
		updatedAt:  time.UnixMilli(1_700_000_000_123),
		importance: 0.7,
	})
	require.Equal(t, []string{
		"id-1",
		"user-1",
		"default",
		"semantic",
		`say "hello"`,
		"1600000000123",
		"1700000000123",
		"0.7",
	}, args)
	require.Contains(t, valkeyReplaceCurrentGroupScriptSource, "redis.call('DEL', KEYS[i])")
	require.Contains(t, valkeyReplaceCurrentGroupScriptSource, "fields[j] ~= ARGV[arg + j - 1]")
}
