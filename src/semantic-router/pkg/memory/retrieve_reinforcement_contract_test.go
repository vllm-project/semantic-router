//go:build !windows

package memory

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"
	glide "github.com/valkey-io/valkey-glide/go/v2"
	glideconfig "github.com/valkey-io/valkey-glide/go/v2/config"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	milvuslifecycle "github.com/vllm-project/semantic-router/src/semantic-router/pkg/milvus"
)

// Retrieve-time reinforcement is the retention contract (S = S0 + AccessCount,
// t = LastAccessed) that every maintained persistent backend must honor. Each
// contract test below runs against a real service — the same setup as the
// storage integration lane — and fails when that backend's Retrieve does not
// reinforce, instead of the behavior diverging silently per backend.

const (
	retrieveReinforcementContent = "The user prefers dark mode in all their applications"
	retrieveReinforcementQuery   = "What are the user's display preferences?"
)

// StorageIntegration: qdrant
func TestQdrantRetrieveReinforcementContract(t *testing.T) {
	storagetest.Require(t, "qdrant")
	store := setupQdrantReinforcementStore(t)
	assertRetrieveReinforces(t, store, true)
}

// StorageIntegration: valkey
func TestValkeyRetrieveReinforcementContract(t *testing.T) {
	storagetest.Require(t, "valkey")
	store := setupValkeyReinforcementStore(t)
	// Valkey reinforcement (by design, "S += 1, t = 0") updates the
	// authoritative access_count/updated_at HASH fields and leaves the
	// metadata-JSON copy of last_accessed frozen.
	assertRetrieveReinforces(t, store, false)
}

// StorageIntegration: milvus
func TestMilvusRetrieveReinforcementContract(t *testing.T) {
	storagetest.Require(t, "milvus")
	store := setupMilvusReinforcementStore(t)
	assertRetrieveReinforces(t, store, true)
}

func uniqueReinforcementSuffix() string {
	return strconv.FormatInt(time.Now().UnixNano(), 36)
}

func reinforcementEmbeddingConfig() *EmbeddingConfig {
	return &EmbeddingConfig{
		Provider:  storageMemoryVectors(),
		Model:     EmbeddingModelQwen3,
		Dimension: 384,
	}
}

func setupQdrantReinforcementStore(t *testing.T) Store {
	t.Helper()
	host := os.Getenv("QDRANT_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6334
	if configured := os.Getenv("QDRANT_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		require.NoError(t, err)
		port = parsed
	}

	client, err := qdrant.NewClient(&qdrant.Config{Host: host, Port: port})
	if err != nil {
		storagetest.Unavailable(t, "qdrant", err)
		return nil
	}
	collection := fmt.Sprintf("memory_reinforce_%s", uniqueReinforcementSuffix())
	store, err := NewQdrantStore(QdrantStoreOptions{
		Client: client,
		Config: config.MemoryConfig{},
		QdrantConfig: &config.MemoryQdrantConfig{
			Host: host, Port: port, Collection: collection, Dimension: 384,
		},
		Enabled:         true,
		EmbeddingConfig: reinforcementEmbeddingConfig(),
	})
	if err != nil {
		_ = client.Close()
		storagetest.Unavailable(t, "qdrant", err)
		return nil
	}
	t.Cleanup(func() {
		_ = client.DeleteCollection(context.Background(), collection)
		_ = store.Close()
	})
	return store
}

func setupValkeyReinforcementStore(t *testing.T) Store {
	t.Helper()
	host := os.Getenv("VALKEY_HOST")
	if host == "" {
		host = "localhost"
	}
	port := 6379
	if configured := os.Getenv("VALKEY_PORT"); configured != "" {
		parsed, err := strconv.Atoi(configured)
		require.NoError(t, err)
		port = parsed
	}

	suffix := uniqueReinforcementSuffix()
	client, err := glide.NewClient(glideconfig.NewClientConfiguration().
		WithAddress(&glideconfig.NodeAddress{Host: host, Port: port}).
		WithRequestTimeout(5 * time.Second))
	if err != nil {
		storagetest.Unavailable(t, "valkey", err)
		return nil
	}
	t.Cleanup(client.Close)
	store, err := NewValkeyStore(ValkeyStoreOptions{
		Client: client,
		ValkeyConfig: &config.MemoryValkeyConfig{
			Host:                host,
			Port:                port,
			Database:            0,
			Timeout:             5,
			CollectionPrefix:    fmt.Sprintf("test_reinforce_%s:", suffix),
			IndexName:           fmt.Sprintf("test_reinforce_idx_%s", suffix),
			Dimension:           384,
			MetricType:          "COSINE",
			IndexM:              16,
			IndexEfConstruction: 256,
		},
		Enabled:         true,
		EmbeddingConfig: reinforcementEmbeddingConfig(),
	})
	if err != nil {
		client.Close()
		storagetest.Unavailable(t, "valkey", err)
		return nil
	}
	return store
}

func setupMilvusReinforcementStore(t *testing.T) Store {
	t.Helper()
	address := os.Getenv("MILVUS_URI")
	if address == "" {
		address = "localhost:19530"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	client, err := milvuslifecycle.ConnectGRPC(ctx, address, 0)
	if err != nil {
		storagetest.Unavailable(t, "milvus", err)
		return nil
	}
	cfg := config.MemoryConfig{Milvus: config.MemoryMilvusConfig{Dimension: 384}}
	collection := fmt.Sprintf("memory_reinforce_%s", uniqueReinforcementSuffix())
	store, err := NewMilvusStore(MilvusStoreOptions{
		Client:          client,
		CollectionName:  collection,
		Config:          cfg,
		Enabled:         true,
		EmbeddingConfig: reinforcementEmbeddingConfig(),
	})
	if err != nil {
		_ = client.Close()
		storagetest.Unavailable(t, "milvus", err)
		return nil
	}
	t.Cleanup(func() {
		_ = client.DropCollection(context.Background(), collection)
		_ = client.Close()
	})
	return store
}

// assertRetrieveReinforces stores a memory, retrieves it, and polls the store
// until the background reinforcement is visible through the read path.
// lastAccessedAdvances is false for Valkey, whose reinforcement leaves the
// metadata-JSON copy of last_accessed frozen (see the contract test above).
func assertRetrieveReinforces(t *testing.T, store Store, lastAccessedAdvances bool) {
	t.Helper()
	ctx := context.Background()

	mem := &Memory{
		ID:      fmt.Sprintf("reinforce-%s", uniqueReinforcementSuffix()),
		Content: retrieveReinforcementContent,
		UserID:  "user-a",
		Type:    MemoryTypeSemantic,
	}
	require.NoError(t, store.Store(ctx, mem))

	// The write can take a moment to enter the query view (Milvus growing
	// segments), so the baseline read waits for visibility.
	baseline := getMemoryEventually(t, store, mem.ID, 15*time.Second)
	require.Equal(t, 0, baseline.AccessCount, "a fresh memory starts unreinforced")

	if lastAccessedAdvances {
		// last_accessed has second granularity; make sure the retrieval lands
		// in a later second than the store so the advance is observable.
		time.Sleep(1100 * time.Millisecond)
	}

	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query:  retrieveReinforcementQuery,
		UserID: "user-a",
		Limit:  5,
	})
	require.NoError(t, err)
	require.Len(t, results, 1, "the fixture query must retrieve the stored memory")
	require.Equal(t, mem.ID, results[0].Memory.ID)

	// Reinforcement is asynchronous by design; poll the read path for it.
	// Get can transiently report the memory missing while a backend's
	// reinforcement upsert (delete+insert) is in flight, so errors inside the
	// window are retried rather than fatal.
	deadline := time.Now().Add(15 * time.Second)
	var reinforced *Memory
	for time.Now().Before(deadline) {
		current, err := store.Get(ctx, mem.ID)
		if err == nil &&
			current.AccessCount > baseline.AccessCount &&
			(!lastAccessedAdvances || current.LastAccessed.After(baseline.LastAccessed)) {
			reinforced = current
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if reinforced == nil {
		current, getErr := store.Get(ctx, mem.ID)
		t.Fatalf("retrieval never reinforced the memory: access_count=%d last_accessed=%v (final get: %v)",
			current.AccessCount, current.LastAccessed, getErr)
	}
	require.Equal(t, baseline.AccessCount+1, reinforced.AccessCount)
	if lastAccessedAdvances {
		require.True(t, reinforced.LastAccessed.After(baseline.LastAccessed),
			"last_accessed must advance with the reinforcement")
	}
}

// getMemoryEventually waits until the memory is visible on the read path.
func getMemoryEventually(t *testing.T, store Store, id string, within time.Duration) *Memory {
	t.Helper()
	deadline := time.Now().Add(within)
	for {
		mem, err := store.Get(context.Background(), id)
		if err == nil {
			return mem
		}
		if time.Now().After(deadline) {
			t.Fatalf("memory %s never became readable: %v", id, err)
		}
		time.Sleep(50 * time.Millisecond)
	}
}
