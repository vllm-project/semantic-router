package extproc

import (
	"context"
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
)

// Run separate test processes against the same disposable Qdrant/Redis
// services: write with one model, verify a second model starts empty, write
// with the second, then reopen each model. No historical data is deleted.
func TestMemoryEmbeddingIdentityPersistent(t *testing.T) {
	path := os.Getenv("VSR_TEST_MEMORY_IDENTITY_CONFIG")
	if path == "" {
		t.Skip("set VSR_TEST_MEMORY_IDENTITY_CONFIG for real model and persistent store regression")
	}
	var fixture struct {
		ModelPath      string   `json:"model_path"`
		UseCPU         bool     `json:"use_cpu"`
		Dimension      int      `json:"dimension"`
		QdrantHost     string   `json:"qdrant_host"`
		QdrantPort     int      `json:"qdrant_port"`
		RedisAddress   string   `json:"redis_address"`
		Collection     string   `json:"collection"`
		UserID         string   `json:"user_id"`
		Before         []string `json:"before"`
		Write          string   `json:"write"`
		After          []string `json:"after"`
		RequireRuntime string   `json:"require_runtime"`
		ReceiptPath    string   `json:"receipt_path"`
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.ModelPath == "" || fixture.UserID == "" || fixture.Collection == "" || fixture.RedisAddress == "" {
		t.Fatal("incomplete model/storage fixture")
	}
	t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", "")
	cfg := &config.RouterConfig{
		Memory: config.MemoryConfig{
			Enabled: true, Backend: "qdrant", EmbeddingModel: "mmbert",
			Qdrant:     &config.MemoryQdrantConfig{Host: fixture.QdrantHost, Port: fixture.QdrantPort, Collection: fixture.Collection, Dimension: fixture.Dimension},
			RedisCache: &config.MemoryRedisCacheConfig{Enabled: true, Address: fixture.RedisAddress, KeyPrefix: fixture.Collection + ":hot:", TTLSeconds: 3600},
		},
	}
	cfg.EmbeddingModels = config.EmbeddingModels{MmBertModelPath: fixture.ModelPath, UseCPU: fixture.UseCPU}
	prepared, err := modelruntime.PrepareOwnedEmbeddings(context.Background(), cfg, native.New(nil))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = prepared.Close() })
	store, err := createMemoryStore(cfg, prepared)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	identity, err := prepared.ResolveIdentity(embedding.ConsumerSettings{ModelType: "mmbert", Dimension: fixture.Dimension, InputPolicy: "memory-content-v1"})
	if err != nil {
		t.Fatal(err)
	}
	if fixture.RequireRuntime != "" && !strings.Contains(identity.Descriptor.Runtime, fixture.RequireRuntime) {
		t.Fatalf("unexpected execution runtime: %s", identity.Descriptor.Runtime)
	}
	if fixture.ReceiptPath != "" {
		data, err := json.MarshalIndent(identity, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(fixture.ReceiptPath, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if _, ok := store.(*memory.CachingStore); !ok {
		t.Fatal("real Redis hot cache was not connected")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	query := "Kyoto is my favorite city."
	identifier := func(label string) string {
		return uuid.NewSHA1(uuid.NameSpaceOID, []byte(fixture.UserID+":"+label)).String()
	}
	check := func(labels []string) {
		t.Helper()
		want := make([]string, 0, len(labels))
		for _, label := range labels {
			want = append(want, identifier(label))
		}
		slices.Sort(want)
		// The second lookup must also be correct through Redis's cached result.
		for range 2 {
			results, err := store.Retrieve(ctx, memory.RetrieveOptions{Query: query, UserID: fixture.UserID, Limit: 10, Threshold: 0.001})
			if err != nil {
				t.Fatal(err)
			}
			got := make([]string, 0, len(results))
			for _, result := range results {
				got = append(got, result.Memory.ID)
			}
			slices.Sort(got)
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("unexpected model/cache records: got %v, want %v", got, want)
			}
		}
	}
	check(fixture.Before)
	if fixture.Write != "" {
		if err := store.Store(ctx, &memory.Memory{ID: identifier(fixture.Write), UserID: fixture.UserID, Content: query, Type: memory.MemoryTypeSemantic, CreatedAt: time.Now()}); err != nil {
			t.Fatal(err)
		}
	}
	check(fixture.After)
}
