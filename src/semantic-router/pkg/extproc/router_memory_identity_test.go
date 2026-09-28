package extproc

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func TestMemoryEmbeddingIdentityIsolatesBackendAndHotCache(t *testing.T) {
	t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", "")
	for _, backend := range []string{"milvus", "valkey", "qdrant"} {
		t.Run(backend, func(t *testing.T) {
			cfg := &config.RouterConfig{Memory: config.MemoryConfig{
				Backend: backend, EmbeddingModel: "mmbert",
				Milvus:     config.MemoryMilvusConfig{Collection: "existing"},
				Valkey:     &config.MemoryValkeyConfig{IndexName: "existing_index", CollectionPrefix: "existing:"},
				Qdrant:     &config.MemoryQdrantConfig{Collection: "existing"},
				RedisCache: &config.MemoryRedisCacheConfig{Enabled: true, KeyPrefix: "hot:"},
			}}
			before := *cfg
			vc, qc, rc := *cfg.Memory.Valkey, *cfg.Memory.Qdrant, *cfg.Memory.RedisCache
			before.Memory.Valkey, before.Memory.Qdrant, before.Memory.RedisCache = &vc, &qc, &rc
			bind := func(fingerprint string) *config.RouterConfig {
				t.Helper()
				result, err := memoryConfigForIdentity(cfg, func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
					if settings.Dimension != 256 || settings.Layer != 0 || settings.ModelType != "mmbert" || settings.InputPolicy == "" {
						t.Fatalf("identity does not describe actual memory inference: %+v", settings)
					}
					return embedding.ContentIdentity{Fingerprint: fingerprint, Descriptor: embedding.RuntimeDescriptor{Dimension: 256}}, nil
				})
				if err != nil {
					t.Fatal(err)
				}
				return result
			}
			a, reopened, b := bind("model-a"), bind("model-a"), bind("model-b")
			if !reflect.DeepEqual(a, reopened) || reflect.DeepEqual(a, b) {
				t.Fatal("memory namespaces are unstable or model changes reuse a namespace")
			}
			if !reflect.DeepEqual(cfg, &before) {
				t.Fatal("binding mutated user configuration")
			}
			if a.Memory.RedisCache.KeyPrefix == b.Memory.RedisCache.KeyPrefix || a.Memory.RedisCache.KeyPrefix == "hot:" {
				t.Fatal("retrieval hot cache can bypass vector identity")
			}
			cfg.Memory.Milvus.Collection = "another"
			cfg.Memory.Valkey.CollectionPrefix = "another:"
			cfg.Memory.Qdrant.Collection = "another"
			otherCollection := bind("model-a")
			if otherCollection.Memory.RedisCache.KeyPrefix == a.Memory.RedisCache.KeyPrefix {
				t.Fatal("hot cache mixed distinct logical collections")
			}
			switch backend {
			case "milvus":
				if a.Memory.Milvus.Collection == "existing" || a.Memory.Milvus.Collection == b.Memory.Milvus.Collection || a.Memory.Milvus.Dimension != 256 {
					t.Fatal("Milvus collection was not isolated")
				}
			case "valkey":
				if a.Memory.Valkey.IndexName == b.Memory.Valkey.IndexName || a.Memory.Valkey.CollectionPrefix == b.Memory.Valkey.CollectionPrefix || a.Memory.Valkey.CollectionPrefix == "existing:" || a.Memory.Valkey.Dimension != 256 {
					t.Fatal("Valkey index or indexed hash prefix was not isolated")
				}
			case "qdrant":
				if a.Memory.Qdrant.Collection == "existing" || a.Memory.Qdrant.Collection == b.Memory.Qdrant.Collection || a.Memory.Qdrant.Dimension != 256 {
					t.Fatal("Qdrant collection was not isolated")
				}
			}
		})
	}
}

func TestMemoryEmbeddingIdentityFailureDoesNotAdoptLegacyData(t *testing.T) {
	t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", "")
	cfg := &config.RouterConfig{Memory: config.MemoryConfig{EmbeddingModel: "mmbert"}}
	boom := errors.New("model is unavailable")
	got, err := memoryConfigForIdentity(cfg, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		return embedding.ContentIdentity{}, boom
	})
	if got != nil || !errors.Is(err, boom) {
		t.Fatalf("missing model fell through to untagged memory: %v, %v", got, err)
	}
	cfg.Memory.EmbeddingModel = "qwen3"
	got, err = memoryConfigForIdentity(cfg, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		t.Fatal("unsupported identity provider was initialized")
		return embedding.ContentIdentity{}, nil
	})
	if got != cfg || err != nil {
		t.Fatal("legacy non-mmbert memory changed")
	}
}

func TestCandleBERTMemoryOpensVersionedNamespace(t *testing.T) {
	t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", "")
	embed := func(context.Context, string) ([]float32, error) {
		return nil, errors.New("namespace binding ran inference")
	}
	for _, backend := range []string{"milvus", "valkey", "qdrant"} {
		t.Run(backend, func(t *testing.T) {
			cfg := &config.RouterConfig{Memory: config.MemoryConfig{
				Backend: backend, EmbeddingModel: "bert",
				Milvus:     config.MemoryMilvusConfig{Collection: "existing"},
				Valkey:     &config.MemoryValkeyConfig{IndexName: "existing_index", CollectionPrefix: "existing:"},
				Qdrant:     &config.MemoryQdrantConfig{Collection: "existing"},
				RedisCache: &config.MemoryRedisCacheConfig{Enabled: true, KeyPrefix: "hot:"},
			}}
			bind := func(runtime string) *config.RouterConfig {
				t.Helper()
				provider, err := embedding.NewFuncProvider(runtime, 384, embed)
				if err != nil {
					t.Fatal(err)
				}
				bound, err := bindMemoryEmbedding(cfg, embedding.NewSet(map[string]embedding.Provider{"bert": provider}, "bert"))
				if err != nil {
					t.Fatal(err)
				}
				return bound
			}
			upgraded, reopened := bind("candle"), bind("candle")
			if !reflect.DeepEqual(upgraded, reopened) || upgraded.Memory.RedisCache.KeyPrefix == "hot:" {
				t.Fatal("Candle BERT memory namespace is unstable or shares the old hot cache")
			}
			if bind("ort") != cfg {
				t.Fatal("BERT runtime with unchanged vectors moved to a new namespace")
			}
			switch backend {
			case "milvus":
				if upgraded.Memory.Milvus.Collection == "existing" || upgraded.Memory.Milvus.Dimension != 384 {
					t.Fatal("upgraded Candle BERT reopened the old Milvus collection")
				}
			case "valkey":
				if upgraded.Memory.Valkey.IndexName == "existing_index" || upgraded.Memory.Valkey.CollectionPrefix == "existing:" || upgraded.Memory.Valkey.Dimension != 384 {
					t.Fatal("upgraded Candle BERT reopened the old Valkey index")
				}
			case "qdrant":
				if upgraded.Memory.Qdrant.Collection == "existing" || upgraded.Memory.Qdrant.Dimension != 384 {
					t.Fatal("upgraded Candle BERT reopened the old Qdrant collection")
				}
			}
		})
	}
}

func TestMemoryEmbeddingIdentitySeparatesDeterministicSimulation(t *testing.T) {
	cfg := &config.RouterConfig{Memory: config.MemoryConfig{EmbeddingModel: "mmbert", RedisCache: &config.MemoryRedisCacheConfig{KeyPrefix: "hot:"}}}
	t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", "")
	native, err := memoryConfigForIdentity(cfg, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		return embedding.ContentIdentity{Fingerprint: "native-model", Descriptor: embedding.RuntimeDescriptor{Dimension: 256}}, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	var simulated *config.RouterConfig
	for _, value := range []string{"1", "true", "yes", "on", "deterministic"} {
		t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", value)
		got, err := memoryConfigForIdentity(cfg, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
			t.Fatal("simulation attempted to load or claim neural weights")
			return embedding.ContentIdentity{}, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		if simulated != nil && !reflect.DeepEqual(got, simulated) {
			t.Fatal("equivalent simulation flags changed representation identity")
		}
		if got.Memory.Milvus.Collection == native.Memory.Milvus.Collection || got.Memory.RedisCache.KeyPrefix == native.Memory.RedisCache.KeyPrefix {
			t.Fatal("simulation can pollute native model storage")
		}
		simulated = got
	}
}

func TestOmniMemoryIdentityResolvesActualDimensionAndIsolatesArtifacts(t *testing.T) {
	t.Setenv("VLLM_SR_DETERMINISTIC_EMBEDDINGS", "")
	for _, backend := range []string{"milvus", "valkey", "qdrant"} {
		for _, size := range []int{384, 768} {
			cfg := &config.RouterConfig{Memory: config.MemoryConfig{EmbeddingModel: "multimodal", Backend: backend, Valkey: &config.MemoryValkeyConfig{}, Qdrant: &config.MemoryQdrantConfig{}, RedisCache: &config.MemoryRedisCacheConfig{KeyPrefix: "hot:"}}}
			bind := func(name string) (*config.RouterConfig, error) {
				return memoryConfigForIdentity(cfg, func(settings embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
					if settings.ModelType != "multimodal" || settings.Dimension != 0 || settings.Layer != 0 {
						t.Fatalf("dimension guessed before resolving artifact: %+v", settings)
					}
					return embedding.ContentIdentity{Fingerprint: name, Descriptor: embedding.RuntimeDescriptor{Dimension: size}}, nil
				})
			}
			a, err := bind("artifact-a")
			if err != nil {
				t.Fatal(err)
			}
			b, err := bind("artifact-b")
			if err != nil {
				t.Fatal(err)
			}
			if a.Memory.RedisCache.KeyPrefix == b.Memory.RedisCache.KeyPrefix {
				t.Fatal("same-width distinct artifacts shared hot cache")
			}
			switch backend {
			case "milvus":
				if a.Memory.Milvus.Dimension != size || a.Memory.Milvus.Collection == b.Memory.Milvus.Collection {
					t.Fatal("wrong milvus representation")
				}
			case "valkey":
				if a.Memory.Valkey.Dimension != size || a.Memory.Valkey.IndexName == b.Memory.Valkey.IndexName {
					t.Fatal("wrong valkey representation")
				}
			case "qdrant":
				if a.Memory.Qdrant.Dimension != size || a.Memory.Qdrant.Collection == b.Memory.Qdrant.Collection {
					t.Fatal("wrong qdrant representation")
				}
			}
			if cfg.Memory.Milvus.Dimension != 0 || cfg.Memory.Valkey.Dimension != 0 || cfg.Memory.Qdrant.Dimension != 0 || cfg.Memory.RedisCache.KeyPrefix != "hot:" {
				t.Fatal("source config changed")
			}
		}
	}
	cfg := &config.RouterConfig{Memory: config.MemoryConfig{EmbeddingModel: "multimodal", Milvus: config.MemoryMilvusConfig{Dimension: 384}}}
	if _, err := memoryConfigForIdentity(cfg, func(embedding.ConsumerSettings) (embedding.ContentIdentity, error) {
		return embedding.ContentIdentity{Fingerprint: "mini", Descriptor: embedding.RuntimeDescriptor{Dimension: 768}}, nil
	}); err == nil {
		t.Fatal("wrong explicit width accepted")
	}
}
