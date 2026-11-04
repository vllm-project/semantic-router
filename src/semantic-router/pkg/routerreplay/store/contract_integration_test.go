package store

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

func contractEndpoint(t *testing.T, hostEnv, portEnv string, port int) (string, int) {
	host := os.Getenv(hostEnv)
	if host == "" {
		host = "localhost"
	}
	if configured := os.Getenv(portEnv); configured != "" {
		parsed, err := strconv.Atoi(configured)
		if err != nil {
			t.Fatalf("invalid %s: %v", portEnv, err)
		}
		port = parsed
	}
	return host, port
}

func contractName() string {
	return fmt.Sprintf("replay_contract_%d", time.Now().UnixNano())
}

// StorageIntegration: postgres
func TestPostgresStoreContractIntegration(t *testing.T) {
	storagetest.Require(t, "postgres")
	raw := os.Getenv("ROUTER_REPLAY_TEST_POSTGRES_CONFIG")
	if raw == "" {
		storagetest.Unavailable(t, "postgres", "set ROUTER_REPLAY_TEST_POSTGRES_CONFIG to run against PostgreSQL")
	}
	runStoreContract(t, func(t *testing.T, ttlSeconds int) Storage {
		var cfg PostgresConfig
		if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
			t.Fatal("invalid PostgreSQL test config")
		}
		cfg.TableName = contractName()
		s, err := NewPostgresStore(&cfg, ttlSeconds, false)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			//nolint:gosec // generated table name is validated by NewPostgresStore
			_, _ = s.db.Exec("DROP TABLE IF EXISTS " + cfg.TableName)
			_ = s.Close()
		})
		return s
	}, 2500, 300)
}

// StorageIntegration: redis
func TestRedisStoreContractIntegration(t *testing.T) {
	storagetest.Require(t, "redis")
	host, port := contractEndpoint(t, "REDIS_HOST", "REDIS_PORT", 6379)
	runStoreContract(t, func(t *testing.T, ttlSeconds int) Storage {
		s, err := NewRedisStore(&RedisConfig{Address: fmt.Sprintf("%s:%d", host, port), KeyPrefix: contractName() + ":"}, ttlSeconds, false)
		if err != nil {
			storagetest.Unavailable(t, "redis", err)
		}
		t.Cleanup(func() {
			ctx := context.Background()
			keys, _ := s.client.Keys(ctx, s.keyPrefix+"*").Result()
			_ = s.client.Del(ctx, append(keys, s.indexKey, s.readyKey)...).Err()
			_ = s.Close()
		})
		return s
	}, 2500, 300)
}

// StorageIntegration: redis
func TestRedisStoreFinishesAnInterruptedIndexPass(t *testing.T) {
	storagetest.Require(t, "redis")
	host, port := contractEndpoint(t, "REDIS_HOST", "REDIS_PORT", 6379)
	cfg := &RedisConfig{Address: fmt.Sprintf("%s:%d", host, port), KeyPrefix: contractName() + ":"}
	s, err := NewRedisStore(cfg, 0, false)
	if err != nil {
		storagetest.Unavailable(t, "redis", err)
	}
	ctx := context.Background()
	t.Cleanup(func() {
		keys, _ := s.client.Keys(ctx, s.keyPrefix+"*").Result()
		_ = s.client.Del(ctx, append(keys, s.indexKey, s.readyKey)...).Err()
		_ = s.client.Close()
	})
	for i := range 3 {
		if _, err = s.Add(ctx, Record{ID: fmt.Sprintf("rec-%d", i), Timestamp: time.Now().UTC()}); err != nil {
			t.Fatal(err)
		}
	}
	// Simulate a pass that stopped partway.
	if err = s.client.ZRem(ctx, s.indexKey, "rec-1", "rec-2").Err(); err != nil {
		t.Fatal(err)
	}
	if err = s.client.Del(ctx, s.readyKey).Err(); err != nil {
		t.Fatal(err)
	}

	reopened, err := NewRedisStore(cfg, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	page, err := reopened.QueryPage(ctx, QueryFilters{}, 10, 0, false)
	if err != nil || page.Total != 3 {
		t.Fatalf("reopened store lists %d records (err %v), want 3", page.Total, err)
	}
}

// StorageIntegration: qdrant
func TestQdrantStoreContractIntegration(t *testing.T) {
	storagetest.Require(t, "qdrant")
	host, port := contractEndpoint(t, "QDRANT_HOST", "QDRANT_PORT", 6334)
	runStoreContract(t, func(t *testing.T, ttlSeconds int) Storage {
		s, err := NewQdrantStore(&QdrantConfig{Host: host, Port: port, CollectionName: contractName()}, ttlSeconds, false)
		if err != nil {
			storagetest.Unavailable(t, "qdrant", err)
		}
		t.Cleanup(func() {
			_ = s.client.DeleteCollection(context.Background(), s.collectionName)
			_ = s.Close()
		})
		return s
	}, 1000, 300)
}

// StorageIntegration: milvus
func TestMilvusStoreContractIntegration(t *testing.T) {
	storagetest.Require(t, "milvus")
	host, port := contractEndpoint(t, "MILVUS_HOST", "MILVUS_PORT", 19530)
	runStoreContract(t, func(t *testing.T, ttlSeconds int) Storage {
		s, err := NewMilvusStore(&MilvusConfig{
			Address:          fmt.Sprintf("%s:%d", host, port),
			CollectionName:   contractName(),
			ConsistencyLevel: "Strong",
		}, ttlSeconds, false)
		if err != nil {
			storagetest.Unavailable(t, "milvus", err)
		}
		t.Cleanup(func() {
			_ = s.client.DropCollection(context.Background(), s.collectionName)
			_ = s.Close()
		})
		return s
	}, 30, 10)
}

// StorageIntegration: milvus
func TestMilvusStorePagesASecondFullerThanAQuery(t *testing.T) {
	storagetest.Require(t, "milvus")
	host, port := contractEndpoint(t, "MILVUS_HOST", "MILVUS_PORT", 19530)
	s, err := NewMilvusStore(&MilvusConfig{
		Address:        fmt.Sprintf("%s:%d", host, port),
		CollectionName: contractName(),
		// Strong waits on every read, and this test reads a few hundred times.
		ConsistencyLevel: "Session",
	}, 0, false)
	if err != nil {
		storagetest.Unavailable(t, "milvus", err)
	}
	ctx := context.Background()
	t.Cleanup(func() {
		_ = s.client.DropCollection(ctx, s.collectionName)
		_ = s.Close()
	})
	// Add flushes every record, so write the second in one insert.
	records := milvusWindowRows + 100
	var ids, data []string
	var timestamps []int64
	var vectors [][]float32
	for _, record := range denseRecords(records) {
		encoded, encodeErr := encodeRecord(&record, milvusMaxRecordBytes)
		if encodeErr != nil {
			t.Fatal(encodeErr)
		}
		ids, data = append(ids, record.ID), append(data, string(encoded))
		timestamps, vectors = append(timestamps, record.Timestamp.Unix()), append(vectors, []float32{0, 0})
	}
	if _, err = s.client.Insert(ctx, s.collectionName, "",
		entity.NewColumnVarChar("id", ids), entity.NewColumnInt64("timestamp", timestamps),
		entity.NewColumnVarChar("data", data), entity.NewColumnFloatVector("vector", 2, vectors)); err != nil {
		t.Fatal(err)
	}
	if err = s.client.Flush(ctx, s.collectionName, false); err != nil {
		t.Fatal(err)
	}
	s.wrote.Store(true)
	checkDensePages(t, s, records)
}
