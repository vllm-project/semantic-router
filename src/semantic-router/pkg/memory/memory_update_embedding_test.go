//go:build !windows

package memory

import (
	"context"
	"errors"
	"strconv"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type memoryUpdateEmbeddingCase struct {
	name              string
	content           string
	explicitEmbedding []float32
	providerEmbedding []float32
	providerErr       error
	expectedEmbedding []float32
	expectedCalls     []string
	wantError         bool
}

func memoryUpdateEmbeddingCases() []memoryUpdateEmbeddingCase {
	oldEmbedding := memoryUpdateTestVector(1)
	generatedEmbedding := memoryUpdateTestVector(2)
	explicitEmbedding := memoryUpdateTestVector(3)

	return []memoryUpdateEmbeddingCase{
		{
			name:              "changed content is re-embedded",
			content:           "updated content",
			providerEmbedding: generatedEmbedding,
			expectedEmbedding: generatedEmbedding,
			expectedCalls:     []string{"updated content"},
		},
		{
			name:              "unchanged content reuses existing vector",
			content:           "original content",
			providerEmbedding: generatedEmbedding,
			expectedEmbedding: oldEmbedding,
			expectedCalls:     []string{},
		},
		{
			name:              "explicit vector takes priority",
			content:           "updated content",
			explicitEmbedding: explicitEmbedding,
			providerEmbedding: generatedEmbedding,
			expectedEmbedding: explicitEmbedding,
			expectedCalls:     []string{},
		},
		{
			name:          "embedding failure prevents write",
			content:       "updated content",
			providerErr:   errors.New("embedding unavailable"),
			expectedCalls: []string{"updated content"},
			wantError:     true,
		},
	}
}

func memoryUpdateTestVector(seed float32) []float32 {
	vector := make([]float32, 384)
	for i := range vector {
		vector[i] = seed + float32(i)/1000
	}
	return vector
}

func memoryUpdateEmbeddingConfig(t *testing.T, testCase memoryUpdateEmbeddingCase) (EmbeddingConfig, *[]string) {
	t.Helper()
	calls := []string{}
	provider, err := embedding.NewFuncProvider("memory-update-test", 384, func(_ context.Context, text string) ([]float32, error) {
		calls = append(calls, text)
		if testCase.providerErr != nil {
			return nil, testCase.providerErr
		}
		return testCase.providerEmbedding, nil
	})
	require.NoError(t, err)
	return EmbeddingConfig{Provider: provider, Model: EmbeddingModelQwen3}, &calls
}

func memoryUpdateSequenceEmbeddingConfig(t *testing.T) (EmbeddingConfig, *[]string, []float32, []float32) {
	t.Helper()
	vectorA := memoryUpdateTestVector(4)
	vectorB := memoryUpdateTestVector(5)
	calls := []string{}
	provider, err := embedding.NewFuncProvider("memory-update-sequence-test", 384, func(_ context.Context, text string) ([]float32, error) {
		calls = append(calls, text)
		switch text {
		case "content A":
			return vectorA, nil
		case "content B":
			return vectorB, nil
		default:
			return nil, errors.New("unexpected embedding input")
		}
	})
	require.NoError(t, err)
	return EmbeddingConfig{Provider: provider, Model: EmbeddingModelQwen3}, &calls, vectorA, vectorB
}

func TestMilvusStore_UpdateEmbedding(t *testing.T) {
	ctx := context.Background()
	createdAt := time.Unix(1704067200, 0)
	oldEmbedding := memoryUpdateTestVector(1)

	for _, testCase := range memoryUpdateEmbeddingCases() {
		t.Run(testCase.name, func(t *testing.T) {
			store, mockClient := setupTestStore()
			embeddingConfig, calls := memoryUpdateEmbeddingConfig(t, testCase)
			store.embeddingConfig = embeddingConfig

			mockClient.QueryFunc = func(_ context.Context, _ string, _ []string, _ string, _ []string, _ ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
				return []entity.Column{
					entity.NewColumnVarChar("id", []string{"mem-update"}),
					entity.NewColumnVarChar("content", []string{"original content"}),
					entity.NewColumnVarChar("user_id", []string{"user-1"}),
					entity.NewColumnVarChar("memory_type", []string{"semantic"}),
					entity.NewColumnVarChar("metadata", []string{`{}`}),
					entity.NewColumnInt64("created_at", []int64{createdAt.Unix()}),
					entity.NewColumnInt64("updated_at", []int64{createdAt.Unix()}),
					entity.NewColumnFloatVector("embedding", len(oldEmbedding), [][]float32{oldEmbedding}),
				}, nil
			}

			var writtenEmbedding []float32
			var writtenCreatedAt int64
			var writtenUpdatedAt int64
			var writtenContent string
			mockClient.UpsertFunc = func(_ context.Context, _, _ string, columns ...entity.Column) (entity.Column, error) {
				for _, column := range columns {
					switch column.Name() {
					case "embedding":
						writtenEmbedding = append([]float32(nil), column.(*entity.ColumnFloatVector).Data()[0]...)
					case "content":
						writtenContent, _ = column.(*entity.ColumnVarChar).ValueByIdx(0)
					case "created_at":
						writtenCreatedAt, _ = column.(*entity.ColumnInt64).ValueByIdx(0)
					case "updated_at":
						writtenUpdatedAt, _ = column.(*entity.ColumnInt64).ValueByIdx(0)
					}
				}
				return nil, nil
			}

			updated := &Memory{
				Content:   testCase.content,
				UserID:    "user-1",
				Type:      MemoryTypeSemantic,
				Embedding: testCase.explicitEmbedding,
			}
			err := store.Update(ctx, "mem-update", updated)
			if testCase.wantError {
				require.Error(t, err)
				assert.Zero(t, mockClient.UpsertCallCount, "embedding failure must not persist")
			} else {
				require.NoError(t, err)
				assert.Equal(t, 1, mockClient.UpsertCallCount)
				assert.Equal(t, testCase.expectedEmbedding, writtenEmbedding, "assert the vector sent to Milvus")
				assert.Equal(t, createdAt.Unix(), writtenCreatedAt, "Update must preserve CreatedAt")
				assert.Equal(t, testCase.content, writtenContent)
				assert.Greater(t, writtenUpdatedAt, writtenCreatedAt, "Update must refresh UpdatedAt")
			}
			assert.Equal(t, testCase.expectedCalls, *calls)
		})
	}
}

func TestMilvusStore_UpdateSameMemoryEmbedsEachContent(t *testing.T) {
	ctx := context.Background()
	createdAt := time.Unix(1704067200, 0)
	oldEmbedding := memoryUpdateTestVector(1)

	for _, firstWriteFails := range []bool{false, true} {
		name := "after successful first update"
		if firstWriteFails {
			name = "after failed first update"
		}
		t.Run(name, func(t *testing.T) {
			store, mockClient := setupTestStore()
			embeddingConfig, calls, vectorA, vectorB := memoryUpdateSequenceEmbeddingConfig(t)
			store.embeddingConfig = embeddingConfig

			storedContent := "original content"
			storedEmbedding := oldEmbedding
			failNextUpsert := firstWriteFails
			var writtenContents []string
			var writtenVectors [][]float32
			mockClient.QueryFunc = func(_ context.Context, _ string, _ []string, _ string, _ []string, _ ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
				return []entity.Column{
					entity.NewColumnVarChar("id", []string{"mem-update"}),
					entity.NewColumnVarChar("content", []string{storedContent}),
					entity.NewColumnVarChar("user_id", []string{"user-1"}),
					entity.NewColumnVarChar("memory_type", []string{"semantic"}),
					entity.NewColumnVarChar("metadata", []string{`{}`}),
					entity.NewColumnInt64("created_at", []int64{createdAt.Unix()}),
					entity.NewColumnInt64("updated_at", []int64{createdAt.Unix()}),
					entity.NewColumnFloatVector("embedding", len(storedEmbedding), [][]float32{storedEmbedding}),
				}, nil
			}
			mockClient.UpsertFunc = func(_ context.Context, _, _ string, columns ...entity.Column) (entity.Column, error) {
				var content string
				var vector []float32
				for _, column := range columns {
					switch column.Name() {
					case "content":
						content, _ = column.(*entity.ColumnVarChar).ValueByIdx(0)
					case "embedding":
						vector = append([]float32(nil), column.(*entity.ColumnFloatVector).Data()[0]...)
					}
				}
				writtenContents = append(writtenContents, content)
				writtenVectors = append(writtenVectors, vector)
				if failNextUpsert {
					failNextUpsert = false
					return nil, errors.New("injected Milvus upsert failure")
				}
				storedContent = content
				storedEmbedding = vector
				return nil, nil
			}

			updated := &Memory{Content: "content A", UserID: "user-1", Type: MemoryTypeSemantic}
			err := store.Update(ctx, "mem-update", updated)
			if firstWriteFails {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			assert.Empty(t, updated.Embedding, "generated vectors must stay local to the write")

			updated.Content = "content B"
			require.NoError(t, store.Update(ctx, "mem-update", updated))
			assert.Empty(t, updated.Embedding, "a repeated update must not turn an internal vector into explicit input")
			assert.Equal(t, []string{"content A", "content B"}, *calls)
			assert.Equal(t, []string{"content A", "content B"}, writtenContents)
			assert.Equal(t, [][]float32{vectorA, vectorB}, writtenVectors)
			assert.Equal(t, "content B", storedContent)
			assert.Equal(t, vectorB, storedEmbedding)
			assert.Equal(t, createdAt, updated.CreatedAt)
		})
	}
}

type memoryUpdateValkeyClient struct {
	hashes            map[string]map[string]string
	hsetCalls         int
	lastHSetKey       string
	lastHSet          map[string]string
	failNextHSet      bool
	writtenContents   []string
	writtenEmbeddings [][]float32
}

func (c *memoryUpdateValkeyClient) CustomCommand(context.Context, []string) (any, error) {
	return nil, nil
}

func (c *memoryUpdateValkeyClient) HGetAll(_ context.Context, key string) (map[string]string, error) {
	fields := c.hashes[key]
	result := make(map[string]string, len(fields))
	for name, value := range fields {
		result[name] = value
	}
	return result, nil
}

func (c *memoryUpdateValkeyClient) HSet(_ context.Context, key string, fields map[string]string) (int64, error) {
	c.hsetCalls++
	c.lastHSetKey = key
	c.lastHSet = make(map[string]string, len(fields))
	for name, value := range fields {
		c.lastHSet[name] = value
	}
	c.writtenContents = append(c.writtenContents, fields["content"])
	c.writtenEmbeddings = append(c.writtenEmbeddings, valkeyBytesToFloat32([]byte(fields["embedding"])))
	if c.failNextHSet {
		c.failNextHSet = false
		return 0, errors.New("injected Valkey HSET failure")
	}
	for name, value := range fields {
		c.hashes[key][name] = value
	}
	return int64(len(fields)), nil
}

func (c *memoryUpdateValkeyClient) Del(_ context.Context, keys []string) (int64, error) {
	var deleted int64
	for _, key := range keys {
		if _, ok := c.hashes[key]; ok {
			delete(c.hashes, key)
			deleted++
		}
	}
	return deleted, nil
}

func TestValkeyStore_UpdateEmbedding(t *testing.T) {
	ctx := context.Background()
	createdAt := time.UnixMilli(1704067200123)
	oldEmbedding := memoryUpdateTestVector(1)

	for _, testCase := range memoryUpdateEmbeddingCases() {
		t.Run(testCase.name, func(t *testing.T) {
			embeddingConfig, calls := memoryUpdateEmbeddingConfig(t, testCase)
			existing := &Memory{
				ID:        "mem-update",
				Content:   "original content",
				UserID:    "user-1",
				Type:      MemoryTypeSemantic,
				CreatedAt: createdAt,
				Embedding: oldEmbedding,
			}
			fields, err := valkeyBuildHashFields(existing, oldEmbedding)
			require.NoError(t, err)
			client := &memoryUpdateValkeyClient{hashes: map[string]map[string]string{"mem:mem-update": fields}}
			store := &ValkeyStore{
				client:           client,
				enabled:          true,
				maxRetries:       1,
				embeddingConfig:  embeddingConfig,
				collectionPrefix: "mem:",
			}

			updated := &Memory{
				Content:   testCase.content,
				UserID:    "user-1",
				Type:      MemoryTypeSemantic,
				Embedding: testCase.explicitEmbedding,
			}
			err = store.Update(ctx, "mem-update", updated)
			if testCase.wantError {
				require.Error(t, err)
				assert.Zero(t, client.hsetCalls, "embedding failure must not persist")
			} else {
				require.NoError(t, err)
				assert.Equal(t, 1, client.hsetCalls)
				assert.Equal(t, "mem:mem-update", client.lastHSetKey)
				writtenEmbedding := valkeyBytesToFloat32([]byte(client.lastHSet["embedding"]))
				assert.Equal(t, testCase.expectedEmbedding, writtenEmbedding, "assert the vector sent to Valkey")
				assert.Equal(t, strconv.FormatInt(createdAt.UnixMilli(), 10), client.lastHSet["created_at"], "Update must preserve CreatedAt")
				assert.Equal(t, testCase.content, client.lastHSet["content"])
				updatedAt, parseErr := strconv.ParseInt(client.lastHSet["updated_at"], 10, 64)
				require.NoError(t, parseErr)
				assert.Greater(t, updatedAt, createdAt.UnixMilli(), "Update must refresh UpdatedAt")
			}
			assert.Equal(t, testCase.expectedCalls, *calls)
		})
	}
}

func TestValkeyStore_UpdateSameMemoryEmbedsEachContent(t *testing.T) {
	ctx := context.Background()
	createdAt := time.UnixMilli(1704067200123)
	oldEmbedding := memoryUpdateTestVector(1)

	for _, firstWriteFails := range []bool{false, true} {
		name := "after successful first update"
		if firstWriteFails {
			name = "after failed first update"
		}
		t.Run(name, func(t *testing.T) {
			embeddingConfig, calls, vectorA, vectorB := memoryUpdateSequenceEmbeddingConfig(t)
			existing := &Memory{
				ID:        "mem-update",
				Content:   "original content",
				UserID:    "user-1",
				Type:      MemoryTypeSemantic,
				CreatedAt: createdAt,
				Embedding: oldEmbedding,
			}
			fields, err := valkeyBuildHashFields(existing, oldEmbedding)
			require.NoError(t, err)
			client := &memoryUpdateValkeyClient{
				hashes:       map[string]map[string]string{"mem:mem-update": fields},
				failNextHSet: firstWriteFails,
			}
			store := &ValkeyStore{
				client:           client,
				enabled:          true,
				maxRetries:       1,
				embeddingConfig:  embeddingConfig,
				collectionPrefix: "mem:",
			}

			updated := &Memory{Content: "content A", UserID: "user-1", Type: MemoryTypeSemantic}
			err = store.Update(ctx, "mem-update", updated)
			if firstWriteFails {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			assert.Empty(t, updated.Embedding, "generated vectors must stay local to the write")

			updated.Content = "content B"
			require.NoError(t, store.Update(ctx, "mem-update", updated))
			assert.Empty(t, updated.Embedding, "a repeated update must not turn an internal vector into explicit input")
			assert.Equal(t, []string{"content A", "content B"}, *calls)
			assert.Equal(t, 2, client.hsetCalls)
			assert.Equal(t, []string{"content A", "content B"}, client.writtenContents)
			assert.Equal(t, [][]float32{vectorA, vectorB}, client.writtenEmbeddings)

			storedFields := client.hashes["mem:mem-update"]
			assert.Equal(t, "content B", storedFields["content"])
			assert.Equal(t, vectorB, valkeyBytesToFloat32([]byte(storedFields["embedding"])))
			assert.Equal(t, strconv.FormatInt(createdAt.UnixMilli(), 10), storedFields["created_at"])
			assert.Equal(t, createdAt, updated.CreatedAt)
		})
	}
}
