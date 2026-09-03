//go:build !windows && cgo

package memory

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ============================================================================
// Write Operations Tests
// ============================================================================

func TestMilvusStore_Store_Success(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	var capturedColumns []entity.Column
	mockClient.InsertFunc = func(ctx context.Context, coll string, part string, cols ...entity.Column) (entity.Column, error) {
		capturedColumns = cols
		return nil, nil
	}

	memory := &Memory{
		ID:      "test-mem-1",
		Content: "User's budget is $10,000",
		UserID:  "user-123",
		Type:    MemoryTypeSemantic,
	}

	err := store.Store(ctx, memory)
	require.NoError(t, err)
	assert.Equal(t, 1, mockClient.InsertCallCount)

	// Verify columns were created
	assert.GreaterOrEqual(t, len(capturedColumns), 7, "Expected at least 7 columns for insert")
}

func TestMilvusStore_Store_MissingRequiredFields(t *testing.T) {
	store, _ := setupTestStore()
	ctx := context.Background()

	tests := []struct {
		name   string
		memory *Memory
		errMsg string
	}{
		{
			name:   "missing ID",
			memory: &Memory{Content: "test", UserID: "user-1"},
			errMsg: "memory ID is required",
		},
		{
			name:   "missing content",
			memory: &Memory{ID: "id-1", UserID: "user-1"},
			errMsg: "memory content is required",
		},
		{
			name:   "missing user ID",
			memory: &Memory{ID: "id-1", Content: "test"},
			errMsg: "user ID is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := store.Store(ctx, tt.memory)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tt.errMsg)
		})
	}
}

func TestMilvusStore_Store_DisabledStore(t *testing.T) {
	options := MilvusStoreOptions{Enabled: false}
	store, _ := NewMilvusStore(options)
	ctx := context.Background()

	err := store.Store(ctx, &Memory{ID: "1", Content: "test", UserID: "u1"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not enabled")
}

func TestMilvusStore_Get_Success(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	// Setup mock to return a memory
	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		// Verify the expression contains the ID filter
		assert.Contains(t, expr, "id == \"mem-123\"")

		return []entity.Column{
			entity.NewColumnVarChar("id", []string{"mem-123"}),
			entity.NewColumnVarChar("content", []string{"Test content"}),
			entity.NewColumnVarChar("user_id", []string{"user-456"}),
			entity.NewColumnVarChar("memory_type", []string{"semantic"}),
			entity.NewColumnVarChar("metadata", []string{`{"project_id":"proj-1","source":"test"}`}),
			entity.NewColumnInt64("created_at", []int64{1704067200}),
			entity.NewColumnInt64("updated_at", []int64{1704067200}),
		}, nil
	}

	memory, err := store.Get(ctx, "mem-123")
	require.NoError(t, err)
	require.NotNil(t, memory)
	assert.Equal(t, "mem-123", memory.ID)
	assert.Equal(t, "Test content", memory.Content)
	assert.Equal(t, "user-456", memory.UserID)
	assert.Equal(t, MemoryTypeSemantic, memory.Type)
	assert.Equal(t, "proj-1", memory.ProjectID)
	assert.Equal(t, "test", memory.Source)
}

func TestMilvusStore_Get_NotFound(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	// Return empty result
	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		return []entity.Column{}, nil
	}

	memory, err := store.Get(ctx, "non-existent")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "memory not found")
	assert.Nil(t, memory)
}

func TestMilvusStore_Forget_Success(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	var capturedExpr string
	mockClient.DeleteFunc = func(ctx context.Context, coll string, part string, expr string) error {
		capturedExpr = expr
		return nil
	}

	err := store.Forget(ctx, "mem-to-delete")
	require.NoError(t, err)
	assert.Equal(t, 1, mockClient.DeleteCallCount)
	assert.Contains(t, capturedExpr, "id == \"mem-to-delete\"")
}

func TestMilvusStore_Forget_MissingID(t *testing.T) {
	store, _ := setupTestStore()
	ctx := context.Background()

	err := store.Forget(ctx, "")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "memory ID is required")
}

func TestMilvusStore_ForgetByScope_UserOnly(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	var capturedExpr string
	mockClient.DeleteFunc = func(ctx context.Context, coll string, part string, expr string) error {
		capturedExpr = expr
		return nil
	}

	err := store.ForgetByScope(ctx, MemoryScope{UserID: "user-123"})
	require.NoError(t, err)
	assert.Contains(t, capturedExpr, "user_id == \"user-123\"")
}

func TestMilvusStore_ForgetByScope_WithTypes(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	var capturedExpr string
	mockClient.DeleteFunc = func(ctx context.Context, coll string, part string, expr string) error {
		capturedExpr = expr
		return nil
	}

	err := store.ForgetByScope(ctx, MemoryScope{
		UserID: "user-123",
		Types:  []MemoryType{MemoryTypeSemantic, MemoryTypeProcedural},
	})
	require.NoError(t, err)
	assert.Contains(t, capturedExpr, "user_id == \"user-123\"")
	assert.Contains(t, capturedExpr, "memory_type == \"semantic\"")
	assert.Contains(t, capturedExpr, "memory_type == \"procedural\"")
}

func TestMilvusStore_ForgetByScope_MissingUserID(t *testing.T) {
	store, _ := setupTestStore()
	ctx := context.Background()

	err := store.ForgetByScope(ctx, MemoryScope{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "user ID is required")
}

func TestMilvusStore_Update_Success(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	// Setup mock for Get (Query) - returns existing memory with embedding
	// Update calls Get to fetch CreatedAt and Embedding when missing from the input
	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		embedding := make([]float32, 384)
		embedding[0] = 0.1
		return []entity.Column{
			entity.NewColumnVarChar("id", []string{"mem-123"}),
			entity.NewColumnVarChar("content", []string{"Old content"}),
			entity.NewColumnVarChar("user_id", []string{"user-456"}),
			entity.NewColumnVarChar("memory_type", []string{"semantic"}),
			entity.NewColumnVarChar("metadata", []string{`{}`}),
			entity.NewColumnInt64("created_at", []int64{1704067200}),
			entity.NewColumnInt64("updated_at", []int64{1704067200}),
			entity.NewColumnFloatVector("embedding", 384, [][]float32{embedding}),
		}, nil
	}

	// Setup mock for Upsert (Update now uses atomic Upsert instead of Delete+Insert)
	mockClient.UpsertFunc = func(ctx context.Context, coll string, part string, cols ...entity.Column) (entity.Column, error) {
		return nil, nil
	}

	updatedMemory := &Memory{
		ID:      "mem-123",
		Content: "Updated content with new budget $15,000",
		UserID:  "user-456",
		Type:    MemoryTypeSemantic,
	}

	err := store.Update(ctx, "mem-123", updatedMemory)
	require.NoError(t, err)
	assert.Equal(t, 1, mockClient.UpsertCallCount, "Update should call Upsert once")
	assert.Equal(t, 0, mockClient.DeleteCallCount, "Update should not call Delete (uses Upsert)")
	assert.Equal(t, 0, mockClient.InsertCallCount, "Update should not call Insert (uses Upsert)")
}

func TestMilvusStore_Update_NotFound(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	// Return empty result for Get
	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		return []entity.Column{}, nil
	}

	err := store.Update(ctx, "non-existent", &Memory{
		ID:      "non-existent",
		Content: "New content",
		UserID:  "user-123",
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "memory not found")
}

// ============================================================================
// Schema Tests
// ============================================================================

func TestMilvusStore_Schema_UserIDPartitionKey(t *testing.T) {
	mockClient := &MockMilvusClient{}

	// Return false for HasCollection to trigger schema creation
	mockClient.HasCollectionFunc = func(ctx context.Context, coll string) (bool, error) {
		return false, nil
	}

	testEmbeddingConfig := EmbeddingConfig{
		Model: EmbeddingModelBERT,
	}

	config := DefaultMemoryConfig()
	config.Milvus.Dimension = 384

	options := MilvusStoreOptions{
		Client:          mockClient,
		CollectionName:  "test_partition_key",
		Config:          config,
		Enabled:         true,
		EmbeddingConfig: &testEmbeddingConfig,
	}

	_, err := NewMilvusStore(options)
	require.NoError(t, err)

	// Verify schema was captured
	require.NotNil(t, mockClient.CapturedSchema, "Schema should be captured during collection creation")

	// Find the user_id field and verify IsPartitionKey
	var userIDField *entity.Field
	for _, field := range mockClient.CapturedSchema.Fields {
		if field.Name == "user_id" {
			userIDField = field
			break
		}
	}

	require.NotNil(t, userIDField, "user_id field should exist in schema")
	assert.True(t, userIDField.IsPartitionKey, "user_id field should have IsPartitionKey=true for efficient per-user queries")
}

func TestMilvusStore_LegacyAndOmittedDimensionsKeepSameResult(t *testing.T) {
	baseConfig := DefaultMemoryConfig()
	baseConfig.EmbeddingModel = string(EmbeddingModelBERT)
	baseConfig.Milvus.Dimension = 384

	newStore := func(name string, milvusDimension, embeddingDimension int) *MilvusStore {
		t.Helper()
		storeConfig := baseConfig
		storeConfig.Milvus.Dimension = milvusDimension
		client := &MockMilvusClient{
			HasCollectionFunc: func(context.Context, string) (bool, error) {
				return false, nil
			},
		}
		store, err := NewMilvusStore(MilvusStoreOptions{
			Client:         client,
			CollectionName: name,
			Config:         storeConfig,
			Enabled:        true,
			EmbeddingConfig: &EmbeddingConfig{
				Model:     EmbeddingModelBERT,
				Dimension: embeddingDimension,
			},
		})
		require.NoError(t, err)
		return store
	}

	legacyStore := newStore("legacy_dimension", 384, 384)
	omittedStore := newStore("omitted_dimension", 512, 0)

	assert.Equal(t, legacyStore.effectiveDimension, omittedStore.effectiveDimension)
	assert.Equal(t, legacyStore.embeddingConfig.Dimension, omittedStore.embeddingConfig.Dimension)
	assert.Equal(t, legacyStore.config.Milvus.Dimension, omittedStore.config.Milvus.Dimension)
}

func TestMilvusStore_ExistingCollectionDimensionMismatch(t *testing.T) {
	mockClient := &MockMilvusClient{
		HasCollectionFunc: func(context.Context, string) (bool, error) {
			return true, nil
		},
		DescribeCollectionFunc: func(context.Context, string) (*entity.Collection, error) {
			return &entity.Collection{Schema: memoryCollectionSchema("existing", 768)}, nil
		},
	}
	store := &MilvusStore{
		client:             mockClient,
		collectionName:     "existing",
		config:             DefaultMemoryConfig(),
		enabled:            true,
		effectiveDimension: 384,
	}

	err := store.ensureCollection(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "stored=768")
	assert.Contains(t, err.Error(), "expected=384")
	assert.Zero(t, mockClient.LoadCollectionCallCount)
}

func TestMilvusStore_ExistingCollectionMatchingDimensionLoads(t *testing.T) {
	mockClient := &MockMilvusClient{
		HasCollectionFunc: func(context.Context, string) (bool, error) {
			return true, nil
		},
		DescribeCollectionFunc: func(context.Context, string) (*entity.Collection, error) {
			return &entity.Collection{Schema: memoryCollectionSchema("existing", 384)}, nil
		},
	}
	store := &MilvusStore{
		client:             mockClient,
		collectionName:     "existing",
		config:             DefaultMemoryConfig(),
		enabled:            true,
		effectiveDimension: 384,
	}

	require.NoError(t, store.ensureCollection(context.Background()))
	assert.Equal(t, 1, mockClient.DescribeCollectionCallCount)
	assert.Equal(t, 1, mockClient.LoadCollectionCallCount)
}

func TestMilvusStore_CreatesCollectionWithEffectiveDimension(t *testing.T) {
	mockClient := &MockMilvusClient{
		HasCollectionFunc: func(context.Context, string) (bool, error) {
			return false, nil
		},
	}
	store := &MilvusStore{
		client:             mockClient,
		collectionName:     "new",
		config:             DefaultMemoryConfig(),
		enabled:            true,
		effectiveDimension: 512,
	}

	require.NoError(t, store.ensureCollection(context.Background()))
	require.NotNil(t, mockClient.CapturedSchema)
	var embeddingField *entity.Field
	for _, field := range mockClient.CapturedSchema.Fields {
		if field.Name == "embedding" {
			embeddingField = field
			break
		}
	}
	require.NotNil(t, embeddingField)
	assert.Equal(t, "512", embeddingField.TypeParams["dim"])
	assert.Equal(t, 1, mockClient.LoadCollectionCallCount)
}

// ============================================================================
// Hybrid Search Tests
// ============================================================================

func TestMilvusStore_Retrieve_HybridRerank(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	// Candidate A: high vector score but content doesn't contain query terms.
	// Candidate B: slightly lower vector score but content contains exact query terms.
	// With hybrid search, B should be boosted above A.
	mockResults := []client.SearchResult{
		{
			ResultCount: 2,
			Scores:      []float32{0.90, 0.85},
			Fields: []entity.Column{
				entity.NewColumnVarChar("id", []string{"id_generic", "id_keyword"}),
				entity.NewColumnVarChar("content", []string{
					"The project timeline was discussed in the last meeting and deadlines were set",
					"Portland charity race is scheduled for March 15th with a $5000 budget",
				}),
				entity.NewColumnVarChar("memory_type", []string{"episodic", "episodic"}),
				entity.NewColumnVarChar("metadata", []string{"{}", "{}"}),
			},
		},
	}

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return mockResults, nil
	}

	// Vector-only retrieval: id_generic should rank first (0.90 > 0.85).
	vectorResults, err := store.Retrieve(ctx, RetrieveOptions{
		Query:        "Portland charity race",
		UserID:       "u1",
		Limit:        10,
		Threshold:    0.5,
		HybridSearch: false,
	})
	require.NoError(t, err)
	require.Len(t, vectorResults, 2)
	assert.Equal(t, "id_generic", vectorResults[0].Memory.ID,
		"Without hybrid, higher vector score should rank first")

	// Hybrid retrieval: id_keyword should rank first because BM25 + n-gram
	// boost exact "Portland", "charity", "race" terms.
	hybridResults, err := store.Retrieve(ctx, RetrieveOptions{
		Query:        "Portland charity race",
		UserID:       "u1",
		Limit:        10,
		Threshold:    0.1,
		HybridSearch: true,
		HybridMode:   "weighted",
	})
	require.NoError(t, err)
	require.Len(t, hybridResults, 2)
	assert.Equal(t, "id_keyword", hybridResults[0].Memory.ID,
		"With hybrid, exact keyword match should rank first")
}

func TestMilvusStore_Retrieve_HybridExpandsTopK(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	var capturedTopK int
	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		capturedTopK = topK
		return []client.SearchResult{{ResultCount: 0}}, nil
	}

	// With hybrid off: limit=5 -> topK = max(5*4, 20) = 20
	_, _ = store.Retrieve(ctx, RetrieveOptions{
		Query: "test", UserID: "u1", Limit: 5, HybridSearch: false,
	})
	assert.Equal(t, 20, capturedTopK, "Non-hybrid should use 4x multiplier (min 20)")

	// With hybrid on: limit=5 -> topK = max(5*8, 20) = 40
	_, _ = store.Retrieve(ctx, RetrieveOptions{
		Query: "test", UserID: "u1", Limit: 5, HybridSearch: true,
	})
	assert.Equal(t, 40, capturedTopK, "Hybrid should use 8x multiplier for broader candidate pool")
}
