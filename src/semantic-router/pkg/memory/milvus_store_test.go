//go:build !windows && cgo

package memory

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// TestMain initializes the embedding model before running tests
func TestMain(m *testing.M) {
	// Initialize embedding model for tests
	err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true)
	if err != nil {
		os.Exit(1)
	}

	// Run tests
	code := m.Run()
	os.Exit(code)
}

// MockMilvusClient facilitates testing without a running Milvus instance
type MockMilvusClient struct {
	SearchFunc        func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error)
	HasCollectionFunc func(ctx context.Context, coll string) (bool, error)
	InsertFunc        func(ctx context.Context, coll string, part string, cols ...entity.Column) (entity.Column, error)
	DeleteFunc        func(ctx context.Context, coll string, part string, expr string) error
	QueryFunc         func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error)
	SearchCallCount   int
	InsertCallCount   int
	DeleteCallCount   int
	QueryCallCount    int
}

func (m *MockMilvusClient) Search(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
	m.SearchCallCount++
	if m.SearchFunc != nil {
		return m.SearchFunc(ctx, coll, parts, expr, out, vectors, vField, mType, topK, sp, opts...)
	}
	return nil, errors.New("SearchFunc not implemented")
}

func (m *MockMilvusClient) HasCollection(ctx context.Context, coll string) (bool, error) {
	if m.HasCollectionFunc != nil {
		return m.HasCollectionFunc(ctx, coll)
	}
	return true, nil
}

// Stub out other required methods to satisfy client.Client interface
func (m *MockMilvusClient) Close() error                                             { return nil }
func (m *MockMilvusClient) CheckHealth(context.Context) (*entity.MilvusState, error) { return nil, nil }
func (m *MockMilvusClient) UsingDatabase(context.Context, string) error              { return nil }
func (m *MockMilvusClient) ListDatabases(context.Context) ([]entity.Database, error) { return nil, nil }
func (m *MockMilvusClient) CreateDatabase(context.Context, string, ...client.CreateDatabaseOption) error {
	return nil
}

func (m *MockMilvusClient) DropDatabase(context.Context, string, ...client.DropDatabaseOption) error {
	return nil
}

func (m *MockMilvusClient) AlterDatabase(context.Context, string, ...entity.DatabaseAttribute) error {
	return nil
}

func (m *MockMilvusClient) DescribeDatabase(context.Context, string) (*entity.Database, error) {
	return nil, nil
}

func (m *MockMilvusClient) NewCollection(context.Context, string, int64, ...client.CreateCollectionOption) error {
	return nil
}

func (m *MockMilvusClient) ListCollections(context.Context, ...client.ListCollectionOption) ([]*entity.Collection, error) {
	return nil, nil
}

func (m *MockMilvusClient) CreateCollection(context.Context, *entity.Schema, int32, ...client.CreateCollectionOption) error {
	return nil
}

func (m *MockMilvusClient) DescribeCollection(context.Context, string) (*entity.Collection, error) {
	return nil, nil
}

func (m *MockMilvusClient) DropCollection(context.Context, string, ...client.DropCollectionOption) error {
	return nil
}

func (m *MockMilvusClient) GetCollectionStatistics(context.Context, string) (map[string]string, error) {
	return nil, nil
}

func (m *MockMilvusClient) LoadCollection(context.Context, string, bool, ...client.LoadCollectionOption) error {
	return nil
}

func (m *MockMilvusClient) ReleaseCollection(context.Context, string, ...client.ReleaseCollectionOption) error {
	return nil
}
func (m *MockMilvusClient) RenameCollection(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) AlterCollection(context.Context, string, ...entity.CollectionAttribute) error {
	return nil
}
func (m *MockMilvusClient) CreateAlias(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) DropAlias(context.Context, string) error           { return nil }
func (m *MockMilvusClient) AlterAlias(context.Context, string, string) error  { return nil }
func (m *MockMilvusClient) GetReplicas(context.Context, string) ([]*entity.ReplicaGroup, error) {
	return nil, nil
}
func (m *MockMilvusClient) BackupRBAC(context.Context) (*entity.RBACMeta, error)   { return nil, nil }
func (m *MockMilvusClient) RestoreRBAC(context.Context, *entity.RBACMeta) error    { return nil }
func (m *MockMilvusClient) CreateCredential(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) UpdateCredential(context.Context, string, string, string) error {
	return nil
}
func (m *MockMilvusClient) DeleteCredential(context.Context, string) error       { return nil }
func (m *MockMilvusClient) ListCredUsers(context.Context) ([]string, error)      { return nil, nil }
func (m *MockMilvusClient) CreateRole(context.Context, string) error             { return nil }
func (m *MockMilvusClient) DropRole(context.Context, string) error               { return nil }
func (m *MockMilvusClient) AddUserRole(context.Context, string, string) error    { return nil }
func (m *MockMilvusClient) RemoveUserRole(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) ListRoles(context.Context) ([]entity.Role, error)     { return nil, nil }
func (m *MockMilvusClient) ListUsers(context.Context) ([]entity.User, error)     { return nil, nil }
func (m *MockMilvusClient) Grant(context.Context, string, entity.PriviledgeObjectType, string, string, ...entity.OperatePrivilegeOption) error {
	return nil
}

func (m *MockMilvusClient) Revoke(context.Context, string, entity.PriviledgeObjectType, string, string, ...entity.OperatePrivilegeOption) error {
	return nil
}

func (m *MockMilvusClient) ListGrant(context.Context, string, string, string, string) ([]entity.RoleGrants, error) {
	return nil, nil
}

func (m *MockMilvusClient) ListGrants(context.Context, string, string) ([]entity.RoleGrants, error) {
	return nil, nil
}

func (m *MockMilvusClient) CreatePartition(context.Context, string, string, ...client.CreatePartitionOption) error {
	return nil
}

func (m *MockMilvusClient) DropPartition(context.Context, string, string, ...client.DropPartitionOption) error {
	return nil
}

func (m *MockMilvusClient) ShowPartitions(context.Context, string) ([]*entity.Partition, error) {
	return nil, nil
}

func (m *MockMilvusClient) HasPartition(context.Context, string, string) (bool, error) {
	return false, nil
}

func (m *MockMilvusClient) LoadPartitions(context.Context, string, []string, bool, ...client.LoadPartitionsOption) error {
	return nil
}

func (m *MockMilvusClient) ReleasePartitions(context.Context, string, []string, ...client.ReleasePartitionsOption) error {
	return nil
}

func (m *MockMilvusClient) GetPersistentSegmentInfo(context.Context, string) ([]*entity.Segment, error) {
	return nil, nil
}

func (m *MockMilvusClient) CreateIndex(context.Context, string, string, entity.Index, bool, ...client.IndexOption) error {
	return nil
}

func (m *MockMilvusClient) DescribeIndex(context.Context, string, string, ...client.IndexOption) ([]entity.Index, error) {
	return nil, nil
}

func (m *MockMilvusClient) DropIndex(context.Context, string, string, ...client.IndexOption) error {
	return nil
}

func (m *MockMilvusClient) GetIndexState(context.Context, string, string, ...client.IndexOption) (entity.IndexState, error) {
	return 0, nil
}

func (m *MockMilvusClient) AlterIndex(context.Context, string, string, ...client.IndexOption) error {
	return nil
}

func (m *MockMilvusClient) GetIndexBuildProgress(context.Context, string, string, ...client.IndexOption) (int64, int64, error) {
	return 0, 0, nil
}

func (m *MockMilvusClient) Insert(ctx context.Context, coll string, part string, cols ...entity.Column) (entity.Column, error) {
	m.InsertCallCount++
	if m.InsertFunc != nil {
		return m.InsertFunc(ctx, coll, part, cols...)
	}
	return nil, nil
}

func (m *MockMilvusClient) Flush(context.Context, string, bool, ...client.FlushOption) error {
	return nil
}

func (m *MockMilvusClient) FlushV2(context.Context, string, bool, ...client.FlushOption) ([]int64, []int64, int64, map[string]msgpb.MsgPosition, error) {
	return nil, nil, 0, make(map[string]msgpb.MsgPosition), nil
}

func (m *MockMilvusClient) DeleteByPks(context.Context, string, string, entity.Column) error {
	return nil
}

func (m *MockMilvusClient) Delete(ctx context.Context, coll string, part string, expr string) error {
	m.DeleteCallCount++
	if m.DeleteFunc != nil {
		return m.DeleteFunc(ctx, coll, part, expr)
	}
	return nil
}

func (m *MockMilvusClient) Upsert(context.Context, string, string, ...entity.Column) (entity.Column, error) {
	return nil, nil
}

func (m *MockMilvusClient) QueryByPks(context.Context, string, []string, entity.Column, []string, ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
	return nil, nil
}

func (m *MockMilvusClient) Query(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
	m.QueryCallCount++
	if m.QueryFunc != nil {
		return m.QueryFunc(ctx, coll, parts, expr, out, opts...)
	}
	return nil, nil
}

func (m *MockMilvusClient) Get(context.Context, string, entity.Column, ...client.GetOption) (client.ResultSet, error) {
	return nil, nil
}

func (m *MockMilvusClient) QueryIterator(context.Context, *client.QueryIteratorOption) (*client.QueryIterator, error) {
	return nil, nil
}

func (m *MockMilvusClient) CalcDistance(context.Context, string, []string, entity.MetricType, entity.Column, entity.Column) (entity.Column, error) {
	return nil, nil
}

func (m *MockMilvusClient) CreateCollectionByRow(context.Context, entity.Row, int32) error {
	return nil
}

func (m *MockMilvusClient) InsertByRows(context.Context, string, string, []entity.Row) (entity.Column, error) {
	return nil, nil
}

func (m *MockMilvusClient) InsertRows(context.Context, string, string, []interface{}) (entity.Column, error) {
	return nil, nil
}

func (m *MockMilvusClient) ManualCompaction(context.Context, string, time.Duration) (int64, error) {
	return 0, nil
}

func (m *MockMilvusClient) GetCompactionState(context.Context, int64) (entity.CompactionState, error) {
	return 0, nil
}

func (m *MockMilvusClient) GetCompactionStateWithPlans(context.Context, int64) (entity.CompactionState, []entity.CompactionPlan, error) {
	return 0, nil, nil
}

func (m *MockMilvusClient) BulkInsert(context.Context, string, string, []string, ...client.BulkInsertOption) (int64, error) {
	return 0, nil
}

func (m *MockMilvusClient) GetBulkInsertState(context.Context, int64) (*entity.BulkInsertTaskState, error) {
	return nil, nil
}

func (m *MockMilvusClient) ListBulkInsertTasks(context.Context, string, int64) ([]*entity.BulkInsertTaskState, error) {
	return nil, nil
}

func (m *MockMilvusClient) CreateResourceGroup(context.Context, string, ...client.CreateResourceGroupOption) error {
	return nil
}

func (m *MockMilvusClient) UpdateResourceGroups(context.Context, ...client.UpdateResourceGroupsOption) error {
	return nil
}
func (m *MockMilvusClient) DropResourceGroup(context.Context, string) error { return nil }
func (m *MockMilvusClient) DescribeResourceGroup(context.Context, string) (*entity.ResourceGroup, error) {
	return nil, nil
}
func (m *MockMilvusClient) ListResourceGroups(context.Context) ([]string, error)      { return nil, nil }
func (m *MockMilvusClient) TransferNode(context.Context, string, string, int32) error { return nil }
func (m *MockMilvusClient) TransferReplica(context.Context, string, string, string, int64) error {
	return nil
}

func (m *MockMilvusClient) DescribeUser(context.Context, string) (entity.UserDescription, error) {
	return entity.UserDescription{}, nil
}

func (m *MockMilvusClient) DescribeUsers(context.Context) ([]entity.UserDescription, error) {
	return nil, nil
}

func (m *MockMilvusClient) GetLoadingProgress(context.Context, string, []string) (int64, error) {
	return 0, nil
}

func (m *MockMilvusClient) GetLoadState(context.Context, string, []string) (entity.LoadState, error) {
	return 0, nil
}
func (m *MockMilvusClient) GetVersion(context.Context) (string, error) { return "", nil }
func (m *MockMilvusClient) HybridSearch(context.Context, string, []string, int, []string, client.Reranker, []*client.ANNSearchRequest, ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
	return nil, nil
}

func (m *MockMilvusClient) ReplicateMessage(context.Context, string, uint64, uint64, [][]byte, []*msgpb.MsgPosition, []*msgpb.MsgPosition, ...client.ReplicateMessageOption) (*entity.MessageInfo, error) {
	return nil, nil
}

func setupTestStore() (*MilvusStore, *MockMilvusClient) {
	mockClient := &MockMilvusClient{}
	// Use bert embedding config for tests since that's initialized in TestMain
	testEmbeddingConfig := EmbeddingConfig{
		Model: EmbeddingModelBERT,
	}
	options := MilvusStoreOptions{
		Client:          mockClient,
		CollectionName:  "test_memories",
		Config:          DefaultMemoryConfig(),
		Enabled:         true,
		EmbeddingConfig: &testEmbeddingConfig,
	}
	store, _ := NewMilvusStore(options)
	return store, mockClient
}

func TestMilvusStore_Retrieve_InflateJSONMetadata(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	mockResults := []client.SearchResult{
		{
			ResultCount: 1,
			Scores:      []float32{0.95},
			Fields: []entity.Column{
				entity.NewColumnVarChar("id", []string{"mem_1"}),
				entity.NewColumnVarChar("content", []string{"The budget is $50k"}),
				entity.NewColumnVarChar("memory_type", []string{"semantic"}),
				entity.NewColumnVarChar("metadata", []string{`{"source": "slack", "importance": "high"}`}),
			},
		},
	}

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		// Verify TopK floor of 20 (default limit 5 * 4 = 20, or minimum 20)
		assert.GreaterOrEqual(t, topK, 20, "Expected topK to be at least 20, got %d", topK)
		return mockResults, nil
	}

	results, err := store.Retrieve(ctx, RetrieveOptions{Query: "budget", UserID: "u1", Limit: 5})
	require.NoError(t, err)
	require.Len(t, results, 1)

	// Verify Memory structure and metadata inflation
	require.NotNil(t, results[0].Memory)
	assert.Equal(t, "mem_1", results[0].Memory.ID)
	assert.Equal(t, "The budget is $50k", results[0].Memory.Content)
	assert.Equal(t, MemoryTypeSemantic, results[0].Memory.Type)
	assert.Equal(t, "slack", results[0].Memory.Source)
	assert.Equal(t, float32(0.95), results[0].Score)
	// Note: "importance": "high" is a string, so it won't be set in Memory.Importance (which is float32)
}

func TestMilvusStore_Retrieve_FilterByThreshold(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	mockResults := []client.SearchResult{
		{
			ResultCount: 2,
			Scores:      []float32{0.85, 0.45}, // 0.45 should be dropped
			Fields: []entity.Column{
				entity.NewColumnVarChar("id", []string{"id1", "id2"}),
				entity.NewColumnVarChar("content", []string{"c1", "c2"}),
				entity.NewColumnVarChar("memory_type", []string{"semantic", "semantic"}),
			},
		},
	}

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return mockResults, nil
	}

	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query: "test", UserID: "u1", Threshold: 0.6,
	})
	require.NoError(t, err)
	require.Len(t, results, 1)
	require.NotNil(t, results[0].Memory)
	assert.Equal(t, "id1", results[0].Memory.ID)
	assert.Equal(t, float32(0.85), results[0].Score)
}

func TestMilvusStore_Retrieve_DefaultThreshold(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	// DefaultMemoryConfig has DefaultSimilarityThreshold: 0.70
	mockResults := []client.SearchResult{
		{
			ResultCount: 3,
			Scores:      []float32{0.85, 0.75, 0.45}, // 0.45 should be dropped with default 0.70
			Fields: []entity.Column{
				entity.NewColumnVarChar("id", []string{"id1", "id2", "id3"}),
				entity.NewColumnVarChar("content", []string{"c1", "c2", "c3"}),
				entity.NewColumnVarChar("memory_type", []string{"semantic", "semantic", "semantic"}),
			},
		},
	}

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return mockResults, nil
	}

	// Test with Threshold = 0 (should use default 0.70)
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query: "test", UserID: "u1", Threshold: 0,
	})
	require.NoError(t, err)
	require.Len(t, results, 2, "Should filter out score 0.45 with default threshold 0.70")

	// Verify all results meet default threshold
	for _, result := range results {
		assert.GreaterOrEqual(t, result.Score, float32(0.70),
			"Result score %.4f should be >= default threshold 0.70", result.Score)
	}
}

func TestMilvusStore_Retrieve_ThresholdBoundary(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	mockResults := []client.SearchResult{
		{
			ResultCount: 3,
			Scores:      []float32{0.85, 0.60, 0.59}, // Test boundary at 0.6
			Fields: []entity.Column{
				entity.NewColumnVarChar("id", []string{"id1", "id2", "id3"}),
				entity.NewColumnVarChar("content", []string{"c1", "c2", "c3"}),
				entity.NewColumnVarChar("memory_type", []string{"semantic", "semantic", "semantic"}),
			},
		},
	}

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return mockResults, nil
	}

	// Test with threshold exactly at 0.6
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query: "test", UserID: "u1", Threshold: 0.6,
	})
	require.NoError(t, err)

	// Should include 0.85 and 0.60, exclude 0.59
	require.Len(t, results, 2, "Should include scores >= 0.6")

	// Verify all results meet threshold
	for _, result := range results {
		assert.GreaterOrEqual(t, result.Score, float32(0.6),
			"Result score %.4f should be >= 0.6 threshold", result.Score)
	}

	// Verify specific scores
	scores := make([]float32, len(results))
	for i, r := range results {
		scores[i] = r.Score
	}
	assert.Contains(t, scores, float32(0.85), "Should include high score")
	assert.Contains(t, scores, float32(0.60), "Should include boundary score")
	assert.NotContains(t, scores, float32(0.59), "Should exclude score below threshold")
}

func TestMilvusStore_Retrieve_DifferentThresholdValues(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	mockResults := []client.SearchResult{
		{
			ResultCount: 4,
			Scores:      []float32{0.95, 0.75, 0.55, 0.35},
			Fields: []entity.Column{
				entity.NewColumnVarChar("id", []string{"id1", "id2", "id3", "id4"}),
				entity.NewColumnVarChar("content", []string{"c1", "c2", "c3", "c4"}),
				entity.NewColumnVarChar("memory_type", []string{"semantic", "semantic", "semantic", "semantic"}),
			},
		},
	}

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return mockResults, nil
	}

	// Test with low threshold (0.3)
	resultsLow, err := store.Retrieve(ctx, RetrieveOptions{
		Query: "test", UserID: "u1", Threshold: 0.3,
	})
	require.NoError(t, err)

	// Test with default threshold (0.6)
	resultsDefault, err := store.Retrieve(ctx, RetrieveOptions{
		Query: "test", UserID: "u1", Threshold: 0.6,
	})
	require.NoError(t, err)

	// Test with high threshold (0.8)
	resultsHigh, err := store.Retrieve(ctx, RetrieveOptions{
		Query: "test", UserID: "u1", Threshold: 0.8,
	})
	require.NoError(t, err)

	// Verify threshold ordering: more results with lower threshold
	assert.GreaterOrEqual(t, len(resultsLow), len(resultsDefault),
		"Lower threshold (0.3) should return more or equal results than default (0.6)")
	assert.GreaterOrEqual(t, len(resultsDefault), len(resultsHigh),
		"Default threshold (0.6) should return more or equal results than high (0.8)")

	// Verify all results meet their respective thresholds
	for _, result := range resultsLow {
		assert.GreaterOrEqual(t, result.Score, float32(0.3))
	}
	for _, result := range resultsDefault {
		assert.GreaterOrEqual(t, result.Score, float32(0.6))
	}
	for _, result := range resultsHigh {
		assert.GreaterOrEqual(t, result.Score, float32(0.8))
	}
}

func TestMilvusStore_Retrieve_ThresholdVeryHigh(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	mockResults := []client.SearchResult{
		{
			ResultCount: 2,
			Scores:      []float32{0.95, 0.85}, // Both above 0.9
			Fields: []entity.Column{
				entity.NewColumnVarChar("id", []string{"id1", "id2"}),
				entity.NewColumnVarChar("content", []string{"c1", "c2"}),
				entity.NewColumnVarChar("memory_type", []string{"semantic", "semantic"}),
			},
		},
	}

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return mockResults, nil
	}

	// Test with very high threshold (0.9)
	results, err := store.Retrieve(ctx, RetrieveOptions{
		Query: "test", UserID: "u1", Threshold: 0.9,
	})
	require.NoError(t, err)

	// All results should meet the high threshold
	for _, result := range results {
		assert.GreaterOrEqual(t, result.Score, float32(0.9),
			"Result score %.4f should be >= 0.9 threshold", result.Score)
	}
}

func TestMilvusStore_Retrieve_EmptyResults(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return []client.SearchResult{{ResultCount: 0}}, nil
	}

	results, err := store.Retrieve(ctx, RetrieveOptions{Query: "test", UserID: "u1"})
	require.NoError(t, err)
	assert.Empty(t, results)
}

func TestMilvusStore_RetryLogic_TransientErrors(t *testing.T) {
	store, mockClient := setupTestStore()
	ctx := context.Background()

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return nil, errors.New("connection timeout")
	}

	_, err := store.Retrieve(ctx, RetrieveOptions{Query: "test", UserID: "u1"})
	require.Error(t, err)
	assert.Equal(t, DefaultMaxRetries, mockClient.SearchCallCount)
}

func TestMilvusStore_RetryLogic_ContextCancellation(t *testing.T) {
	store, mockClient := setupTestStore()
	cancelCtx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	mockClient.SearchFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, vectors []entity.Vector, vField string, mType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return nil, errors.New("connection timeout")
	}

	_, err := store.Retrieve(cancelCtx, RetrieveOptions{Query: "test", UserID: "u1"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "context cancelled")
}

func TestIsTransientError(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		transient bool
	}{
		{"connection refused", errors.New("connection refused"), true},
		{"deadline exceeded", errors.New("deadline exceeded"), true},
		{"invalid schema", errors.New("invalid schema"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isTransientError(tt.err)
			assert.Equal(t, tt.transient, result)
		})
	}
}

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

	// Setup mock for Get (Query) - returns existing memory
	queryCount := 0
	mockClient.QueryFunc = func(ctx context.Context, coll string, parts []string, expr string, out []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		queryCount++
		return []entity.Column{
			entity.NewColumnVarChar("id", []string{"mem-123"}),
			entity.NewColumnVarChar("content", []string{"Old content"}),
			entity.NewColumnVarChar("user_id", []string{"user-456"}),
			entity.NewColumnVarChar("memory_type", []string{"semantic"}),
			entity.NewColumnVarChar("metadata", []string{`{}`}),
			entity.NewColumnInt64("created_at", []int64{1704067200}),
			entity.NewColumnInt64("updated_at", []int64{1704067200}),
		}, nil
	}

	// Setup mock for Delete
	mockClient.DeleteFunc = func(ctx context.Context, coll string, part string, expr string) error {
		return nil
	}

	// Setup mock for Insert
	mockClient.InsertFunc = func(ctx context.Context, coll string, part string, cols ...entity.Column) (entity.Column, error) {
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
	assert.Equal(t, 1, mockClient.DeleteCallCount)
	assert.Equal(t, 1, mockClient.InsertCallCount)
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
