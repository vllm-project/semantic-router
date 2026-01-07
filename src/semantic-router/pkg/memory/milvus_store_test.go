//go:build !windows && cgo

package memory

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/milvus-io/milvus-proto/go-api/v2/msgpb"
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
	SearchCallCount   int
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
func (m *MockMilvusClient) Close() error { return nil }
func (m *MockMilvusClient) CheckHealth(context.Context) (*entity.MilvusState, error) { return nil, nil }
func (m *MockMilvusClient) UsingDatabase(context.Context, string) error { return nil }
func (m *MockMilvusClient) ListDatabases(context.Context) ([]entity.Database, error) { return nil, nil }
func (m *MockMilvusClient) CreateDatabase(context.Context, string, ...client.CreateDatabaseOption) error { return nil }
func (m *MockMilvusClient) DropDatabase(context.Context, string, ...client.DropDatabaseOption) error { return nil }
func (m *MockMilvusClient) AlterDatabase(context.Context, string, ...entity.DatabaseAttribute) error { return nil }
func (m *MockMilvusClient) DescribeDatabase(context.Context, string) (*entity.Database, error) { return nil, nil }
func (m *MockMilvusClient) NewCollection(context.Context, string, int64, ...client.CreateCollectionOption) error { return nil }
func (m *MockMilvusClient) ListCollections(context.Context, ...client.ListCollectionOption) ([]*entity.Collection, error) { return nil, nil }
func (m *MockMilvusClient) CreateCollection(context.Context, *entity.Schema, int32, ...client.CreateCollectionOption) error { return nil }
func (m *MockMilvusClient) DescribeCollection(context.Context, string) (*entity.Collection, error) { return nil, nil }
func (m *MockMilvusClient) DropCollection(context.Context, string, ...client.DropCollectionOption) error { return nil }
func (m *MockMilvusClient) GetCollectionStatistics(context.Context, string) (map[string]string, error) { return nil, nil }
func (m *MockMilvusClient) LoadCollection(context.Context, string, bool, ...client.LoadCollectionOption) error { return nil }
func (m *MockMilvusClient) ReleaseCollection(context.Context, string, ...client.ReleaseCollectionOption) error { return nil }
func (m *MockMilvusClient) RenameCollection(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) AlterCollection(context.Context, string, ...entity.CollectionAttribute) error { return nil }
func (m *MockMilvusClient) CreateAlias(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) DropAlias(context.Context, string) error { return nil }
func (m *MockMilvusClient) AlterAlias(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) GetReplicas(context.Context, string) ([]*entity.ReplicaGroup, error) { return nil, nil }
func (m *MockMilvusClient) BackupRBAC(context.Context) (*entity.RBACMeta, error) { return nil, nil }
func (m *MockMilvusClient) RestoreRBAC(context.Context, *entity.RBACMeta) error { return nil }
func (m *MockMilvusClient) CreateCredential(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) UpdateCredential(context.Context, string, string, string) error { return nil }
func (m *MockMilvusClient) DeleteCredential(context.Context, string) error { return nil }
func (m *MockMilvusClient) ListCredUsers(context.Context) ([]string, error) { return nil, nil }
func (m *MockMilvusClient) CreateRole(context.Context, string) error { return nil }
func (m *MockMilvusClient) DropRole(context.Context, string) error { return nil }
func (m *MockMilvusClient) AddUserRole(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) RemoveUserRole(context.Context, string, string) error { return nil }
func (m *MockMilvusClient) ListRoles(context.Context) ([]entity.Role, error) { return nil, nil }
func (m *MockMilvusClient) ListUsers(context.Context) ([]entity.User, error) { return nil, nil }
func (m *MockMilvusClient) Grant(context.Context, string, entity.PriviledgeObjectType, string, string, ...entity.OperatePrivilegeOption) error { return nil }
func (m *MockMilvusClient) Revoke(context.Context, string, entity.PriviledgeObjectType, string, string, ...entity.OperatePrivilegeOption) error { return nil }
func (m *MockMilvusClient) ListGrant(context.Context, string, string, string, string) ([]entity.RoleGrants, error) { return nil, nil }
func (m *MockMilvusClient) ListGrants(context.Context, string, string) ([]entity.RoleGrants, error) { return nil, nil }
func (m *MockMilvusClient) CreatePartition(context.Context, string, string, ...client.CreatePartitionOption) error { return nil }
func (m *MockMilvusClient) DropPartition(context.Context, string, string, ...client.DropPartitionOption) error { return nil }
func (m *MockMilvusClient) ShowPartitions(context.Context, string) ([]*entity.Partition, error) { return nil, nil }
func (m *MockMilvusClient) HasPartition(context.Context, string, string) (bool, error) { return false, nil }
func (m *MockMilvusClient) LoadPartitions(context.Context, string, []string, bool, ...client.LoadPartitionsOption) error { return nil }
func (m *MockMilvusClient) ReleasePartitions(context.Context, string, []string, ...client.ReleasePartitionsOption) error { return nil }
func (m *MockMilvusClient) GetPersistentSegmentInfo(context.Context, string) ([]*entity.Segment, error) { return nil, nil }
func (m *MockMilvusClient) CreateIndex(context.Context, string, string, entity.Index, bool, ...client.IndexOption) error { return nil }
func (m *MockMilvusClient) DescribeIndex(context.Context, string, string, ...client.IndexOption) ([]entity.Index, error) { return nil, nil }
func (m *MockMilvusClient) DropIndex(context.Context, string, string, ...client.IndexOption) error { return nil }
func (m *MockMilvusClient) GetIndexState(context.Context, string, string, ...client.IndexOption) (entity.IndexState, error) { return 0, nil }
func (m *MockMilvusClient) AlterIndex(context.Context, string, string, ...client.IndexOption) error { return nil }
func (m *MockMilvusClient) GetIndexBuildProgress(context.Context, string, string, ...client.IndexOption) (int64, int64, error) { return 0, 0, nil }
func (m *MockMilvusClient) Insert(context.Context, string, string, ...entity.Column) (entity.Column, error) { return nil, nil }
func (m *MockMilvusClient) Flush(context.Context, string, bool, ...client.FlushOption) error { return nil }
func (m *MockMilvusClient) FlushV2(context.Context, string, bool, ...client.FlushOption) ([]int64, []int64, int64, map[string]msgpb.MsgPosition, error) { return nil, nil, 0, make(map[string]msgpb.MsgPosition), nil }
func (m *MockMilvusClient) DeleteByPks(context.Context, string, string, entity.Column) error { return nil }
func (m *MockMilvusClient) Delete(context.Context, string, string, string) error { return nil }
func (m *MockMilvusClient) Upsert(context.Context, string, string, ...entity.Column) (entity.Column, error) { return nil, nil }
func (m *MockMilvusClient) QueryByPks(context.Context, string, []string, entity.Column, []string, ...client.SearchQueryOptionFunc) (client.ResultSet, error) { return nil, nil }
func (m *MockMilvusClient) Query(context.Context, string, []string, string, []string, ...client.SearchQueryOptionFunc) (client.ResultSet, error) { return nil, nil }
func (m *MockMilvusClient) Get(context.Context, string, entity.Column, ...client.GetOption) (client.ResultSet, error) { return nil, nil }
func (m *MockMilvusClient) QueryIterator(context.Context, *client.QueryIteratorOption) (*client.QueryIterator, error) { return nil, nil }
func (m *MockMilvusClient) CalcDistance(context.Context, string, []string, entity.MetricType, entity.Column, entity.Column) (entity.Column, error) { return nil, nil }
func (m *MockMilvusClient) CreateCollectionByRow(context.Context, entity.Row, int32) error { return nil }
func (m *MockMilvusClient) InsertByRows(context.Context, string, string, []entity.Row) (entity.Column, error) { return nil, nil }
func (m *MockMilvusClient) InsertRows(context.Context, string, string, []interface{}) (entity.Column, error) { return nil, nil }
func (m *MockMilvusClient) ManualCompaction(context.Context, string, time.Duration) (int64, error) { return 0, nil }
func (m *MockMilvusClient) GetCompactionState(context.Context, int64) (entity.CompactionState, error) { return 0, nil }
func (m *MockMilvusClient) GetCompactionStateWithPlans(context.Context, int64) (entity.CompactionState, []entity.CompactionPlan, error) { return 0, nil, nil }
func (m *MockMilvusClient) BulkInsert(context.Context, string, string, []string, ...client.BulkInsertOption) (int64, error) { return 0, nil }
func (m *MockMilvusClient) GetBulkInsertState(context.Context, int64) (*entity.BulkInsertTaskState, error) { return nil, nil }
func (m *MockMilvusClient) ListBulkInsertTasks(context.Context, string, int64) ([]*entity.BulkInsertTaskState, error) { return nil, nil }
func (m *MockMilvusClient) CreateResourceGroup(context.Context, string, ...client.CreateResourceGroupOption) error { return nil }
func (m *MockMilvusClient) UpdateResourceGroups(context.Context, ...client.UpdateResourceGroupsOption) error { return nil }
func (m *MockMilvusClient) DropResourceGroup(context.Context, string) error { return nil }
func (m *MockMilvusClient) DescribeResourceGroup(context.Context, string) (*entity.ResourceGroup, error) { return nil, nil }
func (m *MockMilvusClient) ListResourceGroups(context.Context) ([]string, error) { return nil, nil }
func (m *MockMilvusClient) TransferNode(context.Context, string, string, int32) error { return nil }
func (m *MockMilvusClient) TransferReplica(context.Context, string, string, string, int64) error { return nil }
func (m *MockMilvusClient) DescribeUser(context.Context, string) (entity.UserDescription, error) { return entity.UserDescription{}, nil }
func (m *MockMilvusClient) DescribeUsers(context.Context) ([]entity.UserDescription, error) { return nil, nil }
func (m *MockMilvusClient) GetLoadingProgress(context.Context, string, []string) (int64, error) { return 0, nil }
func (m *MockMilvusClient) GetLoadState(context.Context, string, []string) (entity.LoadState, error) { return 0, nil }
func (m *MockMilvusClient) GetVersion(context.Context) (string, error) { return "", nil }
func (m *MockMilvusClient) HybridSearch(context.Context, string, []string, int, []string, client.Reranker, []*client.ANNSearchRequest, ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) { return nil, nil }
func (m *MockMilvusClient) ReplicateMessage(context.Context, string, uint64, uint64, [][]byte, []*msgpb.MsgPosition, []*msgpb.MsgPosition, ...client.ReplicateMessageOption) (*entity.MessageInfo, error) { return nil, nil }

func setupTestStore() (*MilvusStore, *MockMilvusClient) {
	mockClient := &MockMilvusClient{}
	options := MilvusStoreOptions{
		Client:         mockClient,
		CollectionName: "test_memories",
		Config:         DefaultMemoryConfig(),
		Enabled:        true,
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

	// Verify Metadata Inflation
	assert.Equal(t, "slack", results[0].Metadata["source"])
	assert.Equal(t, "high", results[0].Metadata["importance"])
	assert.Contains(t, results[0].Metadata["_raw_source"], "slack")
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
	assert.Equal(t, "id1", results[0].ID)
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
