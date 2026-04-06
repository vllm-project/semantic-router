package milvuslifecycle

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type mockLifecycleClient struct {
	hasCollection       bool
	hasCollectionErr    error
	createCollectionErr error
	dropCollectionErr   error
	createIndexErr      error
	describeIndex       map[string][]entity.Index
	describeIndexErr    error
	loadCollectionErr   error

	createCollectionCalls int
	dropCollectionCalls   int
	loadCollectionCalls   int
	createdIndexes        []string
}

func (m *mockLifecycleClient) HasCollection(context.Context, string) (bool, error) {
	return m.hasCollection, m.hasCollectionErr
}

func (m *mockLifecycleClient) DropCollection(context.Context, string, ...client.DropCollectionOption) error {
	m.dropCollectionCalls++
	m.hasCollection = false
	return m.dropCollectionErr
}

func (m *mockLifecycleClient) CreateCollection(context.Context, *entity.Schema, int32, ...client.CreateCollectionOption) error {
	m.createCollectionCalls++
	m.hasCollection = true
	return m.createCollectionErr
}

func (m *mockLifecycleClient) CreateIndex(
	_ context.Context, _ string, fieldName string, _ entity.Index, _ bool, _ ...client.IndexOption,
) error {
	if m.createIndexErr != nil {
		return m.createIndexErr
	}
	m.createdIndexes = append(m.createdIndexes, fieldName)
	return nil
}

func (m *mockLifecycleClient) DescribeIndex(
	_ context.Context, _ string, fieldName string, _ ...client.IndexOption,
) ([]entity.Index, error) {
	if m.describeIndexErr != nil {
		return nil, m.describeIndexErr
	}
	if m.describeIndex == nil {
		return nil, nil
	}
	return m.describeIndex[fieldName], nil
}

func (m *mockLifecycleClient) LoadCollection(context.Context, string, bool, ...client.LoadCollectionOption) error {
	m.loadCollectionCalls++
	return m.loadCollectionErr
}

func TestEnsureCollectionCreatesMissingCollection(t *testing.T) {
	mockClient := &mockLifecycleClient{}

	result, err := EnsureCollection(context.Background(), mockClient, CollectionSpec{
		Name:     "test_collection",
		Schema:   testSchema("test_collection"),
		ShardNum: 1,
		Load:     true,
		Indexes: []IndexSpec{
			{
				FieldName: "embedding",
				Build: func() (entity.Index, error) {
					return entity.NewIndexAUTOINDEX(entity.IP)
				},
			},
		},
	}, EnsureOptions{AllowCreate: true})
	require.NoError(t, err)
	assert.True(t, result.Created)
	assert.False(t, result.Existed)
	assert.Equal(t, 1, mockClient.createCollectionCalls)
	assert.Equal(t, []string{"embedding"}, mockClient.createdIndexes)
	assert.Equal(t, 1, mockClient.loadCollectionCalls)
}

func TestEnsureCollectionReusesExistingCollectionAndBackfillsIndexes(t *testing.T) {
	mockClient := &mockLifecycleClient{
		hasCollection: true,
		describeIndex: map[string][]entity.Index{
			"vector": nil,
		},
	}

	result, err := EnsureCollection(context.Background(), mockClient, CollectionSpec{
		Name:   "existing_collection",
		Schema: testSchema("existing_collection"),
		Load:   true,
		Indexes: []IndexSpec{
			{
				FieldName:        "vector",
				EnsureOnExisting: true,
				Build: func() (entity.Index, error) {
					return entity.NewIndexAUTOINDEX(entity.L2)
				},
			},
			{
				FieldName: "timestamp",
				Build: func() (entity.Index, error) {
					return entity.NewGenericIndex("timestamp_idx", entity.Sorted, map[string]string{}), nil
				},
			},
		},
	}, EnsureOptions{AllowCreate: true})
	require.NoError(t, err)
	assert.False(t, result.Created)
	assert.True(t, result.Existed)
	assert.Equal(t, 0, mockClient.createCollectionCalls)
	assert.Equal(t, []string{"vector"}, mockClient.createdIndexes)
	assert.Equal(t, 1, mockClient.loadCollectionCalls)
}

func TestEnsureCollectionRespectsAutoCreateDisabled(t *testing.T) {
	mockClient := &mockLifecycleClient{}

	_, err := EnsureCollection(context.Background(), mockClient, CollectionSpec{
		Name:   "missing_collection",
		Schema: testSchema("missing_collection"),
	}, EnsureOptions{AllowCreate: false})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "auto-creation is disabled")
}

func TestRetryHonorsSharedBackoffPolicy(t *testing.T) {
	attempts := 0
	err := Retry(context.Background(), RetryPolicy{
		Attempts: 3,
		Backoff: func(int) time.Duration {
			return 0
		},
		ShouldRetry: func(err error) bool {
			return errors.Is(err, errTransient)
		},
	}, func() error {
		attempts++
		if attempts < 3 {
			return errTransient
		}
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 3, attempts)
}

var errTransient = errors.New("transient")

func testSchema(name string) *entity.Schema {
	return &entity.Schema{
		CollectionName: name,
		Fields: []*entity.Field{
			{
				Name:       "id",
				DataType:   entity.FieldTypeVarChar,
				PrimaryKey: true,
				TypeParams: map[string]string{"max_length": "64"},
			},
		},
	}
}
