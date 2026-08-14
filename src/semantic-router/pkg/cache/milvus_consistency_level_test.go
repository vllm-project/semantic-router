package cache

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolveMilvusConsistencyLevel(t *testing.T) {
	cases := []struct {
		name  string
		input string
		level entity.ConsistencyLevel
		ok    bool
	}{
		{"unset leaves the SDK default", "", entity.ClStrong, false},
		{"strong", "Strong", entity.ClStrong, true},
		{"session lowercase", "session", entity.ClSession, true},
		{"bounded uppercase", "BOUNDED", entity.ClBounded, true},
		{"eventually padded", " Eventually ", entity.ClEventually, true},
		{"unknown falls back to the SDK default", "customized", entity.ClStrong, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			level, ok := resolveMilvusConsistencyLevel(tc.input)
			assert.Equal(t, tc.level, level)
			assert.Equal(t, tc.ok, ok)
		})
	}
}

// appliedConsistencyLevel replays the read options against a sentinel value
// that differs from every mapped level, so "option did nothing" (ClStrong is
// the zero value) cannot masquerade as an explicit Strong.
func appliedConsistencyLevel(t *testing.T, opts []client.SearchQueryOptionFunc) (entity.ConsistencyLevel, bool) {
	t.Helper()
	applied := &client.SearchQueryOption{ConsistencyLevel: entity.ClCustomized}
	for _, opt := range opts {
		opt(applied)
	}
	if len(opts) == 0 {
		return entity.ClCustomized, false
	}
	return applied.ConsistencyLevel, true
}

// milvusCacheWithConsistencyLevel builds a cache whose config carries only the
// consistency-level field relevant to the option helpers.
func milvusCacheWithConsistencyLevel(level string) *MilvusCache {
	return &MilvusCache{config: milvusCacheTestConfig(level)}
}

func TestMilvusCacheSearchQueryOptions(t *testing.T) {
	cases := []struct {
		name      string
		config    string
		wantLevel entity.ConsistencyLevel
		wantSet   bool
	}{
		{"unset keeps the SDK default", "", entity.ClCustomized, false},
		{"strong", "Strong", entity.ClStrong, true},
		{"session lowercase", "session", entity.ClSession, true},
		{"bounded uppercase", "BOUNDED", entity.ClBounded, true},
		{"eventually padded", " Eventually ", entity.ClEventually, true},
		{"unrecognized keeps the SDK default", "customized", entity.ClCustomized, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cache := milvusCacheWithConsistencyLevel(tc.config)
			level, set := appliedConsistencyLevel(t, cache.searchQueryOptions())
			assert.Equal(t, tc.wantSet, set)
			if tc.wantSet {
				assert.Equal(t, tc.wantLevel, level)
			}
		})
	}
}

// recordingMilvusClient stubs the Milvus calls the cache issues so tests can
// assert the consistency-level options actually reach the SDK; the embedded
// nil interface covers every other client method.
type recordingMilvusClient struct {
	client.Client
	searchOpts    []client.SearchQueryOptionFunc
	queryOpts     []client.SearchQueryOptionFunc
	createOpts    []client.CreateCollectionOption
	queryResults  client.ResultSet
	searchResults []client.SearchResult
	created       bool
}

func (f *recordingMilvusClient) Search(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, vectors []entity.Vector, vectorField string, metricType entity.MetricType, topK int, sp entity.SearchParam, opts ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
	f.searchOpts = opts
	return f.searchResults, nil
}

func (f *recordingMilvusClient) Query(ctx context.Context, collectionName string, partitionNames []string, expr string, outputFields []string, opts ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
	f.queryOpts = opts
	return f.queryResults, nil
}

func (f *recordingMilvusClient) CreateCollection(ctx context.Context, schema *entity.Schema, shardsNum int32, opts ...client.CreateCollectionOption) error {
	f.createOpts = opts
	f.created = true
	return nil
}

func (f *recordingMilvusClient) CreateIndex(ctx context.Context, collectionName string, fieldName string, idx entity.Index, async bool, opts ...client.IndexOption) error {
	return nil
}

// milvusCacheTestConfig returns a cache config with the fields the cache
// methods touch during reads and collection creation.
func milvusCacheTestConfig(level string) *config.MilvusConfig {
	cfg := &config.MilvusConfig{}
	cfg.Collection.VectorField.Name = "embedding"
	cfg.Collection.VectorField.Dimension = 384
	cfg.Collection.VectorField.MetricType = "IP"
	cfg.Collection.Index.Params.M = 16
	cfg.Collection.Index.Params.EfConstruction = 64
	cfg.Search.Params.Ef = 8
	cfg.Search.TopK = 5
	cfg.Search.ConsistencyLevel = level
	return cfg
}

func TestMilvusCacheSearchHonorsConsistencyLevel(t *testing.T) {
	fake := &recordingMilvusClient{}
	cache := &MilvusCache{
		enabled:        true,
		client:         fake,
		config:         milvusCacheTestConfig("Strong"),
		collectionName: "test_cache",
	}

	_, err := cache.milvusSearchSimilarVectors(context.Background(), "model", []float32{0.1, 0.2})
	require.NoError(t, err)
	level, set := appliedConsistencyLevel(t, fake.searchOpts)
	assert.True(t, set, "configured level must reach the SDK search call")
	assert.Equal(t, entity.ClStrong, level)
}

func TestMilvusCacheSearchUnsetKeepsSDKDefault(t *testing.T) {
	fake := &recordingMilvusClient{}
	cache := &MilvusCache{
		enabled:        true,
		client:         fake,
		config:         milvusCacheTestConfig(""),
		collectionName: "test_cache",
	}

	_, err := cache.milvusSearchSimilarVectors(context.Background(), "model", []float32{0.1, 0.2})
	require.NoError(t, err)
	assert.Empty(t, fake.searchOpts, "unset level must not pass a consistency option")
}

func TestMilvusCacheGetByIDHonorsConsistencyLevel(t *testing.T) {
	fake := &recordingMilvusClient{queryResults: []entity.Column{
		entity.NewColumnVarChar("response_body", []string{"cached"}),
	}}
	cache := &MilvusCache{
		enabled:        true,
		client:         fake,
		config:         milvusCacheTestConfig("Session"),
		collectionName: "test_cache",
	}

	body, err := cache.GetByID(context.Background(), "req-1", "model")
	require.NoError(t, err)
	assert.Equal(t, []byte("cached"), body)
	level, set := appliedConsistencyLevel(t, fake.queryOpts)
	assert.True(t, set, "configured level must reach the SDK query call")
	assert.Equal(t, entity.ClSession, level)
}

func TestMilvusCacheFindExactHonorsConsistencyLevel(t *testing.T) {
	fake := &recordingMilvusClient{queryResults: []entity.Column{
		entity.NewColumnVarChar("response_body", []string{"exact"}),
	}}
	cache := &MilvusCache{
		enabled:        true,
		client:         fake,
		config:         milvusCacheTestConfig("Eventually"),
		collectionName: "test_cache",
	}

	result, err := cache.FindExact("model", "fingerprint")
	require.NoError(t, err)
	assert.True(t, result.Found)
	level, set := appliedConsistencyLevel(t, fake.queryOpts)
	assert.True(t, set, "configured level must reach the exact-lookup query")
	assert.Equal(t, entity.ClEventually, level)
}

func TestMilvusCacheCreateCollectionConsistencyOptions(t *testing.T) {
	cases := []struct {
		name     string
		config   string
		wantOpts int
	}{
		{"unset keeps the SDK default", "", 0},
		{"configured pins the collection", "Strong", 1},
		{"unrecognized keeps the SDK default", "bogus", 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			fake := &recordingMilvusClient{}
			cache := &MilvusCache{
				enabled:        true,
				client:         fake,
				config:         milvusCacheTestConfig(tc.config),
				collectionName: "test_cache",
			}

			require.NoError(t, cache.createCollection(context.Background()))
			assert.True(t, fake.created)
			assert.Len(t, fake.createOpts, tc.wantOpts)
		})
	}
}
