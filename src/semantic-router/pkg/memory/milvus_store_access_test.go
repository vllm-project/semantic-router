package memory

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
	"github.com/stretchr/testify/require"
)

// A retrieval must not read-modify-upsert what it returned: under Bounded
// consistency that write can recreate a memory forgotten in the meantime.
func TestMilvusRetrieveDoesNotWriteBackAccessCounts(t *testing.T) {
	store, mockClient := setupTestStore()
	var queries, upserts atomic.Int32
	mockClient.SearchFunc = func(context.Context, string, []string, string, []string, []entity.Vector, string, entity.MetricType, int, entity.SearchParam, ...client.SearchQueryOptionFunc) ([]client.SearchResult, error) {
		return []client.SearchResult{{
			ResultCount: 1,
			Scores:      []float32{0.9},
			Fields: []entity.Column{
				entity.NewColumnVarChar("id", []string{"forgotten"}),
				entity.NewColumnVarChar("content", []string{"allergic to peanuts"}),
				entity.NewColumnVarChar("memory_type", []string{"semantic"}),
			},
		}}, nil
	}
	mockClient.QueryFunc = func(context.Context, string, []string, string, []string, ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
		queries.Add(1)
		return client.ResultSet{}, nil
	}
	mockClient.UpsertFunc = func(context.Context, string, string, ...entity.Column) (entity.Column, error) {
		upserts.Add(1)
		return nil, nil
	}

	results, err := store.Retrieve(context.Background(), RetrieveOptions{Query: "allergies", UserID: "u1", Threshold: 0.5})
	require.NoError(t, err)
	require.Len(t, results, 1)

	require.Never(t, func() bool { return queries.Load() > 0 || upserts.Load() > 0 },
		200*time.Millisecond, 10*time.Millisecond, "retrieval must not read or upsert in the background")
}
