package memory

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/qdrant/go-client/qdrant"
	"github.com/stretchr/testify/require"
)

// Every memory belongs to a user, and Milvus and Valkey both refuse a Retrieve
// with an empty user id. Qdrant used to accept it and quietly search for
// memories whose owner is the empty string.
func TestQdrantRetrieveRejectsEmptyUserID(t *testing.T) {
	qdrantClient, err := qdrant.NewClient(&qdrant.Config{Host: "127.0.0.1", Port: 1})
	require.NoError(t, err)
	t.Cleanup(func() { _ = qdrantClient.Close() })

	store := &QdrantStore{
		client:         qdrantClient,
		collectionName: "agentic_memory",
		enabled:        true,
	}

	errorsBefore := testutil.ToFloat64(MemoryRetrievalCount.WithLabelValues("qdrant", "error"))

	// A non-empty query keeps the failure unambiguous: only the missing user id
	// can produce this error, and it must be returned before the store reaches
	// the embedding provider or the Qdrant client.
	_, err = store.Retrieve(context.Background(), RetrieveOptions{Query: "anything", UserID: ""})

	require.EqualError(t, err, "user id is required")
	require.Equal(t, errorsBefore+1, testutil.ToFloat64(MemoryRetrievalCount.WithLabelValues("qdrant", "error")),
		"a rejected retrieve must be counted as a retrieval error, as the other backends do")
}
