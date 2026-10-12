package memory

import (
	"context"
	"testing"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/stretchr/testify/require"
)

// A memory id is interpolated into a Milvus boolean-expression string literal,
// so it needs the same escaping as a user id or a project id. Get and Forget
// used to build `id == "<id>"` with a plain %s, which let an id containing a
// quote change what the filter means.
func TestMilvusMemoryIDFiltersEscapeInjection(t *testing.T) {
	const malicious = `x" || user_id != "y`
	const wantFilter = `id == "x\" || user_id != \"y"`

	// captureDeleteExpr runs Forget and returns the expression it handed to Milvus.
	captureDeleteExpr := func(t *testing.T, id string) string {
		t.Helper()
		store, mockClient := setupTestStore()
		var captured string
		mockClient.DeleteFunc = func(_ context.Context, _ string, _ string, expr string) error {
			captured = expr
			return nil
		}
		require.NoError(t, store.Forget(context.Background(), id))
		return captured
	}

	t.Run("Get", func(t *testing.T) {
		store, mockClient := setupTestStore()
		var captured string
		mockClient.QueryFunc = func(_ context.Context, _ string, _ []string, expr string, _ []string, _ ...client.SearchQueryOptionFunc) (client.ResultSet, error) {
			captured = expr
			return client.ResultSet{}, nil
		}

		// The empty result set is irrelevant here, only the filter matters.
		_, _ = store.Get(context.Background(), malicious)

		require.Equal(t, wantFilter, captured)
	})

	t.Run("Forget", func(t *testing.T) {
		require.Equal(t, wantFilter, captureDeleteExpr(t, malicious))
	})

	t.Run("benign id is unchanged", func(t *testing.T) {
		require.Equal(t, `id == "memory-1"`, captureDeleteExpr(t, "memory-1"))
	})
}
