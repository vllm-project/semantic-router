package vectorstore

import (
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/postgres"
)

// The explicit fixture config points at a disposable integration database.
// Ordinary unit runs do not require a PostgreSQL server.
// StorageIntegration: postgres
func TestEmbeddingIdentityPostgresRegistry(t *testing.T) {
	storagetest.Require(t, "postgres")
	path := os.Getenv("VSR_TEST_POSTGRES_CONFIG")
	if path == "" {
		storagetest.Unavailable(t, "postgres", "set VSR_TEST_POSTGRES_CONFIG for the persistent registry regression")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var cfg postgres.Config
	if err = json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	cfg.TableName = "embedding_identity_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	registry, err := NewPostgresMetadataRegistry(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = registry.db.Exec("DROP TABLE " + cfg.TableName)
		_ = registry.Close()
	})
	testEmbeddingIdentityLifecycle(t, registry)
}
