package maskingregistry

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/postgres"
)

// StorageIntegration: postgres
func TestPostgresRegistry(t *testing.T) {
	storagetest.Require(t, "postgres")
	path := os.Getenv("VSR_TEST_POSTGRES_CONFIG")
	if path == "" {
		storagetest.Unavailable(t, "postgres", "set VSR_TEST_POSTGRES_CONFIG to a disposable database config file")
	}
	raw, e := os.ReadFile(path)
	if e != nil {
		t.Fatal(e)
	}
	var cfg postgres.Config
	if e = json.Unmarshal(raw, &cfg); e != nil {
		t.Fatal(e)
	}
	name := "test-" + uuid.NewString()
	s, e := openPostgres(&cfg, name, "key")
	if e != nil {
		t.Fatal(e)
	}
	p := s.(*Postgres)
	defer p.db.Close()
	defer func() {
		for _, statement := range []string{
			"DELETE FROM vsr_masking_indices WHERE namespace=$1",
			"DELETE FROM vsr_masking_counters WHERE namespace=$1",
			"DELETE FROM vsr_masking_templates WHERE namespace=$1",
			"DELETE FROM vsr_masking_namespaces WHERE namespace=$1",
		} {
			_, _ = p.db.ExecContext(context.Background(), statement, name)
		}
	}()
	testStore(t, p)
	// Two independent pools exercise the same counter lock, not a Go mutex.
	second, e := openPostgres(&cfg, name, "key")
	if e != nil {
		t.Fatal(e)
	}
	defer second.(*Postgres).db.Close()
	i, e := second.Allocate("scope", "pii:PERSON", "digest")
	if e != nil || i != 1 {
		t.Fatalf("second pool: %d %v", i, e)
	}
	if _, e = openPostgres(&cfg, name, "wrong"); e == nil {
		t.Fatal("mismatched postgres key accepted")
	}
}
