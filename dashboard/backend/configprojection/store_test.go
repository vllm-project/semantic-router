package configprojection

import (
	"path/filepath"
	"testing"
	"time"
)

func TestRefreshFromCanonicalPersistsActiveProjection(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	store, err := Open(filepath.Join(dir, "projection.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	if refreshErr := store.RefreshFromCanonical(RefreshInput{
		Version:     "20260101-120000",
		Source:      SourceDSL,
		YAMLBytes:   []byte(testCanonicalYAML),
		DSLSnapshot: "ROUTE default-business",
	}); refreshErr != nil {
		t.Fatalf("RefreshFromCanonical: %v", refreshErr)
	}

	active, err := store.GetActiveProjection()
	if err != nil {
		t.Fatalf("GetActiveProjection: %v", err)
	}
	if active.Status != StatusOK {
		t.Fatalf("expected active status ok, got %+v", active)
	}
	if active.ActiveVersion != "20260101-120000" {
		t.Fatalf("unexpected active version: %q", active.ActiveVersion)
	}
	if active.Deployment == nil {
		t.Fatal("expected active deployment payload")
	}
	if active.Deployment.Validation.Status != "ok" {
		t.Fatalf("unexpected validation: %+v", active.Deployment.Validation)
	}

	deployments, err := store.ListDeployments()
	if err != nil {
		t.Fatalf("ListDeployments: %v", err)
	}
	if len(deployments) != 1 || deployments[0].Version != "20260101-120000" {
		t.Fatalf("unexpected deployments: %+v", deployments)
	}
}

func TestOpenInitializesSchemaVersion(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	store, err := Open(filepath.Join(dir, "projection.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	var version int
	if err := store.db.QueryRow(`SELECT version FROM config_projection_schema_version WHERE id = 1`).Scan(&version); err != nil {
		t.Fatalf("read schema version: %v", err)
	}
	if version != currentSchemaVersion {
		t.Fatalf("expected schema version %d, got %d", currentSchemaVersion, version)
	}
}

func TestRefreshFailureMarksStaleWithoutMutatingDeployments(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	store, err := Open(filepath.Join(dir, "projection.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	if seedErr := store.RefreshFromCanonical(RefreshInput{
		Version:   "20260101-120000",
		Source:    SourceManual,
		YAMLBytes: []byte(testCanonicalYAML),
	}); seedErr != nil {
		t.Fatalf("seed projection: %v", seedErr)
	}

	err = store.RefreshFromCanonical(RefreshInput{
		Version:   "20260101-130000",
		Source:    SourceManual,
		YAMLBytes: []byte("routing: ["),
	})
	if err == nil {
		t.Fatal("expected invalid refresh to fail")
	}

	active, getErr := store.GetActiveProjection()
	if getErr != nil {
		t.Fatalf("GetActiveProjection: %v", getErr)
	}
	if active.Status != StatusStale {
		t.Fatalf("expected stale status, got %+v", active)
	}
	if active.ActiveVersion != "20260101-120000" {
		t.Fatalf("expected previous active version to remain, got %q", active.ActiveVersion)
	}
	if active.LastError == "" {
		t.Fatal("expected stale last_error")
	}

	deployments, listErr := store.ListDeployments()
	if listErr != nil {
		t.Fatalf("ListDeployments: %v", listErr)
	}
	if len(deployments) != 1 || deployments[0].Version != "20260101-120000" {
		t.Fatalf("expected only seeded deployment, got %+v", deployments)
	}
}

func TestActiveStatusReadsSeededTimestamp(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	store, err := Open(filepath.Join(dir, "projection.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	// The schema seed is the only row a fresh store carries, so reading the
	// active status right after Open must succeed.
	seeded, err := store.GetActiveProjection()
	if err != nil {
		t.Fatalf("GetActiveProjection on the schema seed: %v", err)
	}
	if seeded.Status != StatusFailed {
		t.Fatalf("expected the seeded failed status, got %+v", seeded)
	}
	if seeded.UpdatedAt.IsZero() {
		t.Fatal("expected a parsed updated_at for the schema seed, got the zero time")
	}

	// Rewrite the seed with SQLite's datetime('now') so the reader's tolerance
	// for that layout is asserted no matter how the writer seeds it later.
	before := time.Now().UTC().Truncate(time.Second)
	if _, execErr := store.db.Exec(
		`UPDATE config_projection_active SET updated_at = datetime('now') WHERE id = 1`,
	); execErr != nil {
		t.Fatalf("write sqlite-layout updated_at: %v", execErr)
	}

	reread, err := store.GetActiveProjection()
	if err != nil {
		t.Fatalf("GetActiveProjection after the sqlite-layout write: %v", err)
	}
	if reread.UpdatedAt.Before(before.Add(-time.Second)) {
		t.Fatalf("updated_at %s predates the write at %s", reread.UpdatedAt, before)
	}
	if reread.UpdatedAt.After(time.Now().UTC().Add(2 * time.Second)) {
		t.Fatalf("updated_at %s is in the future", reread.UpdatedAt)
	}
}
