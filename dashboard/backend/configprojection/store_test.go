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

func TestFreshStoreReadsSeededActiveProjection(t *testing.T) {
	t.Parallel()

	store, err := Open(filepath.Join(t.TempDir(), "projection.sqlite"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()

	active, err := store.GetActiveProjection()
	if err != nil {
		t.Fatalf("read fresh active projection: %v", err)
	}
	if active.Status != StatusFailed || active.LastError != "projection not initialized" {
		t.Fatalf("unexpected seed status: %+v", active)
	}
	if active.UpdatedAt.IsZero() {
		t.Fatal("fresh active projection has no timestamp")
	}

	const sqliteTimestamp = "2026-09-23 10:18:22"
	_, err = store.db.Exec(
		`UPDATE config_projection_active SET updated_at = ? WHERE id = 1`,
		sqliteTimestamp,
	)
	if err != nil {
		t.Fatalf("write SQLite timestamp: %v", err)
	}
	active, err = store.GetActiveProjection()
	if err != nil {
		t.Fatalf("read SQLite timestamp: %v", err)
	}
	want := time.Date(2026, 9, 23, 10, 18, 22, 0, time.UTC)
	if !active.UpdatedAt.Equal(want) {
		t.Fatalf("timestamp = %s, want %s", active.UpdatedAt, want)
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
