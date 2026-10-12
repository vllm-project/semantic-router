package configsnapshot

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHistoryKeepsTheNewestRecordsAndTheActiveVersion(t *testing.T) {
	h, err := NewHistory(2, nil)
	if err != nil {
		t.Fatal(err)
	}
	at := time.Date(2026, 10, 6, 12, 0, 0, 0, time.UTC)
	active, err := h.Add(Record{Version: 3, Hash: "h3", RecordedAt: at})
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		if _, err := h.Add(Record{Hash: "legacy", RecordedAt: at.Add(time.Duration(i+1) * time.Second)}); err != nil {
			t.Fatal(err)
		}
	}
	records := h.List()
	if len(records) != 2 || records[1].ID != active.ID || records[0].Version != 0 {
		t.Fatalf("records = %+v, want the newest backup and the active version", records)
	}
	if latest, ok := h.Latest(); !ok || latest.Version != 3 {
		t.Fatalf("Latest() = %+v, %v", latest, ok)
	}
	if _, ok := h.ByVersion(3); !ok {
		t.Fatal("the active version was trimmed")
	}
	if _, err := NewHistory(0, nil); err == nil {
		t.Fatal("NewHistory accepted a limit of 0")
	}
}

func TestHistoryRecordIDsAreUniqueWithinASecond(t *testing.T) {
	h, _ := NewHistory(5, nil)
	at := time.Date(2026, 10, 6, 12, 0, 0, 0, time.Local)
	first, _ := h.Add(Record{Version: 1, RecordedAt: at})
	second, _ := h.Add(Record{Version: 2, RecordedAt: at})
	if first.ID != "20261006-120000" || second.ID != "20261006-120000-001" {
		t.Fatalf("IDs = %q, %q", first.ID, second.ID)
	}
}

func TestDirStoreKeepsTheBackupLayoutAndReadsEveryWritersRecords(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "config-backups")
	store := NewDirStore(dir)
	at := time.Date(2026, 10, 6, 12, 30, 0, 0, time.Local)
	saved, saveErr := store.Save(Record{
		ID: "20261006-123000", Version: 4, Hash: "h4", RecordedAt: at, Document: []byte("version: v0.3\n"),
		Origin: Origin{Source: SourceAPI, Principal: "admin", RequestID: "req-1"},
	})
	if saveErr != nil {
		t.Fatalf("Save() error = %v", saveErr)
	}
	if saved.ID != "20261006-123000" {
		t.Fatalf("ID = %q", saved.ID)
	}
	if info, err := os.Stat(dir); err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("history directory mode = %v, %v", info.Mode().Perm(), err)
	}
	for _, suffix := range []string{".yaml", ".source", ".snapshot.json"} {
		info, err := os.Stat(filepath.Join(dir, "config."+saved.ID+suffix))
		if err != nil || info.Mode().Perm() != 0o600 {
			t.Fatalf("%s: %v %v", suffix, info, err)
		}
	}
	if source, _ := os.ReadFile(filepath.Join(dir, "config."+saved.ID+".source")); string(source) != "api\n" {
		t.Fatalf("source file = %q", source)
	}

	// A backup the management API wrote before snapshots, and one the
	// Dashboard writes, which uses another ID layout and no source file.
	writeLegacy := func(id, document, source string) {
		if err := os.WriteFile(filepath.Join(dir, "config."+id+".yaml"), []byte(document), 0o600); err != nil {
			t.Fatal(err)
		}
		if source != "" {
			if err := os.WriteFile(filepath.Join(dir, "config."+id+".source"), []byte(source+"\n"), 0o600); err != nil {
				t.Fatal(err)
			}
		}
	}
	writeLegacy("20261006-110000", "legacy: api\n", "rollback")
	writeLegacy("20261006-113000.123456789", "legacy: dashboard\n", "")
	if _, err := store.Save(Record{
		ID: "20261006-123100", Version: 5, Hash: "h5", RecordedAt: at.Add(time.Minute), Origin: Origin{Source: SourceKubernetes},
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Save(Record{ID: "../escape"}); err == nil {
		t.Fatal("Save() accepted an ID outside the record layout")
	}

	records, err := store.Load()
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	byID := make(map[string]Record)
	for _, r := range records {
		byID[r.ID] = r
	}
	if got := byID["20261006-123000"]; got.Version != 4 || got.Origin.Principal != "admin" || string(got.Document) != "version: v0.3\n" {
		t.Fatalf("snapshot record = %+v", got)
	}
	if got := byID["20261006-110000"]; got.Version != 0 || got.Origin.Source != SourceRollback ||
		got.Hash != documentDigest([]byte("legacy: api\n")) || got.RecordedAt.IsZero() {
		t.Fatalf("API backup record = %+v", got)
	}
	if got := byID["20261006-113000.123456789"]; got.Version != 0 || got.Origin.Source != "" || got.Document == nil {
		t.Fatalf("Dashboard backup record = %+v", got)
	}
	if got := byID["20261006-123100"]; got.Version != 5 || got.Document != nil {
		t.Fatalf("metadata-only record = %+v", got)
	}

	if err := store.Remove(saved.ID); err != nil {
		t.Fatal(err)
	}
	if matches, _ := filepath.Glob(filepath.Join(dir, "config."+saved.ID+".*")); len(matches) != 0 {
		t.Fatalf("Remove() left %v", matches)
	}
}

func TestDirStoreNeverReplacesAnotherWritersRecord(t *testing.T) {
	dir := t.TempDir()
	at := time.Date(2026, 10, 6, 12, 30, 0, 0, time.Local)
	if err := os.WriteFile(filepath.Join(dir, "config.20261006-123000.yaml"), []byte("theirs"), 0o600); err != nil {
		t.Fatal(err)
	}
	saved, err := NewDirStore(dir).Save(Record{ID: "20261006-123000", Version: 1, RecordedAt: at, Document: []byte("ours")})
	if err != nil {
		t.Fatal(err)
	}
	if saved.ID != "20261006-123000-001" {
		t.Fatalf("ID = %q", saved.ID)
	}
	if theirs, _ := os.ReadFile(filepath.Join(dir, "config.20261006-123000.yaml")); string(theirs) != "theirs" {
		t.Fatal("Save() replaced another writer's record")
	}
}

func documentUpdate(source Source, document string) Update {
	cfg := &config.RouterConfig{DocumentHash: documentDigest([]byte(document))}
	return Update{Origin: Origin{Source: source}, Config: cfg, Document: []byte(document)}
}

func persistentManager(t *testing.T, dir string, limit int) *Manager {
	t.Helper()
	history, err := NewHistory(limit, NewDirStore(dir))
	if err != nil {
		t.Fatalf("NewHistory() error = %v", err)
	}
	return NewManager(Options{Runtime: &fakeRuntime{}, History: history})
}

func TestRestartRecoversTheHistoryAndContinuesVersions(t *testing.T) {
	dir := t.TempDir()
	first := persistentManager(t, dir, 10)
	if _, err := first.Install(context.Background(), documentUpdate(SourceStartup, "a")); err != nil {
		t.Fatal(err)
	}
	for _, document := range []string{"b", "c"} {
		if _, err := first.Apply(context.Background(), documentUpdate(SourceFile, document)); err != nil {
			t.Fatal(err)
		}
	}

	restarted := persistentManager(t, dir, 10)
	snapshot, err := restarted.Install(context.Background(), documentUpdate(SourceStartup, "c"))
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Version() != 3 || len(restarted.History().List()) != 3 {
		t.Fatalf("restart on the newest document: v%d, %d records; want v3 and no new record",
			snapshot.Version(), len(restarted.History().List()))
	}
	record, ok := restarted.History().ByVersion(1)
	if !ok || string(record.Document) != "a" || record.Origin.Source != SourceStartup {
		t.Fatalf("recovered version 1 = %+v, %v", record, ok)
	}

	rollback := documentUpdate(SourceRollback, "a")
	rollback.Origin.RollbackOf = 1
	snapshot, err = restarted.Apply(context.Background(), rollback)
	if err != nil {
		t.Fatal(err)
	}
	if snapshot.Version() != 4 || snapshot.Origin().RollbackOf != 1 {
		t.Fatalf("rollback = v%d %+v, want a new version 4 that names version 1", snapshot.Version(), snapshot.Origin())
	}

	again := persistentManager(t, dir, 10)
	snapshot, err = again.Install(context.Background(), documentUpdate(SourceStartup, "edited while down"))
	if err != nil || snapshot.Version() != 5 {
		t.Fatalf("restart on another document = v%d, %v; want v5", snapshot.Version(), err)
	}
}

func TestHistoryLimitBoundsTheDirectory(t *testing.T) {
	dir := t.TempDir()
	m := persistentManager(t, dir, 3)
	if _, err := m.Install(context.Background(), documentUpdate(SourceStartup, "v1")); err != nil {
		t.Fatal(err)
	}
	for i := 2; i <= 6; i++ {
		if _, err := m.Apply(context.Background(), documentUpdate(SourceFile, "v"+strings.Repeat("x", i))); err != nil {
			t.Fatal(err)
		}
	}
	records := m.History().List()
	if len(records) != 3 || records[0].Version != 6 || records[1].Version != 5 || records[2].Version != 4 {
		t.Fatalf("records = %+v", records)
	}
	if records[0].ID <= records[1].ID || records[1].ID <= records[2].ID {
		t.Fatalf("IDs %s, %s, %s do not follow the write order: a trimmed ID was reused",
			records[0].ID, records[1].ID, records[2].ID)
	}
	documents, _ := filepath.Glob(filepath.Join(dir, "config.*.yaml"))
	sidecars, _ := filepath.Glob(filepath.Join(dir, "config.*.snapshot.json"))
	if len(documents) != 3 || len(sidecars) != 3 {
		t.Fatalf("directory holds %d documents and %d sidecars, want 3 each", len(documents), len(sidecars))
	}
}

func TestAttributionNamesWhoCausedAFileUpdate(t *testing.T) {
	m := NewManager(Options{Runtime: &fakeRuntime{}})
	if _, err := m.Install(context.Background(), documentUpdate(SourceStartup, "a")); err != nil {
		t.Fatal(err)
	}
	api := Origin{Source: SourceAPI, Principal: "operator", RequestID: "req-7"}
	m.Attribute(documentDigest([]byte("b")), api)
	snapshot, err := m.Apply(context.Background(), documentUpdate(SourceFile, "b"))
	if err != nil || snapshot.Origin() != api {
		t.Fatalf("attributed update origin = %+v, %v", snapshot.Origin(), err)
	}
	if record, _ := m.History().ByVersion(snapshot.Version()); record.Origin != api {
		t.Fatalf("history origin = %+v", record.Origin)
	}
	snapshot, _ = m.Apply(context.Background(), documentUpdate(SourceFile, "b"))
	if snapshot.Origin().Source != SourceFile {
		t.Fatal("an attribution applied twice")
	}

	withdraw := m.Attribute(documentDigest([]byte("c")), api)
	withdraw()
	snapshot, _ = m.Apply(context.Background(), documentUpdate(SourceFile, "c"))
	if snapshot.Origin().Source != SourceFile {
		t.Fatal("a withdrawn attribution applied")
	}
	m.Attribute(documentDigest([]byte("d")), api)
	snapshot, _ = m.Apply(context.Background(), documentUpdate(SourceKubernetes, "d"))
	if snapshot.Origin().Source != SourceKubernetes {
		t.Fatal("an attribution applied to another source")
	}
}
