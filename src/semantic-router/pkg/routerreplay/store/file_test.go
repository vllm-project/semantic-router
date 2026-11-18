package store

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"strconv"
	"syscall"
	"testing"
	"time"
)

// A file store writes a record once, when it ends, and keeps what arrives
// later for its next rewrite, which closing the store makes. Opened again at
// its path, it holds the ended records the last one kept: at most maxRecords,
// none older than the TTL, in a file only its owner can read. A record still
// in progress is not on disk, a line cut short by a crash is skipped, and a
// rewrite's leftover temporary file is removed. The running store lists a
// record past the TTL, as the memory store does, and rewrites drop it.
func TestFileStoreKeepsEndedRecordsAcrossReopen(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), ".vllm-sr", "router-replay.jsonl")
	leftover := filepath.Join(filepath.Dir(path), ".router-replay-12345")
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(leftover, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	open := func() *FileStore {
		t.Helper()
		store, err := OpenFileStore(path, 3, 3600)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = store.Close() })
		return store
	}
	end := func(store *FileStore, record Record) {
		t.Helper()
		if _, err := store.Add(ctx, record); err != nil {
			t.Fatal(err)
		}
		if err := store.UpdateLifecycle(ctx, record.ID, LifecycleCompleted, time.Now(), 5, "done"); err != nil {
			t.Fatal(err)
		}
	}
	store := open()
	end(store, Record{ID: "expired", Timestamp: time.Now().Add(-2 * time.Hour)})
	tokens := 3
	if _, err := store.Add(ctx, Record{ID: "kept", Timestamp: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if err := store.UpdateUsageCost(ctx, "kept", UsageCost{TotalTokens: &tokens}); err != nil {
		t.Fatal(err)
	}
	if err := store.UpdateLifecycle(ctx, "kept", LifecycleCompleted, time.Now(), 5, "done"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendOutcome(ctx, "kept", Outcome{Source: "router", Verdict: "late"}); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Add(ctx, Record{ID: "open", Timestamp: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if lines := fileLines(t, path); lines != 2 {
		t.Fatalf("file holds %d lines, want one for each ended record", lines)
	}
	if _, err := os.Stat(leftover); !os.IsNotExist(err) {
		t.Fatalf("the leftover temporary file is still there: %v", err)
	}
	if _, found, _ := store.Get(ctx, "expired"); !found {
		t.Fatal("the running store does not list a record past the TTL")
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(path); bytes.Contains(data, []byte(`"expired"`)) {
		t.Fatal("the rewrite kept a record past the TTL")
	}
	appendCutLine(t, path)

	store = open()
	if data, _ := os.ReadFile(path); len(data) == 0 || data[len(data)-1] != '\n' {
		t.Fatal("the next record would be appended to the cut line")
	}
	kept, found, err := store.Get(ctx, "kept")
	if err != nil || !found || kept.TotalTokens == nil || *kept.TotalTokens != 3 ||
		kept.LifecycleState != LifecycleCompleted || len(kept.Outcomes) != 1 {
		t.Fatalf("reopened record = %+v, found %v, err %v; want its usage, lifecycle and late outcome", kept, found, err)
	}
	for _, id := range []string{"expired", "open"} {
		if _, found, _ = store.Get(ctx, id); found {
			t.Fatalf("record %s survived the reopen", id)
		}
	}
	for _, id := range []string{"b", "c", "d"} {
		end(store, Record{ID: id, Timestamp: time.Now()})
	}
	if err = store.Close(); err != nil {
		t.Fatal(err)
	}

	store = open()
	records, err := store.List(ctx)
	if err != nil || len(records) != 3 || records[0].ID != "d" || records[2].ID != "b" {
		t.Fatalf("reopened records = %+v, err %v; want the newest three, d to b", records, err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("file mode = %v, want 0600", info.Mode().Perm())
	}
}

func fileLines(t *testing.T, path string) int {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return bytes.Count(data, []byte{'\n'})
}

func appendCutLine(t *testing.T, path string) {
	t.Helper()
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = file.WriteString(`{"id":"cut","timest`); err != nil {
		t.Fatal(err)
	}
	if err = file.Close(); err != nil {
		t.Fatal(err)
	}
}

// The Router generations of a configuration reload share one store: the new
// generation holds what the old one records until it is retired, and the old
// one closing leaves the store open for the new.
func TestFileStoreIsSharedAcrossRouterGenerations(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router-replay.jsonl")
	old, err := OpenFileStore(path, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	next, err := OpenFileStore(path, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = next.Close() })
	if _, err = old.Add(ctx, Record{ID: "during-reload", Timestamp: time.Now()}); err != nil {
		t.Fatal(err)
	}
	if err = old.Close(); err != nil {
		t.Fatal(err)
	}
	if _, found, _ := next.Get(ctx, "during-reload"); !found {
		t.Fatal("the new generation does not hold a record the old one wrote before it retired")
	}
	if _, err = next.Add(ctx, Record{ID: "after-reload", Timestamp: time.Now()}); err != nil {
		t.Fatalf("closing the old generation closed the store: %v", err)
	}
}

// A second Router process on the same file is refused, so that neither
// rewrites the file under the other.
func TestFileStoreRefusesAFileAnotherProcessHolds(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router-replay.jsonl")
	lock, err := os.OpenFile(path+".lock", os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	if err = syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		t.Fatal(err)
	}
	if store, openErr := OpenFileStore(path, 10, 0); openErr == nil {
		_ = store.Close()
		t.Fatal("opened a file another process holds")
	}
}

// A rewrite that fails is tried again only after maxRecords more lines, not
// at every record's end.
func TestFileStoreWaitsAfterAFailedRewrite(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router-replay.jsonl")
	store, err := OpenFileStore(path, 2, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	// A directory at the path fails every rename onto it, for root as well.
	if err = os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if err = os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 5; i++ {
		if _, err = store.Add(ctx, Record{ID: strconv.Itoa(i), Timestamp: time.Now(), LifecycleState: LifecycleCompleted}); err != nil {
			t.Fatal(err)
		}
	}
	if store.lines != 3 {
		t.Fatalf("line count = %d after a failed rewrite at 4 and one more line, want 3", store.lines)
	}
}
