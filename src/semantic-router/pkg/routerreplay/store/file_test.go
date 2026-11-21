package store

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
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
	require.NoError(t, store.UpdateLifecycle(ctx, "kept", LifecycleFailed, time.Now(), 7, "duplicate"))
	require.Eventually(t, func() bool { return fileHasLines(path, 2) }, 3*time.Second, 10*time.Millisecond)
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

func fileHasLines(path string, n int) bool {
	data, err := os.ReadFile(path)
	return err == nil && bytes.Count(data, []byte{'\n'}) == n
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

func TestFileStoreKeepsRecordingDuringABlockedWrite(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "router-replay.jsonl")
	store, err := OpenFileStore(path, 3, 0)
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	reader, pipe, err := os.Pipe()
	require.NoError(t, err)
	t.Cleanup(func() { _ = reader.Close() })
	require.NoError(t, store.file.Close())
	store.file = pipe
	deadline := time.Now().Add(5 * time.Second)
	require.NoError(t, reader.SetReadDeadline(deadline))
	require.NoError(t, pipe.SetWriteDeadline(deadline))
	_, err = store.Add(ctx, Record{ID: "blocked", LifecycleState: LifecycleCompleted, RequestBody: strings.Repeat("x", 1<<20)})
	require.NoError(t, err)
	_, err = io.ReadFull(reader, make([]byte, 1))
	require.NoError(t, err)

	for _, id := range []string{"evicted", "a", "b", "c"} {
		_, err = store.Add(ctx, Record{ID: id})
		require.NoError(t, err)
		require.NoError(t, store.UpdateLifecycle(ctx, id, LifecycleCompleted, time.Now(), 1, "done"))
	}
	require.NoError(t, store.AppendOutcome(ctx, "b", Outcome{Verdict: "late"}))
	records, err := store.List(ctx)
	require.NoError(t, err)
	require.Len(t, records, 3)
	require.Equal(t, "c", records[0].ID)
	require.Equal(t, LifecycleCompleted, records[0].LifecycleState)
	require.True(t, time.Now().Before(deadline), "live operations waited for disk I/O")
	require.NoError(t, reader.SetReadDeadline(time.Time{}))
	go func() { _, _ = io.Copy(io.Discard, reader) }()

	require.Eventually(t, func() bool { return fileHasLines(path, 3) }, 5*time.Second, 10*time.Millisecond)
	data, err := os.ReadFile(path)
	require.NoError(t, err)
	for i, line := range bytes.Split(bytes.TrimSpace(data), []byte{'\n'}) {
		var record Record
		require.NoError(t, json.Unmarshal(line, &record))
		require.Equal(t, []string{"a", "b", "c"}[i], record.ID)
		if record.ID == "b" {
			require.Len(t, record.Outcomes, 1)
			require.Equal(t, "late", record.Outcomes[0].Verdict)
		}
	}
	require.NoError(t, store.UpdateLifecycle(ctx, "c", LifecycleFailed, time.Now(), 2, "duplicate"))
	_, err = store.Add(ctx, Record{ID: "d", LifecycleState: LifecycleAborted})
	require.NoError(t, err)
	require.Eventually(t, func() bool {
		data, err := os.ReadFile(path)
		rows := bytes.Split(bytes.TrimSpace(data), []byte{'\n'})
		var last Record
		return err == nil && len(rows) == 4 && json.Unmarshal(rows[3], &last) == nil &&
			last.ID == "d" && last.LifecycleState == LifecycleAborted
	}, 5*time.Second, 10*time.Millisecond)
}

func TestFileStoreRetriesFailedAppendWhileIdle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router-replay.jsonl")
	store, err := OpenFileStore(path, 2, 0)
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	require.NoError(t, store.file.Close())
	store.file, err = os.Open(path)
	require.NoError(t, err)
	_, err = store.Add(context.Background(), Record{ID: "retry", LifecycleState: LifecycleFailed})
	require.NoError(t, err)
	require.Eventually(t, func() bool { return fileHasLines(path, 1) }, 5*time.Second, 10*time.Millisecond)
}

func TestFileStoreReportsFinalRewriteFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router-replay.jsonl")
	store, err := OpenFileStore(path, 2, 0)
	require.NoError(t, err)
	t.Cleanup(func() { _ = store.Close() })
	require.NoError(t, os.Remove(path))
	require.NoError(t, os.Mkdir(path, 0o700))
	require.Error(t, store.Close())
	_, err = store.Add(context.Background(), Record{})
	require.ErrorIs(t, err, ErrStorageClosed)
}
