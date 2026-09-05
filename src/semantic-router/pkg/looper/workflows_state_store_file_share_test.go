package looper

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestFileStateStore_SameDirectoryTTLChangeOnReload(t *testing.T) {
	dir := t.TempDir()
	old := newWorkflowFileToolStateStore(dir, time.Hour)
	defer old.Close()

	ctx := context.Background()
	stale := time.Now().UTC().Add(-2 * time.Second)
	takeState := makeTestState("ttl-take")
	takeState.CreatedAt = stale
	sweepState := makeTestState("ttl-sweep")
	sweepState.CreatedAt = stale
	if _, err := old.Put(ctx, takeState); err != nil {
		t.Fatalf("Put ttl-take: %v", err)
	}
	if _, err := old.Put(ctx, sweepState); err != nil {
		t.Fatalf("Put ttl-sweep: %v", err)
	}

	reloaded := newWorkflowFileToolStateStore(dir, time.Second)
	defer reloaded.Close()
	if old != reloaded {
		t.Fatal("expected same-directory reload to share the store")
	}

	taken, ok, err := reloaded.Take(ctx, "ttl-take")
	if err != nil {
		t.Fatalf("Take: %v", err)
	}
	if ok || taken != nil {
		t.Fatal("Take used the previous generation TTL")
	}

	reloaded.cleanupExpired(time.Now().UTC())
	if _, err := os.Stat(filepath.Join(reloaded.dir, "ttl-sweep.json")); !os.IsNotExist(err) {
		t.Fatalf("sweeper used the previous generation TTL, leftover err=%v", err)
	}
}
