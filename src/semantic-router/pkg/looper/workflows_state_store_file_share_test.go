package looper

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

	claim, ok, err := reloaded.Claim(ctx, config.DefaultRecipeName, "ttl-take")
	if err != nil {
		t.Fatalf("Claim before TTL commit: %v", err)
	}
	if !ok || claim == nil {
		t.Fatal("candidate TTL applied before reload commit")
	}
	if releaseErr := reloaded.Release(ctx, config.DefaultRecipeName, "ttl-take", claim.Token); releaseErr != nil {
		t.Fatalf("Release: %v", releaseErr)
	}

	reloaded.replaceTTL(time.Second)
	claimed, ok, err := reloaded.Claim(ctx, config.DefaultRecipeName, "ttl-take")
	if err != nil {
		t.Fatalf("Claim after TTL commit: %v", err)
	}
	if ok || claimed != nil {
		t.Fatal("Take used the previous generation TTL")
	}

	reloaded.cleanupExpired(time.Now().UTC())
	namespaced, err := workflowNamespacedStateID(config.DefaultRecipeName, "ttl-sweep")
	if err != nil {
		t.Fatalf("namespace: %v", err)
	}
	if _, err := os.Stat(filepath.Join(reloaded.dir, namespaced+".json")); !os.IsNotExist(err) {
		t.Fatalf("sweeper used the previous generation TTL, leftover err=%v", err)
	}
}

func TestFileStateStore_RegistryRemovesClosedDirectories(t *testing.T) {
	workflowFileStoreRegistryMu.Lock()
	baseline := len(workflowFileStoreRegistry)
	workflowFileStoreRegistryMu.Unlock()

	for i := 0; i < 8; i++ {
		s := newWorkflowFileToolStateStore(filepath.Join(t.TempDir(), "state"), time.Hour)
		if err := s.Close(); err != nil {
			t.Fatalf("Close[%d]: %v", i, err)
		}
	}

	workflowFileStoreRegistryMu.Lock()
	got := len(workflowFileStoreRegistry)
	workflowFileStoreRegistryMu.Unlock()
	if got != baseline {
		t.Fatalf("registry size = %d, want baseline %d", got, baseline)
	}
}

func TestFileStateStore_CloseDoesNotDeleteReplacementRegistration(t *testing.T) {
	dir := t.TempDir()
	old := newWorkflowFileToolStateStore(dir, time.Hour)
	if err := old.Close(); err != nil {
		t.Fatalf("Close old: %v", err)
	}

	replacement := newWorkflowFileToolStateStore(dir, time.Minute)
	t.Cleanup(func() { _ = replacement.Close() })
	if replacement == old {
		t.Fatal("replacement reused a closed store instance")
	}
	if err := old.Close(); err != nil {
		t.Fatalf("second Close old: %v", err)
	}

	ctx := context.Background()
	if _, err := replacement.Put(ctx, makeTestState("still-alive")); err != nil {
		t.Fatalf("Put on replacement after stale Close: %v", err)
	}
	got, ok, err := consumeWorkflowState(replacement, "still-alive")
	if err != nil || !ok || got == nil {
		t.Fatalf("replacement consume: ok=%v err=%v", ok, err)
	}
}
