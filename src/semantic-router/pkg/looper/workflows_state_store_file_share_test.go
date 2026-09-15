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
