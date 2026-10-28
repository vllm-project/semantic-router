package extproc

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestPollConfigFileChangesAcrossAtomicReplacement(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.yaml")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}

	ticks := make(chan time.Time)
	changes := make(chan struct{}, 2)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		pollConfigFileChanges(ctx, path, ticks, func() { changes <- struct{}{} })
	}()
	defer func() {
		cancel()
		<-done
	}()

	// The first tick cannot be received until the initial file is sampled.
	ticks <- time.Now()
	if err := os.WriteFile(path, []byte("after!"), 0o600); err != nil {
		t.Fatal(err)
	}
	ticks <- time.Now()
	waitForPolledConfigChange(t, changes)

	replacement := filepath.Join(filepath.Dir(path), "router.yaml.tmp")
	if err := os.WriteFile(replacement, []byte("newest"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, path); err != nil {
		t.Fatal(err)
	}
	ticks <- time.Now()
	waitForPolledConfigChange(t, changes)
}

func TestPollConfigFileChangesWhenSourceAppears(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router.yaml")
	ticks := make(chan time.Time)
	changes := make(chan struct{}, 1)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		pollConfigFileChanges(ctx, path, ticks, func() { changes <- struct{}{} })
	}()
	defer func() {
		cancel()
		<-done
	}()

	ticks <- time.Now()
	if err := os.WriteFile(path, []byte("created"), 0o600); err != nil {
		t.Fatal(err)
	}
	ticks <- time.Now()
	waitForPolledConfigChange(t, changes)
}

func waitForPolledConfigChange(t *testing.T, changes <-chan struct{}) {
	t.Helper()
	select {
	case <-changes:
	case <-time.After(time.Second):
		t.Fatal("config file change was not detected")
	}
}
