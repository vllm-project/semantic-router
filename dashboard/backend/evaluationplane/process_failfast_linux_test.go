//go:build linux

package evaluationplane

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestOversizedWorkerLineKillsProcessGroupAndReleasesCapacity(t *testing.T) {
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create evaluation root: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	pidPath := filepath.Join(t.TempDir(), "descendant.pid")
	workerPath := filepath.Join(t.TempDir(), "oversized-worker")
	worker := fmt.Sprintf(
		"#!/bin/sh\nsleep 30 &\nchild=$!\nprintf '%%s\\n' \"$child\" > %s\nhead -c %d /dev/zero | tr '\\000' x\nprintf '\\n'\nwait\n",
		strconv.Quote(pidPath), maxWorkerEventLineBytes+1024,
	)
	if err := os.WriteFile(workerPath, []byte(worker), 0o700); err != nil {
		t.Fatalf("write worker helper: %v", err)
	}
	service, err := NewService(Options{
		DataDir: root, PythonPath: workerPath, ConfigPath: configPath,
		CodeRevision: testSourceRevision, MaxConcurrent: 1, WorkerTimeout: time.Minute,
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	t.Cleanup(func() {
		if closeErr := service.Close(); closeErr != nil {
			t.Errorf("close evaluation service: %v", closeErr)
		}
	})
	run, err := service.CreateRun(context.Background(), validCreateRequest())
	if err != nil {
		t.Fatalf("CreateRun: %v", err)
	}
	startedAt := time.Now()
	if _, startErr := service.StartRun(context.Background(), run.ID); startErr != nil {
		t.Fatalf("StartRun: %v", startErr)
	}
	failed := waitForRunStatus(t, service, run.ID, StatusFailed)
	if time.Since(startedAt) > 2*time.Second || !strings.Contains(failed.Error, "protected server diagnostics") {
		t.Fatalf("oversized protocol line did not fail fast and safely: elapsed=%s run=%+v", time.Since(startedAt), failed)
	}
	capacityDeadline := time.Now().Add(time.Second)
	for len(service.semaphore) != 0 && time.Now().Before(capacityDeadline) {
		time.Sleep(time.Millisecond)
	}
	if len(service.semaphore) != 0 {
		t.Fatalf("failed worker retained a concurrency slot: %d", len(service.semaphore))
	}
	pidBytes, err := os.ReadFile(pidPath)
	if err != nil {
		t.Fatalf("read descendant pid: %v", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(pidBytes)))
	if err != nil {
		t.Fatalf("parse descendant pid: %v", err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		err := syscall.Kill(pid, 0)
		if errors.Is(err, syscall.ESRCH) || linuxProcessIsZombie(pid) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("oversized-line worker descendant %d survived fail-fast termination: %v", pid, err)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
