package handlers

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// newTestOpenClawHandler returns an OpenClawHandler with a SQLite workflow store under dataDir.
func newTestOpenClawHandler(tb testing.TB, dataDir string, readOnly bool) *OpenClawHandler {
	tb.Helper()
	if os.Getenv("OPENCLAW_CONTAINER_RUNTIME") == "" && os.Getenv("CONTAINER_RUNTIME") == "" {
		fakeRuntime := filepath.Join(tb.TempDir(), "unavailable-container-runtime")
		if err := os.WriteFile(fakeRuntime, []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
			tb.Fatalf("write unavailable container runtime: %v", err)
		}
		tb.Setenv("OPENCLAW_CONTAINER_RUNTIME", fakeRuntime)
	}
	wfPath := filepath.Join(dataDir, "workflow.sqlite")
	wf, err := workflowstore.Open(wfPath, workflowstore.Options{LegacyOpenClawDir: dataDir})
	if err != nil {
		tb.Fatalf("workflow store: %v", err)
	}
	tb.Cleanup(func() { _ = wf.Close() })
	return NewOpenClawHandler(dataDir, readOnly, wf)
}
