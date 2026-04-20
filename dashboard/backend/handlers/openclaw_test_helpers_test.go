package handlers

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// newTestOpenClawHandler returns an OpenClawHandler with a SQLite workflow store under dataDir.
func newTestOpenClawHandler(tb testing.TB, dataDir string, readOnly bool) *OpenClawHandler {
	tb.Helper()
	wfPath := filepath.Join(dataDir, "workflow.sqlite")
	wf, err := workflowstore.Open(wfPath, workflowstore.Options{LegacyOpenClawDir: dataDir})
	if err != nil {
		tb.Fatalf("workflow store: %v", err)
	}
	tb.Cleanup(func() { _ = wf.Close() })
	return NewOpenClawHandler(dataDir, readOnly, wf)
}
