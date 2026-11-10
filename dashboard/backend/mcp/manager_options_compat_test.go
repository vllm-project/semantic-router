package mcp

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// Releases before the Auto Reconnect control was removed persisted reconnect
// settings that the runtime never enforced. Those stored configs must keep
// loading, with the enforced timeout intact, after the fields are gone.
func TestLoadConfigsToleratesLegacyReconnectOptions(t *testing.T) {
	t.Parallel()

	const legacyServerID = "legacy-options-server"
	legacyPayload := `{"id":"legacy-options-server","name":"Legacy","transport":"stdio",` +
		`"connection":{"command":"echo"},"enabled":true,` +
		`"options":{"auto_reconnect":false,"reconnect_interval":1500,"max_retries":5,"timeout":12345}}`

	storePath := filepath.Join(t.TempDir(), "wf.sqlite")
	store, err := workflowstore.Open(storePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if err = store.PutMCPServerJSON(legacyServerID, legacyPayload); err != nil {
		t.Fatal(err)
	}

	manager, err := NewManager(store)
	if err != nil {
		t.Fatalf("load stored config with legacy options: %v", err)
	}

	config := manager.configs[legacyServerID]
	if config == nil {
		t.Fatal("stored config with legacy options did not load")
	}
	if config.Options.Timeout != 12345 {
		t.Fatalf("Options.Timeout = %d, want 12345", config.Options.Timeout)
	}
}
