package workflowstore

import (
	"path/filepath"
	"testing"
)

func TestMCPServerSurvivesStoreReopen(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "wf.sqlite")

	const (
		serverID = "mcp-test-server"
		payload  = `{"id":"mcp-test-server","name":"Test MCP","transport":"stdio","connection":{"command":"echo"},"enabled":true}`
	)

	s1, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if putErr := s1.PutMCPServerJSON(serverID, payload); putErr != nil {
		t.Fatal(putErr)
	}
	_ = s1.Close()

	s2, err := Open(path, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()

	rows, err := s2.ListMCPServerJSON()
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 1 || rows[0] != payload {
		t.Fatalf("rows = %#v, want one persisted config", rows)
	}

	if delErr := s2.DeleteMCPServer(serverID); delErr != nil {
		t.Fatal(delErr)
	}
	count, err := s2.MCPServerCount()
	if err != nil {
		t.Fatal(err)
	}
	if count != 0 {
		t.Fatalf("count = %d, want 0", count)
	}
}
