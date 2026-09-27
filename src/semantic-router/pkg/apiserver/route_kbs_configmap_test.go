//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

func TestKnowledgeBaseMutationsRejectConfigMapAssetStorage(t *testing.T) {
	server, _, configPath := newTestKnowledgeBaseAPIServer(t)
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(configwriter.ConfigMapNameEnv, "router-config")
	t.Setenv(configwriter.ConfigMapNamespaceEnv, "router-test")

	for _, tc := range []struct {
		name   string
		method string
		path   string
		handle http.HandlerFunc
	}{
		{"create", http.MethodPost, "/api/v1/storage/knowledge-bases", server.handleCreateKnowledgeBase},
		{"update", http.MethodPut, "/api/v1/storage/knowledge-bases/example", server.handleUpdateKnowledgeBase},
		{"delete", http.MethodDelete, "/api/v1/storage/knowledge-bases/example", server.handleDeleteKnowledgeBase},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := httptest.NewRecorder()
			tc.handle(response, httptest.NewRequest(tc.method, tc.path, nil))
			if response.Code != http.StatusForbidden || !strings.Contains(response.Body.String(), "KB_ASSET_STORAGE_READ_ONLY") {
				t.Fatalf("response = HTTP %d: %s", response.Code, response.Body.String())
			}
			after, err := os.ReadFile(configPath)
			if err != nil || !bytes.Equal(after, before) {
				t.Fatalf("rejected KB mutation changed config: %v", err)
			}
		})
	}
}
