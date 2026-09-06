//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
)

func TestContextCompressionPreviewIsRedactedAndApplied(t *testing.T) {
	server := &ClassificationAPIServer{
		contextCompression: contextcompression.NewService(),
	}
	body := map[string]interface{}{
		"configuration": map[string]interface{}{
			"enabled": true,
			"mode":    "always",
			"targets": map[string]interface{}{
				"tool_outputs": map[string]interface{}{
					"mode":          "extractive",
					"min_tokens":    100,
					"target_tokens": 20,
				},
			},
		},
		"request_body": map[string]interface{}{
			"model": "model",
			"messages": []interface{}{
				map[string]interface{}{"role": "user", "content": "find failure"},
				map[string]interface{}{
					"role": "tool",
					"content": strings.Repeat("secret-value irrelevant logs ", 300) +
						"target failure",
				},
			},
		},
		"model": "model",
	}
	encoded, _ := json.Marshal(body)
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/context-compression/preview",
		bytes.NewReader(encoded),
	)
	recorder := httptest.NewRecorder()
	server.handleContextCompressionPreview(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", recorder.Code, recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "secret-value") {
		t.Fatal("preview response exposed source content")
	}
	var response contextCompressionPreviewResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode preview: %v", err)
	}
	if !response.Applied || response.TokensAfter >= response.TokensBefore {
		t.Fatalf("preview = %#v", response)
	}
}

func TestContextCompressionRoutesUseDedicatedPermissions(t *testing.T) {
	permissions := map[string]RoutePermission{}
	for _, route := range apiContextCompressionRoutes() {
		permissions[route.Path] = route.Permission
	}
	if permissions["/api/v1/context-compression/stats"] != PermCompressionRead {
		t.Fatalf("stats permission = %q", permissions["/api/v1/context-compression/stats"])
	}
	if permissions["/api/v1/context-compression/preview"] != PermCompressionPreview {
		t.Fatalf("preview permission = %q", permissions["/api/v1/context-compression/preview"])
	}
	if permissions["/api/v1/context-compression/recovery/invalidate"] != PermCompressionManage {
		t.Fatalf(
			"invalidate permission = %q",
			permissions["/api/v1/context-compression/recovery/invalidate"],
		)
	}
}

func TestContextCompressionPreviewHonorsDisabledPolicy(t *testing.T) {
	server := &ClassificationAPIServer{
		contextCompression: contextcompression.NewService(),
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/context-compression/preview",
		strings.NewReader(`{
			"configuration":{"enabled":false},
			"request_body":{"messages":[]}
		}`),
	)
	recorder := httptest.NewRecorder()
	server.handleContextCompressionPreview(recorder, request)
	if recorder.Code != http.StatusOK ||
		!strings.Contains(recorder.Body.String(), `"skip_reason":"disabled"`) {
		t.Fatalf("disabled preview status=%d body=%s", recorder.Code, recorder.Body.String())
	}
}

func TestContextCompressionStatsDoNotExposeContent(t *testing.T) {
	server := &ClassificationAPIServer{
		contextCompression: contextcompression.NewService(),
		managementAuditEntries: []managementAuditEntry{
			{Sequence: 1, Action: AuditActionCompressionPreview, Hash: "hash"},
			{Sequence: 2, Action: AuditActionCacheFlush, Hash: "other"},
		},
	}
	request := httptest.NewRequest(
		http.MethodGet,
		"/api/v1/context-compression/stats",
		nil,
	)
	recorder := httptest.NewRecorder()
	server.handleContextCompressionStats(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d", recorder.Code)
	}
	for _, forbidden := range []string{"prompt", "content", "request_body"} {
		if strings.Contains(recorder.Body.String(), forbidden) {
			t.Fatalf("stats exposed %q: %s", forbidden, recorder.Body.String())
		}
	}
	if !strings.Contains(recorder.Body.String(), "compression.preview") ||
		strings.Contains(recorder.Body.String(), "cache.flush") {
		t.Fatalf("stats audit was not scoped: %s", recorder.Body.String())
	}
}

func TestContextCompressionRecoveryInvalidateUsesTrustedScope(t *testing.T) {
	store := newPreviewRecoveryStore()
	scope := cache.CombineFingerprints(
		"recipe",
		"decision",
		cache.UserScopeNamespace("user"),
		"request",
	)
	if err := store.Put(context.Background(), contextcompression.RecoveryEntry{
		Key:     "key",
		Scope:   scope,
		Content: "sensitive content",
	}, time.Minute); err != nil {
		t.Fatalf("seed recovery store: %v", err)
	}
	server := &ClassificationAPIServer{
		contextCompression:  contextcompression.NewService(),
		compressionRecovery: store,
	}
	request := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/context-compression/recovery/invalidate",
		strings.NewReader(`{
			"recipe":"recipe",
			"decision":"decision",
			"user_id":"user",
			"request_id":"request"
		}`),
	)
	recorder := httptest.NewRecorder()
	server.handleContextCompressionRecoveryInvalidate(recorder, request)
	if recorder.Code != http.StatusOK ||
		!strings.Contains(recorder.Body.String(), `"deleted":1`) {
		t.Fatalf("invalidate status=%d body=%s", recorder.Code, recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "user") ||
		strings.Contains(recorder.Body.String(), "request") {
		t.Fatalf("invalidate response leaked scope: %s", recorder.Body.String())
	}
}
