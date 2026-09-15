package config

import (
	"strings"
	"testing"
)

func TestParseYAMLBytesRejectsUnknownProtectionFields(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: cheap
global:
  router:
    learning:
      enabled: true
      protection:
        enabled: true
        privacy_affinity: true
`))
	if err == nil {
		t.Fatal("expected unknown protection field to be rejected")
	}
	if !strings.Contains(err.Error(), "global.router.learning.protection.privacy_affinity") {
		t.Fatalf("expected unknown protection field in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsUnknownProtectionIdentityHeader(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: cheap
global:
  router:
    learning:
      enabled: true
      protection:
        enabled: true
        identity:
          headers:
            session: x-session-id
            conversation: x-conversation-id
            run: x-run-id
`))
	if err == nil {
		t.Fatal("expected unknown protection identity header field to be rejected")
	}
	if !strings.Contains(err.Error(), "global.router.learning.protection.identity.headers.run") {
		t.Fatalf("expected unknown protection identity header in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsUnknownSuccessFields(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: cheap
global:
  router:
    learning:
      enabled: true
      adaptation:
        success:
          target_probability: 0.85
`))
	if err == nil {
		t.Fatal("expected unknown success field to be rejected")
	}
	if !strings.Contains(err.Error(), "global.router.learning.adaptation.success.target_probability") {
		t.Fatalf("expected unknown success field in error, got %v", err)
	}
}

func TestParseYAMLBytesRejectsInvalidSuccessValues(t *testing.T) {
	_, err := ParseYAMLBytes([]byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: cheap
global:
  router:
    learning:
      enabled: true
      adaptation:
        success:
          outcome: tool_success
          stale_after_seconds: -1
`))
	if err == nil {
		t.Fatal("expected invalid success values to be rejected")
	}
	if !strings.Contains(err.Error(), "adaptation.success") {
		t.Fatalf("expected success path in error, got %v", err)
	}
}
