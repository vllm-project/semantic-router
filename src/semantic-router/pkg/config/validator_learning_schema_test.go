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

func TestParseYAMLBytesAcceptsAgentIdentityHeader(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(`
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
        identity:
          headers:
            session: session-id
            agent: thread-id
`))
	if err != nil {
		t.Fatal(err)
	}
	if got := cfg.RouterLearning.Protection.HeaderName("agent"); got != "thread-id" {
		t.Fatalf("agent header = %q", got)
	}
	if got := cfg.RouterLearning.Protection.HeaderName("session"); got != "session-id" {
		t.Fatalf("session header = %q", got)
	}
}

func TestParseYAMLBytesRejectsEmptyAgentIdentityHeader(t *testing.T) {
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
      protection:
        identity:
          headers:
            agent: " "
`))
	if err == nil || !strings.Contains(err.Error(), "identity.headers.agent cannot be empty") {
		t.Fatalf("expected empty agent header to be rejected, got %v", err)
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
