package config

import (
	"strings"
	"testing"
)

func TestParseYAMLBytesRejectsUnknownProtectionFields(t *testing.T) {
	_, err := ParseYAMLBytes(canonicalLearningFixture("", `
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
	_, err := ParseYAMLBytes(canonicalLearningFixture("", `
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
