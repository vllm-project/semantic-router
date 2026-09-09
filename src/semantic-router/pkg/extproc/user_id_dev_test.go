//go:build dev

package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestDevBuildUsesIngressIdentity(t *testing.T) {
	ctx := &RequestContext{
		Headers:         map[string]string{"x-authz-user-id": "spoofed"},
		TrustedIdentity: authz.TrustedIdentity{UserID: "user_from_ingress"},
		SemanticRequest: &llmprotocol.Request{Metadata: map[string]string{
			"user_id": "user_from_metadata",
		}},
	}
	if got := extractUserID(ctx); got != "user_from_ingress" {
		t.Fatalf("extractUserID() = %q, want ingress identity", got)
	}
}

func TestDevBuildDoesNotUseBodyMetadataForMemory(t *testing.T) {
	ctx := &RequestContext{
		RequestID: "req_123",
		SemanticRequest: &llmprotocol.Request{Metadata: map[string]string{
			"user_id": "user_from_metadata",
		}, Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "hello"}},
		}}},
	}

	_, userID, history, err := extractMemoryInfo(ctx)
	require.Error(t, err)
	require.Empty(t, userID)
	require.Len(t, history, 1)
}
