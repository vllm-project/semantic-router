package agentruntime

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agenttoolsource"
)

type ToolCredentialStore interface {
	ResolveToolCredentialSecret(context.Context, string, string, string) (agentmanagement.ToolCredentialSecret, error)
}

// CredentialVault decrypts only the exact version pinned into a Tool Registry
// manifest. Plaintext exists only for one remote call and is cleared by the
// Tool Source client on close.
type CredentialVault struct {
	store ToolCredentialStore
	codec agentmanagement.SecretCodec
	now   func() time.Time
}

func NewCredentialVault(
	store ToolCredentialStore, codec agentmanagement.SecretCodec, now func() time.Time,
) (*CredentialVault, error) {
	if store == nil || codec == nil {
		return nil, fmt.Errorf("agent Tool credential vault dependencies are incomplete")
	}
	if now == nil {
		now = time.Now
	}
	return &CredentialVault{store: store, codec: codec, now: now}, nil
}

func (vault *CredentialVault) Resolve(
	ctx context.Context, namespaceID, credentialID, versionID string,
) (agenttoolsource.PinnedCredential, error) {
	stored, err := vault.store.ResolveToolCredentialSecret(ctx, namespaceID, credentialID, versionID)
	if err != nil {
		return agenttoolsource.PinnedCredential{}, err
	}
	if stored.ExpiresAt != nil && !vault.now().UTC().Before(stored.ExpiresAt.UTC()) {
		return agenttoolsource.PinnedCredential{}, agentmanagement.ErrToolUnavailable
	}
	secret, err := vault.codec.Decrypt(ctx, stored.Secret)
	if err != nil || len(secret) == 0 {
		clear(secret)
		return agenttoolsource.PinnedCredential{}, agentmanagement.ErrToolUnavailable
	}
	return agenttoolsource.PinnedCredential{
		VersionID: stored.VersionID, Secret: secret,
	}, nil
}

var _ agenttoolsource.CredentialResolver = (*CredentialVault)(nil)
