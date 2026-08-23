package invitationmanagement

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestInvitationServiceAndFirstKeyIssuerOwnSecretKeyrings(t *testing.T) {
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1", Keys: map[string][]byte{
			"command-v1": []byte(strings.Repeat("c", 32)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer commands.Close()
	invitationPeppers := accesscredential.PepperKeyring{ActiveVersion: "invite-v1", Keys: map[string][]byte{
		"invite-v1": []byte(strings.Repeat("i", 32)),
	}}
	apiKeyPeppers := accesscredential.PepperKeyring{ActiveVersion: "api-key-v1", Keys: map[string][]byte{
		"api-key-v1": []byte(strings.Repeat("a", 32)),
	}}
	responseKEK := accesscredential.KEKKeyring{ActiveVersion: "response-v1", Keys: map[string][]byte{
		"response-v1": []byte(strings.Repeat("r", 32)),
	}}
	cursorKeyring := securitykeyring.Symmetric{ActiveVersion: "cursor-v1", Keys: map[string][]byte{
		"cursor-v1": []byte(strings.Repeat("u", 32)),
	}}
	firstKeys, err := NewAPIKeyFirstKeyPreparer(apiKeyPeppers, nil)
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewService(Options{
		Repository: lifecycleRepository{}, Commands: commands,
		CursorKeyring:     cursorKeyring,
		InvitationPeppers: invitationPeppers, ResponseKEK: responseKEK, FirstKeys: firstKeys,
		IdempotencyTTL: time.Hour, SecretDeliveryTTL: 10 * time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range []map[string][]byte{
		invitationPeppers.Keys, apiKeyPeppers.Keys, responseKEK.Keys, cursorKeyring.Keys,
	} {
		for _, key := range source {
			for index := range key {
				key[index] = 0
			}
		}
	}
	if service.responseKEK.Validate() != nil || firstKeys.peppers.Validate() != nil {
		t.Fatal("invitation service retained caller-owned secret key bytes")
	}
	if _, _, _, err := service.tokens.Issue("11111111-1111-4111-8111-111111111111"); err != nil {
		t.Fatalf("invitation token issue after source erasure: %v", err)
	}
	ownedResponse := service.responseKEK.Keys["response-v1"]
	ownedFirstKeyPepper := firstKeys.peppers.Keys["api-key-v1"]
	ownedCursorKey := service.cursors.keys["cursor-v1"]
	service.Close()
	for _, key := range [][]byte{ownedResponse, ownedFirstKeyPepper, ownedCursorKey} {
		for _, item := range key {
			if item != 0 {
				t.Fatal("invitation service Close did not erase an owned secret key")
			}
		}
	}
}

type lifecycleRepository struct{}

func (lifecycleRepository) Ready(context.Context, *managementcommand.Codec, []string, []string) error {
	return nil
}

func (lifecycleRepository) Get(context.Context, string, string) (Invitation, error) {
	return Invitation{}, ErrNotFound
}

func (lifecycleRepository) GetByID(context.Context, string) (Invitation, []byte, string, error) {
	return Invitation{}, nil, "", ErrNotFound
}

func (lifecycleRepository) List(context.Context, InvitationQuery) (RepositoryPage, error) {
	return RepositoryPage{}, nil
}

func (lifecycleRepository) ReplaySecret(context.Context, managementcommand.Command) (StoredSecret, bool, error) {
	return StoredSecret{}, false, nil
}

func (lifecycleRepository) ResolveSnapshot(context.Context, string, string, []RequestedRoleGrant, *TeamAssignment) (OnboardingSnapshot, error) {
	return OnboardingSnapshot{}, nil
}

func (lifecycleRepository) Create(context.Context, CreateMutation) (MutationResult, error) {
	return MutationResult{}, nil
}

func (lifecycleRepository) Rotate(context.Context, RotateMutation) (MutationResult, error) {
	return MutationResult{}, nil
}

func (lifecycleRepository) Revoke(context.Context, RevokeRequest) (MutationResult, error) {
	return MutationResult{}, nil
}

func (lifecycleRepository) Onboard(context.Context, PrivilegedOnboardingMutation) (AcceptanceEnvelope, error) {
	return AcceptanceEnvelope{}, nil
}

var _ Repository = lifecycleRepository{}
