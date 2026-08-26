package invitationmanagement

import (
	"context"
	"encoding/json"
	"errors"
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
	revealKEK := accesscredential.KEKKeyring{ActiveVersion: "reveal-v1", Keys: map[string][]byte{
		"reveal-v1": []byte(strings.Repeat("v", 32)),
	}}
	cursorKeyring := securitykeyring.Symmetric{ActiveVersion: "cursor-v1", Keys: map[string][]byte{
		"cursor-v1": []byte(strings.Repeat("u", 32)),
	}}
	firstKeys, err := NewAPIKeyFirstKeyPreparer(apiKeyPeppers, &revealKEK, nil)
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewService(Options{
		Repository: lifecycleRepository{}, Commands: commands,
		CursorKeyring:     cursorKeyring,
		InvitationPeppers: invitationPeppers, ResponseKEK: responseKEK, FirstKeys: firstKeys,
		IdempotencyTTL: time.Hour, SecretDeliveryTTL: 10 * time.Minute,
		PublicationWaiter:  &invitationPublicationWaiterStub{},
		PublicationTimeout: 5 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range []map[string][]byte{
		invitationPeppers.Keys, apiKeyPeppers.Keys, responseKEK.Keys, revealKEK.Keys, cursorKeyring.Keys,
	} {
		for _, key := range source {
			for index := range key {
				key[index] = 0
			}
		}
	}
	if service.responseKEK.Validate() != nil || firstKeys.peppers.Validate() != nil ||
		firstKeys.revealKEK == nil || firstKeys.revealKEK.Validate() != nil {
		t.Fatal("invitation service retained caller-owned secret key bytes")
	}
	if _, _, _, err := service.tokens.Issue("11111111-1111-4111-8111-111111111111"); err != nil {
		t.Fatalf("invitation token issue after source erasure: %v", err)
	}
	ownedResponse := service.responseKEK.Keys["response-v1"]
	ownedFirstKeyPepper := firstKeys.peppers.Keys["api-key-v1"]
	ownedFirstKeyReveal := firstKeys.revealKEK.Keys["reveal-v1"]
	ownedCursorKey := service.cursors.keys["cursor-v1"]
	service.Close()
	for _, key := range [][]byte{ownedResponse, ownedFirstKeyPepper, ownedFirstKeyReveal, ownedCursorKey} {
		for _, item := range key {
			if item != 0 {
				t.Fatal("invitation service Close did not erase an owned secret key")
			}
		}
	}
}

type lifecycleRepository struct{}

func TestInvitationSecretCanonicalJSONEncodesRequiredCollectionsAsArrays(t *testing.T) {
	invitation, body, err := marshalIssuedInvitation(Invitation{
		Snapshot: OnboardingSnapshot{RoleGrants: []RoleGrant{{}}},
	}, "vsi_secret", time.Date(2026, 8, 26, 12, 5, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if invitation.Snapshot.RoleGrants == nil || invitation.Snapshot.RoleGrants[0].DelegationCeiling == nil {
		t.Fatal("canonical invitation did not preserve the non-nil collection invariant")
	}
	var payload struct {
		Data struct {
			Onboarding struct {
				RoleGrants []struct {
					DelegationCeiling json.RawMessage `json:"delegationCeiling"`
				} `json:"roleGrants"`
			} `json:"onboarding"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	if string(payload.Data.Onboarding.RoleGrants[0].DelegationCeiling) != "[]" {
		t.Fatalf("canonical InvitationRoleGrant.delegationCeiling must be an array: %s", body)
	}
}

type invitationPublicationWaiterStub struct {
	err         error
	waits       int
	namespaceID string
	keyID       string
	publicID    string
	hasDeadline bool
}

func (waiter *invitationPublicationWaiterStub) WaitAPIKeyActive(
	ctx context.Context,
	namespaceID string,
	keyID string,
	publicID string,
) error {
	_, waiter.hasDeadline = ctx.Deadline()
	waiter.waits++
	waiter.namespaceID = namespaceID
	waiter.keyID = keyID
	waiter.publicID = publicID
	return waiter.err
}

func TestFirstKeySecretWaitsForAppliedPublication(t *testing.T) {
	const (
		namespaceID = "11111111-1111-4111-8111-111111111111"
		keyID       = "22222222-2222-4222-8222-222222222222"
		publicID    = "33333333333343338333333333333333"
	)
	issued, err := (accesscredential.PepperKeyring{
		ActiveVersion: "v1",
		Keys: map[string][]byte{
			"v1": []byte(strings.Repeat("p", 32)),
		},
	}).Issue(accesscredential.KindAPIKey, publicID)
	if err != nil {
		t.Fatal(err)
	}
	waiter := &invitationPublicationWaiterStub{}
	service := &Service{publicationWaiter: waiter, publicationTimeout: time.Second}
	result := AcceptanceResult{APIKeyID: keyID, APIKey: issued.Plaintext}

	if err := service.waitFirstKeyActive(context.Background(), namespaceID, result); err != nil {
		t.Fatal(err)
	}
	if waiter.waits != 1 || waiter.namespaceID != namespaceID || waiter.keyID != keyID ||
		waiter.publicID != publicID || !waiter.hasDeadline {
		t.Fatalf("publication waiter received %#v", waiter)
	}

	waiter.err = context.DeadlineExceeded
	if err := service.waitFirstKeyActive(context.Background(), namespaceID, result); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("publication failure = %v, want unavailable", err)
	}
}

func TestPrivilegedOnboardingDoesNotExposeKeyWhenPublicationFails(t *testing.T) {
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1",
		Keys: map[string][]byte{
			"command-v1": []byte(strings.Repeat("c", 32)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = commands.Close() })
	firstKeys, err := NewAPIKeyFirstKeyPreparer(accesscredential.PepperKeyring{
		ActiveVersion: "api-v1",
		Keys: map[string][]byte{
			"api-v1": []byte(strings.Repeat("a", 32)),
		},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewService(Options{
		Repository: onboardingPublicationRepository{},
		Commands:   commands,
		CursorKeyring: securitykeyring.Symmetric{
			ActiveVersion: "cursor-v1",
			Keys: map[string][]byte{
				"cursor-v1": []byte(strings.Repeat("u", 32)),
			},
		},
		InvitationPeppers: accesscredential.PepperKeyring{
			ActiveVersion: "invite-v1",
			Keys: map[string][]byte{
				"invite-v1": []byte(strings.Repeat("i", 32)),
			},
		},
		ResponseKEK: accesscredential.KEKKeyring{
			ActiveVersion: "response-v1",
			Keys: map[string][]byte{
				"response-v1": []byte(strings.Repeat("r", 32)),
			},
		},
		FirstKeys:          firstKeys,
		PublicationWaiter:  &invitationPublicationWaiterStub{err: context.DeadlineExceeded},
		IdempotencyTTL:     time.Hour,
		SecretDeliveryTTL:  10 * time.Minute,
		PublicationTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	const (
		namespaceID = "11111111-1111-4111-8111-111111111111"
		principalID = "22222222-2222-4222-8222-222222222222"
		actorID     = "33333333-3333-4333-8333-333333333333"
		roleID      = "44444444-4444-4444-8444-444444444444"
	)
	result, err := service.Onboard(context.Background(), PrivilegedOnboardingRequest{
		NamespaceID:    namespaceID,
		PrincipalID:    principalID,
		Email:          "new.user@example.com",
		DisplayName:    "New user",
		RoleGrants:     []RequestedRoleGrant{{RoleID: roleID, ScopeKind: "namespace"}},
		CreateFirstKey: true,
		IdempotencyKey: "privileged-onboarding-publication-0001",
		Actor: Actor{
			PrincipalID: actorID,
			ActorChain:  []string{actorID},
			RequestID:   "publication-barrier-test",
			Reason:      "Test publication barrier.",
		},
	})
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("onboarding error = %v, want unavailable", err)
	}
	if result.Result.APIKey != "" || result.Result.APIKeyID != "" || result.CanonicalJSON != nil || result.Replayed {
		t.Fatalf("secret-bearing result escaped with error: %#v", result)
	}
}

type onboardingPublicationRepository struct {
	lifecycleRepository
}

func (onboardingPublicationRepository) ResolveSnapshot(
	_ context.Context,
	_, _ string,
	grants []RequestedRoleGrant,
	team *TeamAssignment,
) (OnboardingSnapshot, error) {
	resolved := make([]RoleGrant, len(grants))
	for index, grant := range grants {
		resolved[index] = RoleGrant{RoleID: grant.RoleID, ScopeKind: grant.ScopeKind}
	}
	return OnboardingSnapshot{RoleGrants: resolved, Team: team}, nil
}

func (onboardingPublicationRepository) Onboard(
	_ context.Context,
	mutation PrivilegedOnboardingMutation,
) (AcceptanceEnvelope, error) {
	envelope, expiresAt, err := mutation.SealResult(AcceptanceResult{
		PrincipalID: mutation.PrincipalID,
		UserID:      mutation.UserID,
	})
	if err != nil {
		return AcceptanceEnvelope{}, err
	}
	return AcceptanceEnvelope{
		Invitation: Invitation{
			NamespaceID:    mutation.NamespaceID,
			AcceptedUserID: mutation.UserID,
		},
		Envelope:  envelope,
		ExpiresAt: expiresAt,
	}, nil
}

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
