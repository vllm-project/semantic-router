package apikeymanagement

import (
	"context"
	"encoding/json"
	"errors"
	"net/netip"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/policymanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	testNamespaceID   = "11111111-1111-4111-8111-111111111111"
	testPrincipalID   = "22222222-2222-4222-8222-222222222222"
	testOwnerID       = "33333333-3333-4333-8333-333333333333"
	testKeyID         = "44444444-4444-4444-8444-444444444444"
	testCredentialID  = "55555555-5555-4555-8555-555555555555"
	testAccessPolicyA = "66666666-6666-4666-8666-666666666666"
	testAccessPolicyB = "77777777-7777-4777-8777-777777777777"
	testAccessBinding = "88888888-8888-4888-8888-888888888888"
	testSecondBinding = "99999999-9999-4999-8999-999999999999"
	testRateBinding   = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
	testRatePolicy    = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
	testRateRule      = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
	testRotated       = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"
)

func TestCreateCompilesImmutablePolicyReceipts(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	repository := &apiKeyRepositoryStub{}
	service := newAPIKeyTestService(t, repository, func() time.Time { return now }, []string{
		testKeyID, testCredentialID, testAccessBinding, testSecondBinding,
		testRateBinding, testRatePolicy, testRateRule,
	})
	result, err := service.Create(context.Background(), CreateRequest{
		NamespaceID: testNamespaceID, Name: "Developer key",
		Owner:           Owner{Kind: accesscontrol.SubjectKindUser, ID: testOwnerID},
		AccessPolicyIDs: []string{testAccessPolicyB, testAccessPolicyA},
		RateLimitOverride: &RateLimitOverrideInput{InlinePolicy: &InlineRateLimitPolicyInput{
			Name: "Burst budget", Description: "One minute",
			Rules: []policymanagement.RateLimitRule{{
				Metric: accesscontrol.RateMetricRequests, Algorithm: accesscontrol.RateAlgorithmSlidingLog,
				Limit: "12", Window: policymanagement.ISODuration(time.Minute),
				Accounting: accesscontrol.RateAccountingRequest, Enforcement: accesscontrol.RateEnforcementEnforce,
			}},
		}},
		IdempotencyKey: "create-api-key-policy-0001", Actor: testAPIKeyActor(),
	})
	if err != nil {
		t.Fatal(err)
	}
	mutation := repository.created
	if len(mutation.AccessBindings) != 2 || mutation.AccessBindings[0].PolicyID != testAccessPolicyA ||
		mutation.AccessBindings[1].PolicyID != testAccessPolicyB {
		t.Fatalf("canonical AccessPolicy bindings = %#v", mutation.AccessBindings)
	}
	if mutation.RateLimitOverride == nil || mutation.RateLimitOverride.InlinePolicy == nil ||
		mutation.RateLimitOverride.InlinePolicy.ID != testRatePolicy ||
		mutation.RateLimitOverride.Binding.ID != testRateBinding ||
		mutation.RateLimitOverride.InlinePolicy.Rules[0].ID != testRateRule {
		t.Fatalf("compiled rate override = %#v", mutation.RateLimitOverride)
	}
	var issued IssuedSecret
	if err := json.Unmarshal(result.CanonicalJSON, &issued); err != nil {
		t.Fatal(err)
	}
	if len(issued.AccessPolicyBindings) != 2 || issued.RateLimitOverride == nil ||
		!issued.RateLimitOverride.Created || issued.RateLimitOverride.PolicyID != testRatePolicy ||
		issued.RateLimitOverride.BindingID != testRateBinding {
		t.Fatalf("issued receipts = %#v / %#v", issued.AccessPolicyBindings, issued.RateLimitOverride)
	}
	if strings.Contains(string(result.CanonicalJSON), "Burst budget") || strings.Contains(string(result.CanonicalJSON), "rules") {
		t.Fatalf("mutable policy snapshot leaked into issued response: %s", result.CanonicalJSON)
	}
}

func TestCreateReplicaReplayUsesTimeAfterCommandLock(t *testing.T) {
	t0 := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	clockCalls := 0
	repository := &apiKeyRepositoryStub{replayCreate: true, replayCredentialDelay: time.Millisecond}
	service := newAPIKeyTestService(t, repository, func() time.Time {
		clockCalls++
		if clockCalls == 1 {
			return t0
		}
		return t0.Add(2 * time.Millisecond)
	}, []string{testKeyID, testCredentialID})
	result, err := service.Create(context.Background(), CreateRequest{
		NamespaceID: testNamespaceID, Name: "Replica key",
		Owner:          Owner{Kind: accesscontrol.SubjectKindUser, ID: testOwnerID},
		IdempotencyKey: "replica-command-race-0001", Actor: testAPIKeyActor(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Replayed || clockCalls < 2 {
		t.Fatalf("replay result=%#v clockCalls=%d", result, clockCalls)
	}
}

func TestRotateUsesIndexedActiveCredentialLookup(t *testing.T) {
	now := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	repository := &apiKeyRepositoryStub{
		key: testAPIKey(now),
		active: RevealSnapshot{NamespaceID: testNamespaceID, Credential: accesscontrol.CredentialVersion{
			ID: testCredentialID, APIKeyID: testKeyID, KID: "existing-credential",
			SecretHMAC: []byte("digest"), PepperVersion: "pepper-v1",
			Status: accesscontrol.CredentialStatusActive, NotBefore: now, CreatedAt: now,
		}},
	}
	service := newAPIKeyTestService(t, repository, func() time.Time { return now }, []string{testRotated})
	result, err := service.Rotate(context.Background(), RotateRequest{
		NamespaceID: testNamespaceID, KeyID: testKeyID, ExpectedRevision: 1,
		IdempotencyKey: "rotate-indexed-credential-01", Actor: testAPIKeyActor(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Credential.ID != testRotated || repository.activeReads != 1 || repository.credentialLists != 0 {
		t.Fatalf("rotate result=%#v activeReads=%d lists=%d", result, repository.activeReads, repository.credentialLists)
	}
}

type apiKeyRepositoryStub struct {
	key                   accesscontrol.APIKey
	active                RevealSnapshot
	created               CreateMutation
	replayCreate          bool
	replayCredentialDelay time.Duration
	activeReads           int
	credentialLists       int
}

func (repository *apiKeyRepositoryStub) Ready(context.Context, *managementcommand.Codec) error {
	return nil
}

func (repository *apiKeyRepositoryStub) ReplaySecret(context.Context, managementcommand.Command) (StoredSecret, bool, error) {
	return StoredSecret{}, false, nil
}

func (repository *apiKeyRepositoryStub) ReplayMutation(context.Context, managementcommand.Command) (MutationResult, bool, error) {
	return MutationResult{}, false, nil
}

func (repository *apiKeyRepositoryStub) GetKey(context.Context, string, string) (accesscontrol.APIKey, error) {
	if repository.key.ID == "" {
		return accesscontrol.APIKey{}, ErrNotFound
	}
	return repository.key, nil
}

func (repository *apiKeyRepositoryStub) ListKeys(context.Context, KeyQuery) (RepositoryPage[accesscontrol.APIKey], error) {
	return RepositoryPage[accesscontrol.APIKey]{}, nil
}

func (repository *apiKeyRepositoryStub) CreateKey(_ context.Context, mutation CreateMutation) (MutationResult, error) {
	repository.created = mutation
	repository.key = mutation.Key
	repository.active = RevealSnapshot{NamespaceID: string(mutation.Key.NamespaceID), Credential: mutation.Credential}
	if repository.replayCredentialDelay > 0 {
		repository.active.Credential.NotBefore = mutation.Credential.NotBefore.Add(repository.replayCredentialDelay)
	}
	if repository.replayCreate {
		stored := StoredSecret{Result: managementcommand.ResourceResult{
			ResourceType: "api_key", ResourceID: string(mutation.Key.ID),
			ResourceRevision: uint64(mutation.Key.Revision), ResponseStatus: 201,
		}, Secret: managementcommand.SecretResponse{
			Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
			KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
		}}
		return MutationResult{Key: mutation.Key, HTTPStatus: 201, Replayed: true, Stored: &stored}, nil
	}
	return MutationResult{Key: mutation.Key, HTTPStatus: 201}, nil
}

func (repository *apiKeyRepositoryStub) UpdateKey(context.Context, accesscontrol.APIKey, uint64, Actor, string) (MutationResult, error) {
	return MutationResult{}, ErrUnavailable
}

func (repository *apiKeyRepositoryStub) UpdateKeyAction(context.Context, UpdateMutation) (MutationResult, error) {
	return MutationResult{}, ErrUnavailable
}

func (repository *apiKeyRepositoryStub) DeleteKey(context.Context, string, string, uint64, Actor) (MutationResult, error) {
	return MutationResult{}, ErrUnavailable
}

func (repository *apiKeyRepositoryStub) ListCredentials(context.Context, CredentialQuery) (RepositoryPage[RevealSnapshot], error) {
	repository.credentialLists++
	return RepositoryPage[RevealSnapshot]{}, errors.New("credential list must not implement identity lookup")
}

func (repository *apiKeyRepositoryStub) GetCredential(_ context.Context, _, _, credentialID string) (RevealSnapshot, error) {
	if string(repository.active.Credential.ID) != credentialID {
		return RevealSnapshot{}, ErrCredentialUnavailable
	}
	return repository.active, nil
}

func (repository *apiKeyRepositoryStub) GetActiveCredential(context.Context, string, string) (RevealSnapshot, error) {
	repository.activeReads++
	return repository.active, nil
}

func (repository *apiKeyRepositoryStub) RotateCredential(_ context.Context, mutation RotateMutation) (MutationResult, error) {
	repository.key.Revision++
	repository.key.UpdatedAt = mutation.Credential.CreatedAt
	repository.active = RevealSnapshot{NamespaceID: mutation.NamespaceID, Credential: mutation.Credential}
	return MutationResult{Key: repository.key, HTTPStatus: 200}, nil
}

func (repository *apiKeyRepositoryStub) RevokeCredential(context.Context, string, string, string, uint64, Actor) (MutationResult, error) {
	return MutationResult{}, ErrUnavailable
}

func (repository *apiKeyRepositoryStub) GetRevealSnapshot(context.Context, string, string, string) (RevealSnapshot, error) {
	return RevealSnapshot{}, ErrUnavailable
}

func (repository *apiKeyRepositoryStub) RecordReveal(context.Context, RevealSnapshot, Actor) error {
	return ErrUnavailable
}

func newAPIKeyTestService(
	t *testing.T,
	repository Repository,
	now func() time.Time,
	ids []string,
) *Service {
	t.Helper()
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1", Keys: map[string][]byte{"command-v1": []byte(strings.Repeat("c", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	index := 0
	service, err := NewService(Options{
		Repository: repository, Commands: commands,
		CursorKeyring: securitykeyring.Symmetric{ActiveVersion: "cursor-v1", Keys: map[string][]byte{
			"cursor-v1": []byte(strings.Repeat("u", 32)),
		}},
		APIKeyPeppers: accesscredential.PepperKeyring{ActiveVersion: "pepper-v1", Keys: map[string][]byte{
			"pepper-v1": []byte(strings.Repeat("p", 32)),
		}},
		ResponseKEK: accesscredential.KEKKeyring{ActiveVersion: "response-v1", Keys: map[string][]byte{
			"response-v1": []byte(strings.Repeat("r", 32)),
		}},
		IdempotencyTTL: time.Hour, SecretDeliveryTTL: 10 * time.Minute, Now: now,
		NewID: func() string {
			if index >= len(ids) {
				t.Fatalf("deterministic ID source exhausted at %d", index)
			}
			value := ids[index]
			index++
			return value
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	return service
}

func testAPIKeyActor() Actor {
	return Actor{
		PrincipalID: testPrincipalID, ActorChain: []string{testPrincipalID},
		RequestID: "api-key-service-test", SourceIP: netip.MustParseAddr("192.0.2.20"),
	}
}

func testAPIKey(now time.Time) accesscontrol.APIKey {
	return accesscontrol.APIKey{
		NamespaceID: testNamespaceID, ID: testKeyID, Name: "Existing key",
		Owner:  accesscontrol.SubjectRef{NamespaceID: testNamespaceID, ID: testOwnerID, Kind: accesscontrol.SubjectKindUser},
		Status: accesscontrol.APIKeyStatusActive, PolicyEpoch: 1, DelegationEpoch: 1,
		Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
}

func TestServiceOwnsAndErasesSecretKeyrings(t *testing.T) {
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1", Keys: map[string][]byte{
			"command-v1": []byte(strings.Repeat("c", 32)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer commands.Close()
	peppers := accesscredential.PepperKeyring{ActiveVersion: "pepper-v1", Keys: map[string][]byte{
		"pepper-v1": []byte(strings.Repeat("p", 32)),
	}}
	responseKEK := accesscredential.KEKKeyring{ActiveVersion: "response-v1", Keys: map[string][]byte{
		"response-v1": []byte(strings.Repeat("r", 32)),
	}}
	revealKEK := accesscredential.KEKKeyring{ActiveVersion: "reveal-v1", Keys: map[string][]byte{
		"reveal-v1": []byte(strings.Repeat("v", 32)),
	}}
	service, err := NewService(Options{
		Repository: &apiKeyRepositoryStub{}, Commands: commands,
		CursorKeyring: securitykeyring.Symmetric{ActiveVersion: "cursor-v1", Keys: map[string][]byte{
			"cursor-v1": []byte(strings.Repeat("u", 32)),
		}},
		APIKeyPeppers: peppers, ResponseKEK: responseKEK, RevealKEK: &revealKEK,
		IdempotencyTTL: time.Hour, SecretDeliveryTTL: 10 * time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range []map[string][]byte{peppers.Keys, responseKEK.Keys, revealKEK.Keys} {
		for _, key := range source {
			for index := range key {
				key[index] = 0
			}
		}
	}
	if service.peppers.Validate() != nil || service.responseKEK.Validate() != nil ||
		service.revealKEK == nil || service.revealKEK.Validate() != nil {
		t.Fatal("service retained caller-owned secret key bytes")
	}
	ownedPepper := service.peppers.Keys["pepper-v1"]
	ownedResponse := service.responseKEK.Keys["response-v1"]
	ownedReveal := service.revealKEK.Keys["reveal-v1"]
	service.Close()
	for _, key := range [][]byte{ownedPepper, ownedResponse, ownedReveal} {
		for _, item := range key {
			if item != 0 {
				t.Fatal("service Close did not erase an owned secret key")
			}
		}
	}
}

var _ Repository = (*apiKeyRepositoryStub)(nil)
