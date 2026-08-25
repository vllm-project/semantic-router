package delegationmanagement

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	testDelegationNamespace = "11111111-1111-4111-8111-111111111111"
	testDelegationPrincipal = "22222222-2222-4222-8222-222222222222"
	testManagementSession   = "33333333-3333-4333-8333-333333333333"
	testOtherSession        = "44444444-4444-4444-8444-444444444444"
	testDelegationUser      = "55555555-5555-4555-8555-555555555555"
	testDelegationKey       = "66666666-6666-4666-8666-666666666666"
	testDelegatedSession    = "77777777-7777-4777-8777-777777777777"
)

func TestCreateStoresOnlyVerifierAndWaitsForActiveProjection(t *testing.T) {
	now := time.Date(2026, 8, 23, 8, 0, 0, 0, time.UTC)
	repository := &delegationRepositoryStub{now: now}
	waiter := &delegationWaiterStub{}
	service := newDelegationTestService(t, repository, waiter, now)

	result, err := service.Create(context.Background(), CreateRequest{
		NamespaceID:    testDelegationNamespace,
		KeyID:          testDelegationKey,
		IdempotencyKey: "delegation-create-0001",
		Actor:          testDelegationActor(testManagementSession),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !waiter.active || waiter.desiredRevision != 7 {
		t.Fatalf("publication waiter = active:%v revision:%d", waiter.active, waiter.desiredRevision)
	}
	if len(repository.created.Session.TokenHMAC) != 32 || repository.created.Session.PepperVersion != "pepper-v1" {
		t.Fatalf("stored verifier = %d bytes / %q", len(repository.created.Session.TokenHMAC), repository.created.Session.PepperVersion)
	}
	if len(repository.created.Response.Ciphertext) == 0 || len(repository.created.Response.Nonce) == 0 {
		t.Fatal("idempotent response was not envelope encrypted")
	}
	if len(result.Session.TokenHMAC) != 0 {
		t.Fatal("response session leaked its verifier")
	}
	var envelope secretEnvelope
	if err := json.Unmarshal(result.CanonicalJSON, &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.ResourceID != testDelegatedSession || envelope.Kind != delegatedCredentialSecretKind ||
		!strings.HasPrefix(envelope.Secret, "vsd_"+testDelegatedSession+"_") {
		t.Fatalf("delegated secret envelope = %#v", envelope)
	}
	if strings.Contains(string(repository.created.Session.TokenHMAC), envelope.Secret) {
		t.Fatal("plaintext secret was persisted as the verifier")
	}
}

func TestReplayRejectsDifferentManagementSession(t *testing.T) {
	now := time.Date(2026, 8, 23, 8, 0, 0, 0, time.UTC)
	repository := &delegationRepositoryStub{now: now}
	service := newDelegationTestService(t, repository, &delegationWaiterStub{}, now)
	request := CreateRequest{
		NamespaceID: testDelegationNamespace, KeyID: testDelegationKey,
		IdempotencyKey: "delegation-replay-0001", Actor: testDelegationActor(testManagementSession),
	}
	if _, err := service.Create(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	repository.replay = true
	request.Actor = testDelegationActor(testOtherSession)
	if _, err := service.Create(context.Background(), request); !errors.Is(err, ErrCredentialInactive) {
		t.Fatalf("cross-session replay error = %v", err)
	}
}

func TestEligibleKeySearchIsCanonicalAndCursorBound(t *testing.T) {
	now := time.Date(2026, 8, 23, 8, 0, 0, 0, time.UTC)
	repository := &delegationRepositoryStub{now: now, eligiblePage: Page[EligibleKey]{
		Items: []EligibleKey{{KeyID: testDelegationKey, CreatedAt: now}}, HasMore: true,
	}}
	service := newDelegationTestService(t, repository, &delegationWaiterStub{}, now)
	request := ListRequest{
		NamespaceID: testDelegationNamespace, PrincipalID: testDelegationPrincipal,
		ManagementSessionID: testManagementSession, Search: "  Developer  ", PageSize: 1,
	}
	first, err := service.ListEligibleKeys(context.Background(), request)
	if err != nil || first.NextCursor == "" || repository.eligibleQuery.Search != "developer" ||
		repository.eligibleCalls != 1 {
		t.Fatalf("first eligible key page = %#v, query=%#v, calls=%d, error=%v",
			first, repository.eligibleQuery, repository.eligibleCalls, err)
	}
	request.Cursor, request.Search = first.NextCursor, "other"
	if _, err := service.ListEligibleKeys(context.Background(), request); !errors.Is(err, ErrInvalidRequest) ||
		repository.eligibleCalls != 1 {
		t.Fatalf("search-swapped cursor error=%v calls=%d", err, repository.eligibleCalls)
	}
}

func TestGetEligibleKeyHidesNonOwnedAndCrossNamespaceKeys(t *testing.T) {
	now := time.Date(2026, 8, 23, 8, 0, 0, 0, time.UTC)
	repository := &delegationRepositoryStub{now: now}
	service := newDelegationTestService(t, repository, &delegationWaiterStub{}, now)
	request := EligibleKeyRequest{
		NamespaceID: testDelegationNamespace, PrincipalID: testDelegationPrincipal,
		ManagementSessionID: testManagementSession, KeyID: testDelegationKey,
	}
	key, err := service.GetEligibleKey(context.Background(), request)
	if err != nil || key.KeyID != testDelegationKey {
		t.Fatalf("eligible key=%#v error=%v", key, err)
	}
	repository.eligibleErr = ErrNotEligible
	if _, err := service.GetEligibleKey(context.Background(), request); !errors.Is(err, ErrNotEligible) {
		t.Fatalf("non-owned key error=%v", err)
	}
	repository.eligibleErr = nil
	request.NamespaceID = testOtherSession
	if _, err := service.GetEligibleKey(context.Background(), request); !errors.Is(err, ErrNotEligible) {
		t.Fatalf("cross-namespace key error=%v", err)
	}
}

type delegationRepositoryStub struct {
	now               time.Time
	eligibleCalls     int
	eligibleNamespace string
	eligibleQuery     EligibleKeyQuery
	eligiblePage      Page[EligibleKey]
	eligibleErr       error
	created           CreateMutation
	session           Session
	stored            StoredSecret
	replay            bool
}

func (repository *delegationRepositoryStub) Ready(context.Context, *managementcommand.Codec) error {
	return nil
}

func (repository *delegationRepositoryStub) ResolveSelf(context.Context, string, string, string, bool) (SelfContext, error) {
	return SelfContext{
		NamespaceID: testDelegationNamespace, QuotaPartition: "partition-1",
		PrincipalID: testDelegationPrincipal, ManagementSessionID: testManagementSession,
		ManagementSessionExpires: repository.now.Add(time.Hour), UserID: testDelegationUser,
		Policy: SelfServicePolicy{MaxDelegatedSessions: 2, DelegatedSessionTTL: 15 * time.Minute, Revision: 1},
	}, nil
}

func (repository *delegationRepositoryStub) ListEligibleKeys(_ context.Context, query EligibleKeyQuery) (Page[EligibleKey], error) {
	repository.eligibleCalls++
	repository.eligibleQuery = query
	return repository.eligiblePage, repository.eligibleErr
}

func (repository *delegationRepositoryStub) GetEligibleKey(_ context.Context, namespaceID, _ string, _ string, _ string) (EligibleKey, error) {
	repository.eligibleNamespace = namespaceID
	if repository.eligibleErr != nil {
		return EligibleKey{}, repository.eligibleErr
	}
	if namespaceID != testDelegationNamespace {
		return EligibleKey{}, ErrNotEligible
	}
	return EligibleKey{
		KeyID: testDelegationKey, OwnerKind: accesscontrol.SubjectKindUser,
		OwnerID: testDelegationUser, DelegationEpoch: 3, CreatedAt: repository.now,
	}, nil
}

func (repository *delegationRepositoryStub) GetKey(context.Context, string, string) (accesscontrol.APIKey, error) {
	return accesscontrol.APIKey{}, nil
}

func (repository *delegationRepositoryStub) ListSessions(context.Context, SessionQuery) (Page[Session], error) {
	return Page[Session]{}, nil
}

func (repository *delegationRepositoryStub) GetSession(context.Context, string, string) (Session, error) {
	return repository.session, nil
}

func (repository *delegationRepositoryStub) ReplaySecret(context.Context, managementcommand.Command) (StoredSecret, bool, error) {
	return repository.stored, repository.replay, nil
}

func (repository *delegationRepositoryStub) Create(_ context.Context, mutation CreateMutation) (MutationResult, error) {
	repository.created = mutation
	repository.session = mutation.Session
	repository.session.TokenHMAC = nil
	repository.stored = StoredSecret{Result: managementcommand.ResourceResult{
		ResourceType: "delegated_inference_session", ResourceID: mutation.Session.ID,
		ResourceRevision: 1, ResponseStatus: 201,
	}, Secret: managementcommand.SecretResponse{
		Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
		KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
	}, DesiredRevision: 7}
	return MutationResult{Session: repository.session, DesiredRevision: 7}, nil
}

func (repository *delegationRepositoryStub) Revoke(context.Context, RevokeRequest) (MutationResult, error) {
	return MutationResult{}, ErrUnavailable
}

func (repository *delegationRepositoryStub) RevokeAll(context.Context, RevokeAllMutation) (RevokeAllResult, error) {
	return RevokeAllResult{}, ErrUnavailable
}

type delegationWaiterStub struct {
	active          bool
	desiredRevision uint64
}

func (waiter *delegationWaiterStub) WaitActive(_ context.Context, _ Session, desiredRevision uint64) error {
	waiter.active = true
	waiter.desiredRevision = desiredRevision
	return nil
}
func (*delegationWaiterStub) WaitApplied(context.Context, string, string, uint64) error { return nil }

func newDelegationTestService(
	t *testing.T,
	repository Repository,
	waiter PublicationWaiter,
	now time.Time,
) *Service {
	t.Helper()
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{ActiveVersion: "command-v1", Keys: map[string][]byte{
		"command-v1": []byte(strings.Repeat("c", 32)),
	}})
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewService(Options{
		Repository: repository, Waiter: waiter, Commands: commands,
		CursorKeyring: securitykeyring.Symmetric{ActiveVersion: "cursor-v1", Keys: map[string][]byte{
			"cursor-v1": []byte(strings.Repeat("u", 32)),
		}},
		DelegationPeppers: accesscredential.PepperKeyring{ActiveVersion: "pepper-v1", Keys: map[string][]byte{
			"pepper-v1": []byte(strings.Repeat("p", 32)),
		}},
		ResponseKEK: accesscredential.KEKKeyring{ActiveVersion: "response-v1", Keys: map[string][]byte{
			"response-v1": []byte(strings.Repeat("r", 32)),
		}},
		Audience: "vllm-sr-inference", IdempotencyTTL: time.Hour, SecretDeliveryTTL: 10 * time.Minute,
		Now: func() time.Time { return now }, NewID: func() string { return testDelegatedSession },
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { service.Close(); commands.Close() })
	return service
}

func testDelegationActor(managementSessionID string) Actor {
	return Actor{
		PrincipalID: testDelegationPrincipal, ManagementSessionID: managementSessionID,
		ActorChain: []string{testDelegationPrincipal}, RequestID: "request-1",
	}
}

var _ Repository = (*delegationRepositoryStub)(nil)
