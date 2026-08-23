package namespacemanagement

import (
	"context"
	"errors"
	"net/netip"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	namespaceTestID        = "11111111-1111-4111-8111-111111111111"
	namespaceTestPrincipal = "22222222-2222-4222-8222-222222222222"
	namespaceTestSession   = "33333333-3333-4333-8333-333333333333"
	namespaceTestSource    = "44444444-4444-4444-8444-444444444444"
	namespaceTestOther     = "55555555-5555-4555-8555-555555555555"
)

type namespaceRepositoryStub struct {
	Repository
	created       CreateNamespaceMutation
	security      ManagementSecurityPolicy
	securityPatch ManagementSecurityPolicy
	listCalls     int
	lastQuery     NamespaceQuery
}

func (repository *namespaceRepositoryStub) Replay(
	context.Context,
	managementcommand.Command,
) (MutationResult, bool, error) {
	return MutationResult{}, false, nil
}

func (repository *namespaceRepositoryStub) CreateNamespace(
	_ context.Context,
	mutation CreateNamespaceMutation,
) (MutationResult, error) {
	repository.created = mutation
	return MutationResult{
		Kind:       "namespace",
		ID:         mutation.Namespace.ID,
		Revision:   mutation.Namespace.Revision,
		HTTPStatus: 201,
	}, nil
}

func (repository *namespaceRepositoryStub) ListNamespaces(
	_ context.Context,
	query NamespaceQuery,
) (RepositoryPage[Namespace], error) {
	repository.listCalls++
	repository.lastQuery = query
	if query.After == nil {
		return RepositoryPage[Namespace]{
			Items: []Namespace{{
				ID:        namespaceTestID,
				CreatedAt: time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC),
			}},
			HasMore: true,
		}, nil
	}
	return RepositoryPage[Namespace]{
		Items: []Namespace{{
			ID:        namespaceTestOther,
			CreatedAt: time.Date(2026, 8, 23, 11, 0, 0, 0, time.UTC),
		}},
	}, nil
}

func (repository *namespaceRepositoryStub) GetManagementSecurityPolicy(
	context.Context,
	string,
) (ManagementSecurityPolicy, error) {
	return repository.security, nil
}

func (repository *namespaceRepositoryStub) PatchManagementSecurityPolicy(
	_ context.Context,
	policy ManagementSecurityPolicy,
	expected uint64,
	_ Actor,
) (MutationResult, error) {
	if expected != repository.security.Revision {
		return MutationResult{}, ErrRevisionConflict
	}
	repository.securityPatch = policy
	return MutationResult{
		Kind:       "management_security_policy",
		ID:         policy.NamespaceID,
		Revision:   expected + 1,
		HTTPStatus: 200,
	}, nil
}

func TestCreateNamespaceBuildsRestrictiveAtomicAggregate(t *testing.T) {
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	repository := &namespaceRepositoryStub{}
	service := newNamespaceTestService(t, repository, now)

	result, err := service.CreateNamespace(context.Background(), CreateNamespaceRequest{
		Name:            "Default",
		BillingCurrency: "USD",
		IdempotencyKey:  "namespace-create-0001",
		Actor:           namespaceTestActor("Create the default namespace"),
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.ID != namespaceTestID || result.Revision != 1 || result.HTTPStatus != 201 {
		t.Fatalf("create result = %#v", result)
	}

	mutation := repository.created
	if mutation.Namespace.ID != namespaceTestID ||
		mutation.Namespace.QuotaPartitionID != namespaceTestID ||
		mutation.Namespace.Status != accesscontrol.NamespaceStatusActive ||
		mutation.Namespace.RuntimeEpoch != 1 ||
		mutation.Command.Scope.Kind != managementcommand.ScopeCluster {
		t.Fatalf("Namespace aggregate = %#v", mutation)
	}
	policy := mutation.SelfService
	if policy.MaxKeysPerUser != 0 || policy.MaxDelegatedSessions != 0 ||
		policy.AllowTeamKeyDelegation || policy.AutomaticFirstKey ||
		policy.DefaultAccessPolicyID != "" || policy.DefaultRateLimitPolicyID != "" ||
		len(policy.TeamAdminCapabilities) != 0 || policy.DelegatedSessionTTL != 15*time.Minute {
		t.Fatalf("initial self-service policy is not restrictive: %#v", policy)
	}
	if mutation.Security.SeedVersion != SecurityPolicySeedVersion ||
		len(mutation.Security.ActionRequirements) != 5 {
		t.Fatalf("initial security policy = %#v", mutation.Security)
	}
	if len(mutation.RoutingClaims.Definitions) != 0 || mutation.RoutingClaims.Revision != 1 {
		t.Fatalf("initial routing claim schema = %#v", mutation.RoutingClaims)
	}
}

func TestNamespaceCursorBindsAuthorizedResultScope(t *testing.T) {
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	repository := &namespaceRepositoryStub{}
	service := newNamespaceTestService(t, repository, now)
	scope := ResultScope{NamespaceIDs: []string{namespaceTestID, namespaceTestOther}}

	first, err := service.ListNamespaces(context.Background(), ListRequest{
		Scope: scope, PageSize: 1,
	})
	if err != nil || len(first.Items) != 1 || !first.HasMore || first.NextCursor == "" {
		t.Fatalf("first page = %#v, error = %v", first, err)
	}
	if repository.listCalls != 1 || repository.lastQuery.Scope.All ||
		len(repository.lastQuery.Scope.NamespaceIDs) != 2 {
		t.Fatalf("repository query = %#v", repository.lastQuery)
	}

	_, err = service.ListNamespaces(context.Background(), ListRequest{
		Scope:    ResultScope{NamespaceIDs: []string{namespaceTestID}},
		PageSize: 1,
		Cursor:   first.NextCursor,
	})
	if !errors.Is(err, ErrInvalidRequest) || repository.listCalls != 1 {
		t.Fatalf("scope-swapped cursor error = %v, calls = %d", err, repository.listCalls)
	}

	second, err := service.ListNamespaces(context.Background(), ListRequest{
		Scope: scope, PageSize: 1, Cursor: first.NextCursor,
	})
	if err != nil || len(second.Items) != 1 || second.Items[0].ID != namespaceTestOther ||
		second.HasMore || repository.listCalls != 2 {
		t.Fatalf("second page = %#v, calls = %d, error = %v", second, repository.listCalls, err)
	}
}

func TestSecurityPolicyWideningRequiresCurrentStrongSession(t *testing.T) {
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	current := ManagementSecurityPolicy{
		NamespaceID:        namespaceTestID,
		ActionRequirements: restrictiveSecurityRequirements(),
		SeedVersion:        SecurityPolicySeedVersion,
		Revision:           3,
		UpdatedAt:          now.Add(-time.Hour),
	}
	repository := &namespaceRepositoryStub{security: current}
	service := newNamespaceTestService(t, repository, now)
	target := cloneRequirements(current.ActionRequirements)
	target[ActionSecretReveal] = managementauth.ActionRequirement{
		AnyOf: []managementauth.AuthenticationRequirement{{
			Kind: managementauth.RequirementHuman,
			Human: &managementauth.HumanRequirement{
				MinimumAAL:                  "aal1",
				AcceptedAMR:                 []string{},
				MaxAuthenticationAgeSeconds: 900,
			},
		}},
	}
	request := PatchManagementSecurityPolicyRequest{
		NamespaceID:        namespaceTestID,
		ExpectedRevision:   3,
		ActionRequirements: target,
		Actor:              namespaceTestActor("Permit lower-assurance reveal"),
	}

	if _, err := service.PatchManagementSecurityPolicy(context.Background(), request); !errors.Is(err, ErrAssurance) {
		t.Fatalf("missing strong session error = %v", err)
	}
	request.Session = namespaceStrongHumanSession(now)
	result, err := service.PatchManagementSecurityPolicy(context.Background(), request)
	if err != nil || result.Revision != 4 || repository.securityPatch.SeedVersion != SecurityPolicySeedVersion {
		t.Fatalf("authorized policy widening = %#v, patch = %#v, error = %v", result, repository.securityPatch, err)
	}
}

func newNamespaceTestService(
	t *testing.T,
	repository Repository,
	now time.Time,
) *Service {
	t.Helper()
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys: map[string][]byte{
			"v1": []byte(strings.Repeat("c", 32)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	service, err := NewService(Options{
		Repository:   repository,
		CommandCodec: commands,
		CursorKeyring: securitykeyring.Symmetric{
			ActiveVersion: "v1",
			Keys: map[string][]byte{
				"v1": []byte(strings.Repeat("p", 32)),
			},
		},
		IdempotencyTTL: time.Hour,
		Now:            func() time.Time { return now },
		NewID:          func() string { return namespaceTestID },
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	t.Cleanup(func() { _ = commands.Close() })
	return service
}

func namespaceTestActor(reason string) Actor {
	return Actor{
		PrincipalID: namespaceTestPrincipal,
		ActorChain:  []string{namespaceTestPrincipal},
		RequestID:   "namespace-test-request",
		SourceIP:    netip.MustParseAddr("192.0.2.40"),
		Reason:      reason,
	}
}

func namespaceStrongHumanSession(now time.Time) managementauth.LiveSession {
	return managementauth.LiveSession{
		Session: managementauth.Session{
			ID:             namespaceTestSession,
			PrincipalID:    namespaceTestPrincipal,
			TokenID:        "namespace-test-token",
			Audience:       "management",
			AuthSourceKind: managementauth.AuthSourceIssuer,
			AuthSourceID:   namespaceTestSource,
			EvidenceKind:   managementauth.EvidenceHuman,
			Human: &managementauth.HumanEvidence{
				AuthenticationTime: now.Unix(),
				AAL:                "aal2",
				AMR:                []string{"pwd", "otp"},
			},
			AuthenticatedAt: now,
			ExpiresAt:       now.Add(time.Hour),
			Status:          managementauth.SessionActive,
			CreatedAt:       now,
		},
		PrincipalStatus:  managementauth.ResourceActive,
		AuthSourceStatus: managementauth.ResourceActive,
	}
}
