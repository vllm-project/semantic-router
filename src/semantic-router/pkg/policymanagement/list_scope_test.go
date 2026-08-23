package policymanagement

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	policyListNamespace = "11111111-1111-4111-8111-111111111111"
	policyListOne       = "22222222-2222-4222-8222-222222222222"
	policyListTwo       = "33333333-3333-4333-8333-333333333333"
	policyBindingOne    = "44444444-4444-4444-8444-444444444444"
	rateBindingOne      = "55555555-5555-4555-8555-555555555555"
)

type policyListRepository struct {
	Repository
	policyCalls      int
	rateCalls        int
	bindingCalls     int
	rateBindingCalls int
	lastPolicy       PolicyQuery
	lastRate         PolicyQuery
	lastBinding      BindingQuery
}

func (repository *policyListRepository) ListRateLimitPolicies(
	_ context.Context,
	query PolicyQuery,
) (RepositoryPage[RateLimitPolicy], error) {
	repository.rateCalls++
	repository.lastRate = query
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	if query.After == nil {
		return RepositoryPage[RateLimitPolicy]{Items: []RateLimitPolicy{{
			ID: policyListOne, NamespaceID: policyListNamespace, CreatedAt: now, UpdatedAt: now,
		}}, HasMore: true}, nil
	}
	return RepositoryPage[RateLimitPolicy]{Items: []RateLimitPolicy{{
		ID: policyListTwo, NamespaceID: policyListNamespace, CreatedAt: now.Add(-time.Second), UpdatedAt: now,
	}}}, nil
}

func (repository *policyListRepository) ListAccessPolicies(
	_ context.Context,
	query PolicyQuery,
) (RepositoryPage[AccessPolicy], error) {
	repository.policyCalls++
	repository.lastPolicy = query
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	if query.After == nil {
		return RepositoryPage[AccessPolicy]{Items: []AccessPolicy{{
			ID: policyListOne, NamespaceID: policyListNamespace, CreatedAt: now, UpdatedAt: now,
		}}, HasMore: true}, nil
	}
	return RepositoryPage[AccessPolicy]{Items: []AccessPolicy{{
		ID: policyListTwo, NamespaceID: policyListNamespace, CreatedAt: now.Add(-time.Second), UpdatedAt: now,
	}}}, nil
}

func (repository *policyListRepository) ListAccessBindings(
	_ context.Context,
	query BindingQuery,
) (RepositoryPage[AccessPolicyBinding], error) {
	repository.bindingCalls++
	repository.lastBinding = query
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	return RepositoryPage[AccessPolicyBinding]{Items: []AccessPolicyBinding{{
		ID: policyBindingOne, NamespaceID: policyListNamespace, PolicyID: policyListOne,
		CreatedAt: now, UpdatedAt: now,
	}}, HasMore: true}, nil
}

func (repository *policyListRepository) ListRateBindings(
	_ context.Context,
	query BindingQuery,
) (RepositoryPage[RateLimitBinding], error) {
	repository.rateBindingCalls++
	repository.lastBinding = query
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	return RepositoryPage[RateLimitBinding]{Items: []RateLimitBinding{{
		ID: rateBindingOne, NamespaceID: policyListNamespace, PolicyID: policyListOne,
		CreatedAt: now, UpdatedAt: now,
	}}, HasMore: true}, nil
}

func TestListPoliciesBindsScopeBeforeStablePagination(t *testing.T) {
	repository := &policyListRepository{}
	service := policyListService(t, repository)
	scope := accessPolicyResultScope(policyListOne, policyListTwo)
	first, listAccessPoliciesErr := service.ListAccessPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, Search: "  Dev  ", PageSize: 1, Scope: scope,
	})
	if listAccessPoliciesErr != nil || len(first.Items) != 1 || !first.HasMore || first.NextCursor == "" || repository.policyCalls != 1 {
		t.Fatalf("first page = %#v, calls = %d, error = %v", first, repository.policyCalls, listAccessPoliciesErr)
	}
	if _, err := service.ListAccessPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, PageSize: 1, Scope: accessPolicyResultScope(policyListOne),
		Search: "dev", Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.policyCalls != 1 {
		t.Fatalf("scope-swapped cursor error = %v, calls = %d", err, repository.policyCalls)
	}
	if _, err := service.ListAccessPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, Search: "prod", PageSize: 1, Scope: scope,
		Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.policyCalls != 1 {
		t.Fatalf("search-swapped cursor error = %v, calls = %d", err, repository.policyCalls)
	}
	second, listAccessPoliciesErr := service.ListAccessPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, Search: "dev", PageSize: 1, Scope: scope, Cursor: first.NextCursor,
	})
	if listAccessPoliciesErr != nil || len(second.Items) != 1 || second.Items[0].ID != policyListTwo ||
		second.HasMore || repository.policyCalls != 2 {
		t.Fatalf("second page = %#v, calls = %d, error = %v", second, repository.policyCalls, listAccessPoliciesErr)
	}
	ids := repository.lastPolicy.Scope.IDs(accesscontrol.ScopeResourceAccessPolicy)
	if len(ids) != 2 || repository.lastPolicy.Search != "dev" {
		t.Fatalf("repository policy scope = %#v", repository.lastPolicy.Scope)
	}
	empty, listAccessPoliciesErr := service.ListAccessPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, PageSize: 23,
		Scope: accesscontrol.ResultScope{
			NamespaceID: policyListNamespace,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceAccessPolicyBinding: {policyBindingOne},
			},
		},
	})
	if listAccessPoliciesErr != nil || len(empty.Items) != 0 || empty.PageSize != 23 || repository.policyCalls != 2 {
		t.Fatalf("empty page = %#v, calls = %d, error = %v", empty, repository.policyCalls, listAccessPoliciesErr)
	}
}

func TestListRatePoliciesBindsSearchToSecondPage(t *testing.T) {
	repository := &policyListRepository{}
	service := policyListService(t, repository)
	scope := accesscontrol.ResultScope{
		NamespaceID: policyListNamespace,
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceRateLimitPolicy: {policyListOne, policyListTwo},
		},
	}
	first, listRateLimitPoliciesErr := service.ListRateLimitPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, Search: "  Hourly  ", PageSize: 1, Scope: scope,
	})
	if listRateLimitPoliciesErr != nil || first.NextCursor == "" || !first.HasMore {
		t.Fatalf("first RateLimitPolicy page = %#v, error = %v", first, listRateLimitPoliciesErr)
	}
	if _, err := service.ListRateLimitPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, Search: "daily", PageSize: 1, Scope: scope, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.rateCalls != 1 {
		t.Fatalf("search-swapped RateLimitPolicy cursor error = %v, calls = %d", err, repository.rateCalls)
	}
	second, listRateLimitPoliciesErr := service.ListRateLimitPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, Search: "hourly", PageSize: 1, Scope: scope, Cursor: first.NextCursor,
	})
	if listRateLimitPoliciesErr != nil || len(second.Items) != 1 || second.Items[0].ID != policyListTwo ||
		second.HasMore || repository.lastRate.Search != "hourly" {
		t.Fatalf("second RateLimitPolicy page = %#v, query = %#v, error = %v", second, repository.lastRate, listRateLimitPoliciesErr)
	}
}

func TestListBindingsUsesAssociatedPolicyScopeAndBindsCursor(t *testing.T) {
	repository := &policyListRepository{}
	service := policyListService(t, repository)
	scope := accessPolicyResultScope(policyListOne)
	first, err := service.ListAccessBindings(context.Background(), ListBindingsRequest{
		NamespaceID: policyListNamespace, PageSize: 1, Scope: scope,
	})
	if err != nil || !first.HasMore || first.NextCursor == "" || repository.bindingCalls != 1 {
		t.Fatalf("first binding page = %#v, calls = %d, error = %v", first, repository.bindingCalls, err)
	}
	if _, err := service.ListAccessBindings(context.Background(), ListBindingsRequest{
		NamespaceID: policyListNamespace, PageSize: 1, Scope: accessPolicyResultScope(policyListTwo),
		Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.bindingCalls != 1 {
		t.Fatalf("binding scope-swap error = %v, calls = %d", err, repository.bindingCalls)
	}
	ids := repository.lastBinding.Scope.IDs(accesscontrol.ScopeResourceAccessPolicy)
	if len(ids) != 1 || ids[0] != policyListOne ||
		len(repository.lastBinding.Scope.IDs(accesscontrol.ScopeResourceAccessPolicyBinding)) != 0 {
		t.Fatalf("repository binding scope = %#v", repository.lastBinding.Scope)
	}
}

func TestRateListsConsumeOnlyRatePolicyScope(t *testing.T) {
	repository := &policyListRepository{}
	service := policyListService(t, repository)
	scope := ratePolicyResultScope(policyListOne, policyListTwo)
	first, testRateListsConsumeOnlyRatePolicyScopeErr := service.ListRateLimitPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, PageSize: 1, Scope: scope,
	})
	if testRateListsConsumeOnlyRatePolicyScopeErr != nil || !first.HasMore || first.NextCursor == "" || repository.rateCalls != 1 {
		t.Fatalf("first RateLimitPolicy page = %#v, calls = %d, error = %v", first, repository.rateCalls, testRateListsConsumeOnlyRatePolicyScopeErr)
	}
	if _, err := service.ListRateLimitPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, PageSize: 1, Cursor: first.NextCursor,
		Scope: ratePolicyResultScope(policyListOne),
	}); !errors.Is(err, ErrInvalidRequest) || repository.rateCalls != 1 {
		t.Fatalf("RateLimitPolicy scope-swap error = %v, calls = %d", err, repository.rateCalls)
	}
	binding, testRateListsConsumeOnlyRatePolicyScopeErr := service.ListRateBindings(context.Background(), ListBindingsRequest{
		NamespaceID: policyListNamespace, PageSize: 1, Scope: ratePolicyResultScope(policyListOne),
	})
	if testRateListsConsumeOnlyRatePolicyScopeErr != nil || !binding.HasMore || binding.NextCursor == "" || repository.rateBindingCalls != 1 ||
		len(repository.lastBinding.Scope.IDs(accesscontrol.ScopeResourceRateLimitPolicy)) != 1 {
		t.Fatalf("RateLimit binding page = %#v, query = %#v, error = %v", binding, repository.lastBinding, testRateListsConsumeOnlyRatePolicyScopeErr)
	}
	empty, testRateListsConsumeOnlyRatePolicyScopeErr := service.ListRateLimitPolicies(context.Background(), ListPoliciesRequest{
		NamespaceID: policyListNamespace, PageSize: 7, Scope: accessPolicyResultScope(policyListOne),
	})
	if testRateListsConsumeOnlyRatePolicyScopeErr != nil || empty.Items == nil || len(empty.Items) != 0 || repository.rateCalls != 1 {
		t.Fatalf("irrelevant AccessPolicy scope page = %#v, calls = %d, error = %v", empty, repository.rateCalls, testRateListsConsumeOnlyRatePolicyScopeErr)
	}
}

func accessPolicyResultScope(ids ...accesscontrol.ResourceID) accesscontrol.ResultScope {
	return accesscontrol.ResultScope{
		NamespaceID: policyListNamespace,
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceAccessPolicy: ids,
		},
	}
}

func ratePolicyResultScope(ids ...accesscontrol.ResourceID) accesscontrol.ResultScope {
	return accesscontrol.ResultScope{
		NamespaceID: policyListNamespace,
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceRateLimitPolicy: ids,
		},
	}
}

func policyListService(t *testing.T, repository Repository) *Service {
	t.Helper()
	cursors, err := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("p", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(cursors.close)
	return &Service{repository: repository, cursors: cursors}
}
