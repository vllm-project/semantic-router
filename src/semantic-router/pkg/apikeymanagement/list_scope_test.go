package apikeymanagement

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	keyListNamespace = "11111111-1111-4111-8111-111111111111"
	keyListOne       = "22222222-2222-4222-8222-222222222222"
	keyListTwo       = "33333333-3333-4333-8333-333333333333"
)

type keyListRepository struct {
	Repository
	calls int
	last  KeyQuery
}

func (repository *keyListRepository) ListKeys(
	_ context.Context,
	query KeyQuery,
) (RepositoryPage[accesscontrol.APIKey], error) {
	repository.calls++
	repository.last = query
	var totalCount *uint64
	if query.IncludeTotal {
		count := uint64(2)
		totalCount = &count
	}
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	if query.After == nil {
		return RepositoryPage[accesscontrol.APIKey]{Items: []accesscontrol.APIKey{{
			NamespaceID: keyListNamespace, ID: keyListOne, CreatedAt: now, UpdatedAt: now,
		}}, HasMore: true, TotalCount: totalCount}, nil
	}
	return RepositoryPage[accesscontrol.APIKey]{Items: []accesscontrol.APIKey{{
		NamespaceID: keyListNamespace, ID: keyListTwo, CreatedAt: now.Add(-time.Second), UpdatedAt: now,
	}}, TotalCount: totalCount}, nil
}

func TestListKeysBindsScopeBeforeStablePagination(t *testing.T) {
	repository := &keyListRepository{}
	cursors, testListKeysBindsScopeBeforeStablePaginationErr := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
	})
	if testListKeysBindsScopeBeforeStablePaginationErr != nil {
		t.Fatal(testListKeysBindsScopeBeforeStablePaginationErr)
	}
	t.Cleanup(cursors.close)
	service := &Service{repository: repository, cursors: cursors}
	scope := accesscontrol.ResultScope{
		NamespaceID: keyListNamespace,
		APIKeyIDs:   []accesscontrol.APIKeyID{keyListOne, keyListTwo},
	}
	first, testListKeysBindsScopeBeforeStablePaginationErr := service.List(context.Background(), ListKeysRequest{
		NamespaceID: keyListNamespace, Search: "  Prod  ", PageSize: 1, IncludeTotal: true, Scope: scope,
	})
	if testListKeysBindsScopeBeforeStablePaginationErr != nil || len(first.Items) != 1 || !first.HasMore ||
		first.NextCursor == "" || repository.calls != 1 || first.TotalCount == nil || *first.TotalCount != 2 {
		t.Fatalf("first page = %#v, calls = %d, error = %v", first, repository.calls, testListKeysBindsScopeBeforeStablePaginationErr)
	}
	swapped := accesscontrol.ResultScope{
		NamespaceID: keyListNamespace,
		APIKeyIDs:   []accesscontrol.APIKeyID{keyListOne},
	}
	if _, err := service.List(context.Background(), ListKeysRequest{
		NamespaceID: keyListNamespace, Search: "prod", PageSize: 1, Scope: swapped, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.calls != 1 {
		t.Fatalf("scope-swapped cursor error = %v, calls = %d", err, repository.calls)
	}
	if _, err := service.List(context.Background(), ListKeysRequest{
		NamespaceID: keyListNamespace, Search: "staging", PageSize: 1, Scope: scope, Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.calls != 1 {
		t.Fatalf("search-swapped cursor error = %v, calls = %d", err, repository.calls)
	}
	second, testListKeysBindsScopeBeforeStablePaginationErr := service.List(context.Background(), ListKeysRequest{
		NamespaceID: keyListNamespace, Search: "prod", PageSize: 1, IncludeTotal: true,
		Scope: scope, Cursor: first.NextCursor,
	})
	if testListKeysBindsScopeBeforeStablePaginationErr != nil || len(second.Items) != 1 || second.Items[0].ID != keyListTwo ||
		second.HasMore || repository.calls != 2 || second.TotalCount == nil || *second.TotalCount != 2 {
		t.Fatalf("second page = %#v, calls = %d, error = %v", second, repository.calls, testListKeysBindsScopeBeforeStablePaginationErr)
	}
	if repository.last.Scope.All || len(repository.last.Scope.APIKeyIDs) != 2 || repository.last.Search != "prod" {
		t.Fatalf("repository scope = %#v", repository.last.Scope)
	}
	empty, testListKeysBindsScopeBeforeStablePaginationErr := service.List(context.Background(), ListKeysRequest{
		NamespaceID: keyListNamespace, PageSize: 19, IncludeTotal: true,
		Scope: accesscontrol.ResultScope{
			NamespaceID: keyListNamespace,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceProviderCredential: {keyListOne},
			},
		},
	})
	if testListKeysBindsScopeBeforeStablePaginationErr != nil || len(empty.Items) != 0 || empty.PageSize != 19 ||
		empty.TotalCount == nil || *empty.TotalCount != 0 || repository.calls != 2 {
		t.Fatalf("empty page = %#v, calls = %d, error = %v", empty, repository.calls, testListKeysBindsScopeBeforeStablePaginationErr)
	}
}

func TestListKeysKeepsTenThousandKeyVisibilityBehindBoundedPages(t *testing.T) {
	const visibleKeyCount = 10_000
	repository := &keyListRepository{}
	cursors, err := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("k", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(cursors.close)
	service := &Service{repository: repository, cursors: cursors}
	visible := make([]accesscontrol.APIKeyID, visibleKeyCount)
	for index := range visible {
		visible[index] = accesscontrol.APIKeyID(fmt.Sprintf("%08x-0000-4000-8000-%012x", index+1, index+1))
	}
	scope := accesscontrol.ResultScope{NamespaceID: keyListNamespace, APIKeyIDs: visible}

	page, err := service.List(context.Background(), ListKeysRequest{
		NamespaceID: keyListNamespace, PageSize: maximumPageSize, Scope: scope,
	})
	if err != nil || page.PageSize != maximumPageSize || repository.calls != 1 {
		t.Fatalf("large-scope page = %#v, calls = %d, error = %v", page, repository.calls, err)
	}
	if repository.last.Limit != maximumPageSize || len(repository.last.Scope.APIKeyIDs) != visibleKeyCount {
		t.Fatalf("large-scope query limit/visibility = %d/%d, want %d/%d",
			repository.last.Limit, len(repository.last.Scope.APIKeyIDs), maximumPageSize, visibleKeyCount)
	}
	if _, err := service.List(context.Background(), ListKeysRequest{
		NamespaceID: keyListNamespace, PageSize: maximumPageSize + 1, Scope: scope,
	}); !errors.Is(err, ErrInvalidRequest) || repository.calls != 1 {
		t.Fatalf("oversized page error = %v, repository calls = %d", err, repository.calls)
	}
}
