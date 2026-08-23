package policybulk

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
	operationListNamespace = "11111111-1111-4111-8111-111111111111"
	operationListPrincipal = "22222222-2222-4222-8222-222222222222"
	operationListPolicyOne = "33333333-3333-4333-8333-333333333333"
	operationListPolicyTwo = "44444444-4444-4444-8444-444444444444"
	operationListOne       = "55555555-5555-4555-8555-555555555555"
	operationListTwo       = "66666666-6666-4666-8666-666666666666"
)

type operationListRepository struct {
	Repository
	pages []RepositoryPage
	calls []OperationQuery
}

func (repository *operationListRepository) List(_ context.Context, query OperationQuery) (RepositoryPage, error) {
	repository.calls = append(repository.calls, query)
	if len(repository.pages) == 0 {
		return RepositoryPage{}, nil
	}
	page := repository.pages[0]
	repository.pages = repository.pages[1:]
	return page, nil
}

func TestOperationListCursorBindsCompleteVisibilityAndKeepsKeysetStable(t *testing.T) {
	created := time.Date(2026, 8, 23, 1, 0, 0, 0, time.UTC)
	repository := &operationListRepository{pages: []RepositoryPage{
		{Items: []Operation{{ID: operationListOne, CreatedAt: created}}, HasMore: true},
		{Items: []Operation{{ID: operationListTwo, CreatedAt: created.Add(-time.Second)}}},
	}}
	service := operationListService(t, repository)
	visibility := operationListVisibility(operationListPolicyOne, operationListOne)

	first, err := service.List(context.Background(), ListRequest{
		NamespaceID: operationListNamespace, PageSize: 1, Visibility: visibility,
	})
	if err != nil || !first.HasMore || first.NextCursor == "" || len(first.Items) != 1 || first.Items[0].ID != operationListOne {
		t.Fatalf("first page = %#v, error = %v", first, err)
	}

	swapped := operationListVisibility(operationListPolicyOne, operationListTwo)
	_, err = service.List(context.Background(), ListRequest{
		NamespaceID: operationListNamespace, PageSize: 1, Visibility: swapped, Cursor: first.NextCursor,
	})
	if !errors.Is(err, ErrInvalidRequest) || len(repository.calls) != 1 {
		t.Fatalf("scope-swapped cursor error/calls = %v/%d", err, len(repository.calls))
	}

	second, err := service.List(context.Background(), ListRequest{
		NamespaceID: operationListNamespace, PageSize: 1, Visibility: visibility, Cursor: first.NextCursor,
	})
	if err != nil || second.HasMore || second.NextCursor != "" || len(second.Items) != 1 ||
		second.Items[0].ID != operationListTwo || len(repository.calls) != 2 || repository.calls[1].After == nil ||
		repository.calls[1].After.ID != operationListOne {
		t.Fatalf("second page = %#v, calls = %#v, error = %v", second, repository.calls, err)
	}
}

func TestOperationListEmptyDomainScopeAvoidsRepository(t *testing.T) {
	repository := &operationListRepository{}
	service := operationListService(t, repository)
	empty := OperationVisibility{
		PrincipalID: operationListPrincipal,
		Operation: accesscontrol.ResultScope{
			NamespaceID: accesscontrol.NamespaceID(operationListNamespace),
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceOperation: {operationListOne},
			},
		},
		Access: accesscontrol.ResultScope{NamespaceID: accesscontrol.NamespaceID(operationListNamespace)},
		Rate:   accesscontrol.ResultScope{NamespaceID: accesscontrol.NamespaceID(operationListNamespace)},
	}
	page, err := service.List(context.Background(), ListRequest{
		NamespaceID: operationListNamespace, PageSize: 25, Visibility: empty,
	})
	if err != nil || page.Items == nil || len(page.Items) != 0 || page.PageSize != 25 || len(repository.calls) != 0 {
		t.Fatalf("empty page = %#v, calls = %d, error = %v", page, len(repository.calls), err)
	}
}

func operationListVisibility(policyID, operationID string) OperationVisibility {
	namespaceID := accesscontrol.NamespaceID(operationListNamespace)
	return OperationVisibility{
		PrincipalID: operationListPrincipal,
		Operation: accesscontrol.ResultScope{
			NamespaceID: namespaceID,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceOperation: {accesscontrol.ResourceID(operationID)},
			},
		},
		Access: accesscontrol.ResultScope{
			NamespaceID: namespaceID,
			ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
				accesscontrol.ScopeResourceAccessPolicy: {accesscontrol.ResourceID(policyID)},
			},
		},
		Rate: accesscontrol.ResultScope{NamespaceID: namespaceID},
	}
}

func operationListService(t *testing.T, repository Repository) *Service {
	t.Helper()
	cursors, err := newOperationCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("c", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(cursors.close)
	return &Service{repository: repository, cursors: cursors}
}
