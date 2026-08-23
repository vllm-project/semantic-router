package management

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	credentialListNamespace = "11111111-1111-4111-8111-111111111111"
	credentialListOne       = "22222222-2222-4222-8222-222222222222"
	credentialListTwo       = "33333333-3333-4333-8333-333333333333"
)

type credentialListRepository struct {
	Repository
	calls int
	last  accesspostgres.ProviderCredentialListRequest
}

func (repository *credentialListRepository) ListProviderCredentials(
	_ context.Context,
	_ accesscontrol.NamespaceID,
	request accesspostgres.ProviderCredentialListRequest,
) (accesspostgres.ProviderCredentialListResult, error) {
	repository.calls++
	repository.last = request
	now := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	id := credentialListOne
	hasMore := true
	if request.AfterID != "" {
		id = credentialListTwo
		hasMore = false
	}
	return accesspostgres.ProviderCredentialListResult{Credentials: []providercredential.Credential{{
		ID: id, NamespaceID: credentialListNamespace, Name: id, ProviderID: "openai",
		Status: providercredential.StatusActive, CreatedAt: now, UpdatedAt: now,
	}}, HasMore: hasMore}, nil
}

func TestListProviderCredentialsBindsScopeBeforeStablePagination(t *testing.T) {
	repository := &credentialListRepository{}
	cursors, testListProviderCredentialsBindsScopeBeforeStablePaginationErr := newCursorCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("c", 32))},
	})
	if testListProviderCredentialsBindsScopeBeforeStablePaginationErr != nil {
		t.Fatal(testListProviderCredentialsBindsScopeBeforeStablePaginationErr)
	}
	t.Cleanup(cursors.close)
	service := &Service{repository: repository, cursors: cursors}
	scope := providerCredentialResultScope(credentialListOne, credentialListTwo)
	first, testListProviderCredentialsBindsScopeBeforeStablePaginationErr := service.List(context.Background(), ListRequest{
		NamespaceID: credentialListNamespace, PageSize: 1, Scope: scope,
	})
	if testListProviderCredentialsBindsScopeBeforeStablePaginationErr != nil || len(first.Credentials) != 1 || !first.HasMore || first.NextCursor == "" || repository.calls != 1 {
		t.Fatalf("first page = %#v, calls = %d, error = %v", first, repository.calls, testListProviderCredentialsBindsScopeBeforeStablePaginationErr)
	}
	if _, err := service.List(context.Background(), ListRequest{
		NamespaceID: credentialListNamespace, PageSize: 1,
		Scope: providerCredentialResultScope(credentialListOne), Cursor: first.NextCursor,
	}); !errors.Is(err, ErrInvalidRequest) || repository.calls != 1 {
		t.Fatalf("scope-swapped cursor error = %v, calls = %d", err, repository.calls)
	}
	second, testListProviderCredentialsBindsScopeBeforeStablePaginationErr := service.List(context.Background(), ListRequest{
		NamespaceID: credentialListNamespace, PageSize: 1, Scope: scope, Cursor: first.NextCursor,
	})
	if testListProviderCredentialsBindsScopeBeforeStablePaginationErr != nil || len(second.Credentials) != 1 || second.Credentials[0].CredentialID != credentialListTwo ||
		second.HasMore || repository.calls != 2 {
		t.Fatalf("second page = %#v, calls = %d, error = %v", second, repository.calls, testListProviderCredentialsBindsScopeBeforeStablePaginationErr)
	}
	ids := repository.last.Scope.IDs(accesscontrol.ScopeResourceProviderCredential)
	if len(ids) != 2 {
		t.Fatalf("repository scope = %#v", repository.last.Scope)
	}
	empty, testListProviderCredentialsBindsScopeBeforeStablePaginationErr := service.List(context.Background(), ListRequest{
		NamespaceID: credentialListNamespace, PageSize: 29,
		Scope: accesscontrol.ResultScope{
			NamespaceID: credentialListNamespace,
			APIKeyIDs:   []accesscontrol.APIKeyID{credentialListOne},
		},
	})
	if testListProviderCredentialsBindsScopeBeforeStablePaginationErr != nil || len(empty.Credentials) != 0 || empty.PageSize != 29 || repository.calls != 2 {
		t.Fatalf("empty page = %#v, calls = %d, error = %v", empty, repository.calls, testListProviderCredentialsBindsScopeBeforeStablePaginationErr)
	}
}

func providerCredentialResultScope(ids ...accesscontrol.ResourceID) accesscontrol.ResultScope {
	return accesscontrol.ResultScope{
		NamespaceID: credentialListNamespace,
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceProviderCredential: ids,
		},
	}
}
