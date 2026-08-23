package agentmanagement

import (
	"errors"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestResourceListCursorBindsNormalizedSearch(t *testing.T) {
	codec, err := newSignedCodec(securitykeyring.Symmetric{
		ActiveVersion: "cursor-v1",
		Keys:          map[string][]byte{"cursor-v1": []byte("0123456789abcdef0123456789abcdef")},
	})
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{codec: codec}
	t.Cleanup(service.Close)
	namespaceID := uuid.NewString()
	scope := accesscontrol.ResultScope{
		NamespaceID: accesscontrol.NamespaceID(namespaceID),
		All:         true,
	}
	scopeDigest, err := scope.Digest()
	if err != nil {
		t.Fatal(err)
	}
	ownerPrincipalID := uuid.NewString()
	query, err := service.listQuery(namespaceID, "profiles", PageRequest{
		PageSize: 20, Search: "  RECIPE  ", Scope: scope, OwnerPrincipalID: ownerPrincipalID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if query.Search != "recipe" {
		t.Fatalf("normalized search = %q", query.Search)
	}
	page, err := makePage(service, namespaceID, "profiles", query, 20, []Profile{{
		ResourceIdentity: ResourceIdentity{
			ID: uuid.NewString(), CreatedAt: time.Date(2026, time.August, 23, 12, 0, 0, 0, time.UTC),
		},
	}}, true, nil)
	if err != nil || page.NextCursor == "" {
		t.Fatalf("page = %+v, error = %v", page, err)
	}
	decoded, err := codec.decodeCursor(page.NextCursor)
	if err != nil || decoded.ScopeDigest != scopeDigest || decoded.Search != "recipe" ||
		decoded.OwnerPrincipalID != ownerPrincipalID {
		t.Fatalf("cursor = %+v, error = %v", decoded, err)
	}

	query, err = service.listQuery(namespaceID, "profiles", PageRequest{
		PageSize: 20, Cursor: page.NextCursor, Search: "  RECIPE  ", Scope: scope,
		OwnerPrincipalID: ownerPrincipalID,
	})
	if err != nil || query.Search != "recipe" || query.After == nil {
		t.Fatalf("list query = %+v, error = %v", query, err)
	}
	if _, err := service.listQuery(namespaceID, "profiles", PageRequest{
		PageSize: 20, Cursor: page.NextCursor, Search: "builder", Scope: scope,
		OwnerPrincipalID: ownerPrincipalID,
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("search-mismatched cursor error = %v", err)
	}
	if _, err := service.listQuery(namespaceID, "profiles", PageRequest{
		PageSize: 20, Cursor: page.NextCursor, Search: "recipe", Scope: scope,
		OwnerPrincipalID: uuid.NewString(),
	}); !errors.Is(err, ErrInvalid) {
		t.Fatalf("owner-mismatched cursor error = %v", err)
	}
}
