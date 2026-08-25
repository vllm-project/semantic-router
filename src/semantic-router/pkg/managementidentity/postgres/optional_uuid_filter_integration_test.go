package postgres

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

func TestOptionalUUIDFiltersSupportFirstPageAndKeysetPagination(t *testing.T) {
	database := bootstrapTestDatabase(t)
	seedPrincipalDirectory(t, database)
	store := newPrincipalDirectoryStore(t, database)
	ctx := context.Background()

	principals, principalsErr := store.ListPrincipals(ctx, managementidentity.ListRequest{Limit: 1})
	if principalsErr != nil || len(principals.Items) != 1 || principals.NextCursor == "" {
		t.Fatalf("first principal page = %#v, %v", principals, principalsErr)
	}
	nextPrincipals, nextPrincipalsErr := store.ListPrincipals(ctx, managementidentity.ListRequest{
		AfterID: principals.NextCursor,
		Limit:   2,
	})
	if nextPrincipalsErr != nil || len(nextPrincipals.Items) != 2 || nextPrincipals.NextCursor != "" {
		t.Fatalf("next principal page = %#v, %v", nextPrincipals, nextPrincipalsErr)
	}

	roles, rolesErr := store.ListRoles(ctx, "", managementidentity.ListRequest{Limit: 1})
	if rolesErr != nil || len(roles.Items) != 1 || roles.NextCursor == "" {
		t.Fatalf("cluster role first page = %#v, %v", roles, rolesErr)
	}
	nextRoles, nextRolesErr := store.ListRoles(ctx, "", managementidentity.ListRequest{
		AfterID: roles.NextCursor,
		Limit:   200,
	})
	if nextRolesErr != nil || len(nextRoles.Items) == 0 {
		t.Fatalf("cluster role next page = %#v, %v", nextRoles, nextRolesErr)
	}

	bindings, bindingsErr := store.ListRoleBindings(ctx, "", managementidentity.ListRequest{Limit: 10})
	if bindingsErr != nil || len(bindings.Items) != 0 {
		t.Fatalf("unfiltered role bindings = %#v, %v", bindings, bindingsErr)
	}
	issuers, issuersErr := store.ListTrustedIdentityIssuers(ctx, managementidentity.ListRequest{Limit: 10})
	if issuersErr != nil || len(issuers.Items) != 0 {
		t.Fatalf("trusted issuer first page = %#v, %v", issuers, issuersErr)
	}

	if _, err := database.ExecContext(ctx, `INSERT INTO management_sessions
  (id,principal_id,token_id,audience,auth_source_kind,evidence_kind,assurance,
   authenticated_at,expires_at,status)
VALUES ('60000000-0000-4000-8000-000000000001',$1,'optional-filter-session-1',
        'vllm-sr-management','issuer','human','{}'::jsonb,
        clock_timestamp(),clock_timestamp()+interval '5 minutes','active')`, directoryPrincipalOne); err != nil {
		t.Fatal(err)
	}
	sessions, err := store.ListManagementSessions(ctx, directoryPrincipalOne, managementidentity.ListRequest{Limit: 10})
	if err != nil || len(sessions.Items) != 1 {
		t.Fatalf("Management session first page = %#v, %v", sessions, err)
	}
	directory, err := store.ListPrincipalDirectory(ctx, managementidentity.PrincipalDirectoryRequest{
		NamespaceID: directoryNamespaceOne,
		Limit:       10,
	})
	if err != nil || len(directory.Items) != 3 {
		t.Fatalf("principal directory first page = %#v, %v", directory, err)
	}
	links, err := store.ListPrincipalUserLinks(ctx, managementidentity.PrincipalUserLinkListRequest{
		NamespaceID: directoryNamespaceOne,
		Limit:       10,
	})
	if err != nil || len(links.Items) != 1 {
		t.Fatalf("principal User-link first page = %#v, %v", links, err)
	}
	principalLinks, err := store.ListPrincipalLinks(ctx, directoryPrincipalOne, managementidentity.ListRequest{Limit: 10})
	if err != nil || len(principalLinks.Items) != 1 {
		t.Fatalf("principal namespace-link first page = %#v, %v", principalLinks, err)
	}

	if _, err := store.ListPrincipals(ctx, managementidentity.ListRequest{AfterID: "not-a-uuid", Limit: 10}); err == nil {
		t.Fatalf("invalid principal cursor error = %v", err)
	}
	if _, err := store.ListPrincipalDirectory(ctx, managementidentity.PrincipalDirectoryRequest{
		NamespaceID: directoryNamespaceOne,
		AfterID:     "not-a-uuid",
		Limit:       10,
	}); !errors.Is(err, managementidentity.ErrInvalidLifecycleRequest) {
		t.Fatalf("invalid directory cursor error = %v", err)
	}
}
