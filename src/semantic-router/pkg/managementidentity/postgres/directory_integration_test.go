package postgres

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"net/netip"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	directoryNamespaceOne   = "11111111-1111-4111-8111-111111111111"
	directoryNamespaceTwo   = "22222222-2222-4222-8222-222222222222"
	directoryPrincipalOne   = "30000000-0000-4000-8000-000000000001"
	directoryPrincipalTwo   = "30000000-0000-4000-8000-000000000002"
	directoryPrincipalThree = "30000000-0000-4000-8000-000000000003"
	directoryUserOne        = "40000000-0000-4000-8000-000000000001"
	directoryUserTwo        = "40000000-0000-4000-8000-000000000002"
	directoryUserThree      = "40000000-0000-4000-8000-000000000003"
)

func TestPrincipalDirectoryIsNamespaceSafeFilteredAndKeysetPaginated(t *testing.T) {
	database := bootstrapTestDatabase(t)
	seedPrincipalDirectory(t, database)
	store := newPrincipalDirectoryStore(t, database)
	ctx := context.Background()

	first, err := store.ListPrincipalDirectory(ctx, managementidentity.PrincipalDirectoryRequest{
		NamespaceID: directoryNamespaceOne, Limit: 1,
	})
	if err != nil || len(first.Items) != 1 || first.NextCursor != directoryPrincipalOne ||
		!first.Items[0].Linked() || string(first.Items[0].UserID) != directoryUserOne {
		t.Fatalf("first directory page = %#v, %v", first, err)
	}
	second, err := store.ListPrincipalDirectory(ctx, managementidentity.PrincipalDirectoryRequest{
		NamespaceID: directoryNamespaceOne, AfterID: first.NextCursor, Limit: 2,
	})
	if err != nil || len(second.Items) != 2 || second.NextCursor != "" ||
		second.Items[0].PrincipalID != directoryPrincipalTwo || second.Items[0].Linked() {
		t.Fatalf("second directory page = %#v, %v", second, err)
	}

	entry, err := store.GetPrincipalDirectoryEntry(ctx, directoryNamespaceTwo, directoryPrincipalTwo)
	if err != nil || !entry.Linked() || string(entry.UserID) != directoryUserThree {
		t.Fatalf("namespace-two entry = %#v, %v", entry, err)
	}
	entry, err = store.GetPrincipalDirectoryEntry(ctx, directoryNamespaceOne, directoryPrincipalTwo)
	if err != nil || entry.Linked() || entry.UserID != "" {
		t.Fatalf("namespace-one projection leaked another namespace link: %#v, %v", entry, err)
	}

	search, err := store.ListPrincipalDirectory(ctx, managementidentity.PrincipalDirectoryRequest{
		NamespaceID: directoryNamespaceOne, Search: "BETA@", Limit: 10,
	})
	if err != nil || len(search.Items) != 1 || search.Items[0].PrincipalID != directoryPrincipalTwo {
		t.Fatalf("bounded prefix search = %#v, %v", search, err)
	}
	links, err := store.ListPrincipalUserLinks(ctx, managementidentity.PrincipalUserLinkListRequest{
		NamespaceID: directoryNamespaceTwo, UserID: directoryUserThree, Limit: 1,
	})
	if err != nil || len(links.Items) != 1 || links.Items[0].PrincipalID != directoryPrincipalTwo ||
		links.Items[0].NamespaceID != directoryNamespaceTwo {
		t.Fatalf("indexed namespace link filter = %#v, %v", links, err)
	}
}

func TestPrincipalUserLinkIdempotencyCASAndAuthorityCarryPrevention(t *testing.T) {
	database := bootstrapTestDatabase(t)
	seedPrincipalDirectory(t, database)
	store := newPrincipalDirectoryStore(t, database)
	ctx := context.Background()
	// Command expiry is validated against PostgreSQL's clock, so the fixture must
	// remain future-dated whenever the suite runs instead of expiring on a fixed
	// calendar date.
	now := time.Now().UTC()
	actor := managementidentity.MutationActor{
		PrincipalID: directoryPrincipalOne, RequestID: "principal-link-create-1",
		SourceIP: netip.MustParseAddr("192.0.2.10"), Reason: "Link principal to User.",
	}
	command := principalLinkCommand(t, store.commands, now, directoryPrincipalOne,
		"principal-link-create-0001", []byte(`{"userId":"`+directoryUserTwo+`"}`))
	request := managementidentity.LinkMutation{
		PrincipalID: directoryPrincipalThree, NamespaceID: directoryNamespaceOne,
		UserID: directoryUserTwo, Command: command, Actor: actor,
	}
	created, err := store.PutPrincipalUserLink(ctx, request)
	if err != nil || created.ResponseStatus != 201 || created.Replayed || created.Revision != 1 {
		t.Fatalf("create principal link = %#v, %v", created, err)
	}
	replayed, err := store.PutPrincipalUserLink(ctx, request)
	if err != nil || !replayed.Replayed || replayed.ID != created.ID || replayed.Revision != created.Revision {
		t.Fatalf("replay principal link = %#v, %v", replayed, err)
	}

	if _, err := store.PutPrincipalUserLink(ctx, managementidentity.LinkMutation{
		PrincipalID: directoryPrincipalThree, NamespaceID: directoryNamespaceOne,
		UserID: directoryUserOne,
		Command: principalLinkCommand(t, store.commands, now, directoryPrincipalOne,
			"principal-link-relink-no-cas", []byte(`{"userId":"`+directoryUserOne+`"}`)),
		Actor: actor,
	}); !errors.Is(err, managementidentity.ErrRevisionConflict) {
		t.Fatalf("relink without CAS error = %v", err)
	}

	if _, err := database.Exec(`INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,namespace_id,resource_id,delegation_ceiling,status,revision)
VALUES ('50000000-0000-4000-8000-000000000001',$1,'10000000-0000-5000-8000-000000000008',
        'user',$2,$3,'[]'::jsonb,'active',1)`, directoryPrincipalThree, directoryNamespaceOne, directoryUserTwo); err != nil {
		t.Fatal(err)
	}
	revision := uint64(1)
	if _, err := store.PutPrincipalUserLink(ctx, managementidentity.LinkMutation{
		PrincipalID: directoryPrincipalThree, NamespaceID: directoryNamespaceOne,
		UserID: directoryUserOne, ExpectedRevision: &revision,
		Command: principalLinkCommand(t, store.commands, now, directoryPrincipalOne,
			"principal-link-relink-cas-1", []byte(`{"userId":"`+directoryUserOne+`","revision":1}`)),
		Actor: actor,
	}); !errors.Is(err, managementidentity.ErrPrincipalLinkInUse) {
		t.Fatalf("authority-carrying relink error = %v", err)
	}
	if _, err := store.DeletePrincipalUserLink(ctx, managementidentity.LinkMutation{
		PrincipalID: directoryPrincipalThree, NamespaceID: directoryNamespaceOne,
		UserID: directoryUserTwo, ExpectedRevision: &revision, Actor: actor,
	}); !errors.Is(err, managementidentity.ErrPrincipalLinkInUse) {
		t.Fatalf("authority-carrying unlink error = %v", err)
	}
}

func newPrincipalDirectoryStore(t *testing.T, database *sql.DB) *Store {
	t.Helper()
	store, err := New(database, principalDirectoryCodec(t))
	if err != nil {
		t.Fatal(err)
	}
	return store
}

func seedPrincipalDirectory(t *testing.T, database *sql.DB) {
	t.Helper()
	statements := []struct {
		query string
		args  []any
	}{
		{`INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status,revision,runtime_epoch)
VALUES ($1,'directory-one','directory-one','USD','active',1,1),
       ($2,'directory-two','directory-two','USD','active',1,1)`, []any{directoryNamespaceOne, directoryNamespaceTwo}},
		{`INSERT INTO access_subjects (namespace_id,id,kind) VALUES
  ($1,$2,'user'),($1,$3,'user'),($4,$5,'user')`, []any{
			directoryNamespaceOne, directoryUserOne, directoryUserTwo, directoryNamespaceTwo, directoryUserThree,
		}},
		{`INSERT INTO access_users (id,namespace_id,email,display_name,status,revision) VALUES
  ($1,$2,'one@example.com','One','active',1),
  ($3,$2,'two@example.com','Two','active',1),
  ($4,$5,'three@example.com','Three','active',1)`, []any{
			directoryUserOne, directoryNamespaceOne, directoryUserTwo, directoryUserThree, directoryNamespaceTwo,
		}},
		{`INSERT INTO management_principals
  (id,issuer,subject,display_name,verified_email,attributes,status,revision) VALUES
  ($1,'https://issuer.example','alpha','Alpha','alpha@example.com','{}'::jsonb,'active',1),
  ($2,'https://issuer.example','beta','Beta','beta@example.com','{}'::jsonb,'active',1),
  ($3,'https://issuer.example','gamma','Gamma','gamma@example.com','{}'::jsonb,'active',1)`, []any{
			directoryPrincipalOne, directoryPrincipalTwo, directoryPrincipalThree,
		}},
		{`INSERT INTO management_principal_user_links
  (principal_id,namespace_id,user_id,revision) VALUES ($1,$2,$3,1),($4,$5,$6,1)`, []any{
			directoryPrincipalOne, directoryNamespaceOne, directoryUserOne,
			directoryPrincipalTwo, directoryNamespaceTwo, directoryUserThree,
		}},
	}
	for _, statement := range statements {
		if _, err := database.Exec(statement.query, statement.args...); err != nil {
			t.Fatal(err)
		}
	}
}

func principalLinkCommand(t *testing.T, codec *managementcommand.Codec, now time.Time, principalID, key string, body []byte) managementcommand.Command {
	t.Helper()
	command, err := codec.Bind(managementcommand.NamespaceCommandScope(directoryNamespaceOne), principalID,
		"/management/v1/namespaces/"+directoryNamespaceOne+"/principal-user-links/"+directoryPrincipalThree,
		key, body, now, now.Add(time.Hour))
	if err != nil {
		t.Fatal(err)
	}
	return command
}

func principalDirectoryCodec(t *testing.T) *managementcommand.Codec {
	t.Helper()
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": bytes.Repeat([]byte{0x42}, 32)},
	})
	if err != nil {
		t.Fatal(err)
	}
	return codec
}
