package postgres

import (
	"context"
	"database/sql"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	controlplanepostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
)

func TestStoreIntegrationLoadsNamespaceSnapshotWithoutAndWithUserLink(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	database := authorizationTestDatabase(t, ctx)

	principalID := uuid.NewString()
	namespaceID := uuid.NewString()
	userID := uuid.NewString()
	if _, err := database.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,$2,$3,'USD','active')`, namespaceID, "Authorization "+namespaceID, "quota:"+namespaceID); err != nil {
		t.Fatalf("seed namespace: %v", err)
	}
	if _, err := database.ExecContext(ctx,
		`INSERT INTO self_service_policies (namespace_id,seed_version) VALUES ($1,1)`, namespaceID); err != nil {
		t.Fatalf("seed self-service policy: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,status)
VALUES ($1,$2,$3,'Authorization integration principal','active')`,
		principalID, "https://issuer.example/"+principalID, "subject-"+principalID); err != nil {
		t.Fatalf("seed principal: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,namespace_id,status)
VALUES ($1,$2,'10000000-0000-5000-8000-000000000002','namespace',$3,'active')`,
		uuid.NewString(), principalID, namespaceID); err != nil {
		t.Fatalf("seed namespace authority: %v", err)
	}
	store, newStoreErr := New(database)
	if newStoreErr != nil {
		t.Fatal(newStoreErr)
	}
	withoutLink, loadWithoutLinkErr := store.Load(ctx,
		accesscontrol.ManagementPrincipalID(principalID), accesscontrol.NamespaceID(namespaceID))
	if loadWithoutLinkErr != nil {
		t.Fatalf("Load() without User link error = %v", loadWithoutLinkErr)
	}
	if len(withoutLink.RoleGrants) != 1 || len(withoutLink.TeamGrants) != 0 || withoutLink.AuthorityDigest == "" {
		t.Fatalf("Load() without User link = %#v", withoutLink)
	}

	if _, err := database.ExecContext(ctx,
		`INSERT INTO access_subjects (namespace_id,id,kind) VALUES ($1,$2,'user')`, namespaceID, userID); err != nil {
		t.Fatalf("seed User subject: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO access_users
  (namespace_id,id,email,display_name,status)
VALUES ($1,$2,$3,'Linked integration user','active')`, namespaceID, userID, userID+"@example.test"); err != nil {
		t.Fatalf("seed User: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_principal_user_links
  (principal_id,namespace_id,user_id) VALUES ($1,$2,$3)`, principalID, namespaceID, userID); err != nil {
		t.Fatalf("seed principal User link: %v", err)
	}
	withLink, loadWithLinkErr := store.Load(ctx,
		accesscontrol.ManagementPrincipalID(principalID), accesscontrol.NamespaceID(namespaceID))
	if loadWithLinkErr != nil {
		t.Fatalf("Load() with User link error = %v", loadWithLinkErr)
	}
	if len(withLink.RoleGrants) != 1 || len(withLink.TeamGrants) != 0 || withLink.AuthorityDigest == "" {
		t.Fatalf("Load() with User link = %#v", withLink)
	}
	if withLink.AuthorityDigest == withoutLink.AuthorityDigest {
		t.Fatal("User-link authority facts did not change the snapshot digest")
	}
}

func authorizationTestDatabase(t *testing.T, ctx context.Context) *sql.DB {
	t.Helper()
	databaseURL := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("PostgreSQL test database is not configured")
	}
	admin, openDatabaseErr := sql.Open("postgres", databaseURL)
	if openDatabaseErr != nil {
		t.Fatal(openDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	schema := "vsr_authorization_test_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	parsed, parseDatabaseURLErr := url.Parse(databaseURL)
	if parseDatabaseURLErr != nil {
		t.Fatal(parseDatabaseURLErr)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	database, err := sql.Open("postgres", parsed.String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	if err := (controlplanepostgres.Migrator{DB: database}).Apply(ctx); err != nil {
		t.Fatalf("apply control-plane migrations: %v", err)
	}
	return database
}
