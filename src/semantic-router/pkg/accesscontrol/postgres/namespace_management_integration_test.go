package postgres_test

import (
	"context"
	"database/sql"
	"errors"
	"net/netip"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	accesspostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol/postgres"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	authorizationpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/namespacemanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	namespaceTestID          = "11111111-1111-4111-8111-111111111111"
	namespaceTestPrincipalID = "22222222-2222-4222-8222-222222222222"
	namespaceTestBindingID   = "33333333-3333-4333-8333-333333333333"
	namespaceTestUserID      = "44444444-4444-4444-8444-444444444444"
	namespaceTestAnalystRole = "10000000-0000-5000-8000-000000000006"
	namespaceTestViewerRole  = "10000000-0000-5000-8000-000000000007"
)

func TestNamespaceManagementPostgresLifecycle(t *testing.T) {
	dsn := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if dsn == "" {
		dsn = os.Getenv("VLLM_SR_ACCESS_CONTROL_TEST_DATABASE_URL")
	}
	if dsn == "" {
		t.Skip("PostgreSQL Namespace Management test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	database := isolatedNamespaceDatabase(t, ctx, dsn)
	if err := (controlpostgres.Migrator{DB: database}).Apply(ctx); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}

	if _, err := database.ExecContext(ctx, `INSERT INTO management_principals
	  (id,issuer,subject,display_name,status) VALUES ($1,'test','namespace-actor','Namespace Actor','active')`, namespaceTestPrincipalID); err != nil {
		t.Fatalf("seed Management principal: %v", err)
	}
	store, testNamespaceManagementPostgresLifecycleErr := accesspostgres.New(database)
	if testNamespaceManagementPostgresLifecycleErr != nil {
		t.Fatal(testNamespaceManagementPostgresLifecycleErr)
	}
	repository, testNamespaceManagementPostgresLifecycleErr := accesspostgres.NewNamespaceManagementRepository(store)
	if testNamespaceManagementPostgresLifecycleErr != nil {
		t.Fatal(testNamespaceManagementPostgresLifecycleErr)
	}
	commands, testNamespaceManagementPostgresLifecycleErr := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys: map[string][]byte{
			"v1": []byte(strings.Repeat("c", 32)),
		},
	})
	if testNamespaceManagementPostgresLifecycleErr != nil {
		t.Fatal(testNamespaceManagementPostgresLifecycleErr)
	}
	t.Cleanup(func() { _ = commands.Close() })
	service, testNamespaceManagementPostgresLifecycleErr := namespacemanagement.NewService(namespacemanagement.Options{
		Repository:   repository,
		CommandCodec: commands,
		CursorKeyring: securitykeyring.Symmetric{
			ActiveVersion: "v1",
			Keys: map[string][]byte{
				"v1": []byte(strings.Repeat("p", 32)),
			},
		},
		IdempotencyTTL: time.Hour,
		NewID:          func() string { return namespaceTestID },
	})
	if testNamespaceManagementPostgresLifecycleErr != nil {
		t.Fatal(testNamespaceManagementPostgresLifecycleErr)
	}
	t.Cleanup(service.Close)
	actor := namespacemanagement.Actor{
		PrincipalID: namespaceTestPrincipalID,
		ActorChain:  []string{namespaceTestPrincipalID},
		RequestID:   "namespace-integration-create",
		SourceIP:    netip.MustParseAddr("192.0.2.55"),
		Reason:      "Create the default tenant",
	}
	create := namespacemanagement.CreateNamespaceRequest{
		Name:            "Default",
		BillingCurrency: "USD",
		IdempotencyKey:  "namespace-postgres-create-0001",
		Actor:           actor,
	}
	assertNamespaceAtomicCreate(t, ctx, database, service, create)

	assertNamespaceDurableDefaults(t, ctx, database, service)

	assertNamespacePolicyCAS(t, ctx, service, actor)

	assertNamespaceAuthorizationScope(t, ctx, database, service)

	assertNamespaceDeleteDependencies(t, ctx, database, service, actor)
}

func assertNamespaceAtomicCreate(
	t *testing.T,
	ctx context.Context,
	database *sql.DB,
	service *namespacemanagement.Service,
	request namespacemanagement.CreateNamespaceRequest,
) {
	t.Helper()
	if _, err := database.ExecContext(ctx, `CREATE FUNCTION reject_namespace_claim_seed()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'injected companion failure';
END $$;
CREATE TRIGGER reject_namespace_claim_seed
BEFORE INSERT ON routing_claim_schemas
FOR EACH ROW EXECUTE FUNCTION reject_namespace_claim_seed()`); err != nil {
		t.Fatalf("install companion failure injection: %v", err)
	}
	if _, err := service.CreateNamespace(ctx, request); err == nil {
		t.Fatal("Namespace create unexpectedly survived a companion insert failure")
	}
	var resources, commands, audits, outbox int
	if err := database.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_namespaces WHERE id=$1) +
  (SELECT count(*) FROM self_service_policies WHERE namespace_id=$1) +
  (SELECT count(*) FROM management_security_policies WHERE namespace_id=$1) +
  (SELECT count(*) FROM routing_claim_schemas WHERE namespace_id=$1),
  (SELECT count(*) FROM management_idempotency),
  (SELECT count(*) FROM access_audit_events WHERE namespace_id=$1),
  (SELECT count(*) FROM policy_outbox WHERE namespace_id=$1)`, namespaceTestID).Scan(
		&resources, &commands, &audits, &outbox,
	); err != nil {
		t.Fatal(err)
	}
	if resources != 0 || commands != 0 || audits != 0 || outbox != 0 {
		t.Fatalf("failed create leaked resources=%d commands=%d audits=%d outbox=%d",
			resources, commands, audits, outbox)
	}
	if _, err := database.ExecContext(ctx, `DROP TRIGGER reject_namespace_claim_seed ON routing_claim_schemas;
DROP FUNCTION reject_namespace_claim_seed()`); err != nil {
		t.Fatalf("remove companion failure injection: %v", err)
	}
	created, err := service.CreateNamespace(ctx, request)
	if err != nil || created.ID != namespaceTestID || created.Revision != 1 || created.Replayed {
		t.Fatalf("create Namespace = %#v, %v", created, err)
	}
	replayed, err := service.CreateNamespace(ctx, request)
	if err != nil || !replayed.Replayed || replayed.ID != namespaceTestID || replayed.Revision != 1 {
		t.Fatalf("replay Namespace = %#v, %v", replayed, err)
	}
	conflict := request
	conflict.Name = "Different"
	if _, err := service.CreateNamespace(ctx, conflict); !errors.Is(err, namespacemanagement.ErrIdempotencyConflict) {
		t.Fatalf("conflicting Namespace replay error = %v", err)
	}
}

func assertNamespaceDurableDefaults(
	t *testing.T,
	ctx context.Context,
	database *sql.DB,
	service *namespacemanagement.Service,
) {
	t.Helper()
	var namespaces, selfService, security, claims, commands, audits, outbox int
	if err := database.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_namespaces WHERE id=$1),
  (SELECT count(*) FROM self_service_policies WHERE namespace_id=$1),
  (SELECT count(*) FROM management_security_policies WHERE namespace_id=$1),
  (SELECT count(*) FROM routing_claim_schemas WHERE namespace_id=$1),
  (SELECT count(*) FROM management_idempotency),
  (SELECT count(*) FROM access_audit_events WHERE namespace_id=$1),
  (SELECT count(*) FROM policy_outbox WHERE namespace_id=$1)`, namespaceTestID).Scan(
		&namespaces, &selfService, &security, &claims, &commands, &audits, &outbox,
	); err != nil {
		t.Fatal(err)
	}
	if namespaces != 1 || selfService != 1 || security != 1 || claims != 1 ||
		commands != 1 || audits != 1 || outbox != 1 {
		t.Fatalf("atomic Namespace rows = namespace:%d self:%d security:%d claims:%d command:%d audit:%d outbox:%d",
			namespaces, selfService, security, claims, commands, audits, outbox)
	}
	policy, err := service.GetSelfServicePolicy(ctx, namespaceTestID)
	if err != nil || policy.MaxKeysPerUser != 0 || policy.MaxDelegatedSessions != 0 ||
		policy.AllowTeamKeyDelegation || policy.AutomaticFirstKey ||
		policy.DefaultAccessPolicyID != "" || policy.DefaultRateLimitPolicyID != "" {
		t.Fatalf("durable restrictive self-service policy = %#v, %v", policy, err)
	}
	if err := service.Ready(ctx); err != nil {
		t.Fatalf("complete active Namespace is not ready: %v", err)
	}
	if _, err := database.ExecContext(ctx, `DELETE FROM routing_claim_schemas WHERE namespace_id=$1`, namespaceTestID); err != nil {
		t.Fatal(err)
	}
	if err := service.Ready(ctx); !errors.Is(err, namespacemanagement.ErrUnavailable) {
		t.Fatalf("missing active companion readiness error = %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO routing_claim_schemas
  (namespace_id,definitions,revision) VALUES ($1,'{}'::jsonb,1)`, namespaceTestID); err != nil {
		t.Fatal(err)
	}
}

func assertNamespacePolicyCAS(
	t *testing.T,
	ctx context.Context,
	service *namespacemanagement.Service,
	actor namespacemanagement.Actor,
) {
	t.Helper()
	actor.RequestID = "namespace-integration-policy"
	actor.Reason = "Allow one self-service key"
	maximumKeys := 1
	request := namespacemanagement.PatchSelfServicePolicyRequest{
		NamespaceID: namespaceTestID, ExpectedRevision: 1, MaxKeysPerUser: &maximumKeys, Actor: actor,
	}
	patched, err := service.PatchSelfServicePolicy(ctx, request)
	if err != nil || patched.Revision != 2 {
		t.Fatalf("patch self-service policy = %#v, %v", patched, err)
	}
	if _, err := service.PatchSelfServicePolicy(ctx, request); !errors.Is(err, namespacemanagement.ErrRevisionConflict) {
		t.Fatalf("stale self-service CAS error = %v", err)
	}
}

func assertNamespaceAuthorizationScope(
	t *testing.T,
	ctx context.Context,
	database *sql.DB,
	service *namespacemanagement.Service,
) {
	t.Helper()
	if _, err := database.ExecContext(ctx, `INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,namespace_id,status)
VALUES ($1,$2,$3,'namespace',$4,'active')`, namespaceTestBindingID,
		namespaceTestPrincipalID, namespaceTestViewerRole, namespaceTestID,
	); err != nil {
		t.Fatalf("seed Namespace role binding: %v", err)
	}
	authority, authorityErr := authorizationpostgres.New(database)
	if authorityErr != nil {
		t.Fatal(authorityErr)
	}
	if _, err := authority.ResolveNamespaceResultScope(ctx, namespaceTestPrincipalID); !errors.Is(err, managementauthorization.ErrDenied) {
		t.Fatalf("routing-only viewer Namespace scope error = %v", err)
	}
	if _, err := database.ExecContext(ctx, `UPDATE management_role_bindings SET role_id=$1 WHERE id=$2`,
		namespaceTestAnalystRole, namespaceTestBindingID,
	); err != nil {
		t.Fatalf("replace viewer binding with analyst binding: %v", err)
	}
	scope, scopeErr := authority.ResolveNamespaceResultScope(ctx, namespaceTestPrincipalID)
	if scopeErr != nil || scope.All || len(scope.NamespaceIDs) != 1 || scope.NamespaceIDs[0] != namespaceTestID {
		t.Fatalf("compiled Namespace result scope = %#v, %v", scope, scopeErr)
	}
	page, listErr := service.ListNamespaces(ctx, namespacemanagement.ListRequest{Scope: scope, PageSize: 10})
	if listErr != nil || len(page.Items) != 1 || page.Items[0].ID != namespaceTestID {
		t.Fatalf("scope-pushed Namespace list = %#v, %v", page, listErr)
	}
}

func assertNamespaceDeleteDependencies(
	t *testing.T,
	ctx context.Context,
	database *sql.DB,
	service *namespacemanagement.Service,
	actor namespacemanagement.Actor,
) {
	t.Helper()
	tx, beginErr := database.BeginTx(ctx, nil)
	if beginErr != nil {
		t.Fatal(beginErr)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_subjects
  (namespace_id,id,kind) VALUES ($1,$2,'user')`, namespaceTestID, namespaceTestUserID); err != nil {
		_ = tx.Rollback()
		t.Fatal(err)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_users
  (id,namespace_id,email,display_name,status) VALUES ($1,$2,'active@example.com','Active User','active')`,
		namespaceTestUserID, namespaceTestID,
	); err != nil {
		_ = tx.Rollback()
		t.Fatal(err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	actor.RequestID = "namespace-integration-delete"
	actor.Reason = "Delete Namespace"
	request := namespacemanagement.DeleteNamespaceRequest{
		NamespaceID: namespaceTestID, ExpectedRevision: 1, Actor: actor,
	}
	if _, err := service.DeleteNamespace(ctx, request); !errors.Is(err, namespacemanagement.ErrDependency) {
		t.Fatalf("Namespace delete with active dependency error = %v", err)
	}
	for _, statement := range []struct {
		query string
		id    string
	}{
		{`UPDATE access_users SET status='disabled' WHERE id=$1`, namespaceTestUserID},
		{`UPDATE management_role_bindings SET status='disabled' WHERE id=$1`, namespaceTestBindingID},
	} {
		if _, err := database.ExecContext(ctx, statement.query, statement.id); err != nil {
			t.Fatal(err)
		}
	}
	deleted, deleteErr := service.DeleteNamespace(ctx, request)
	if deleteErr != nil || deleted.HTTPStatus != 204 || deleted.Revision != 2 {
		t.Fatalf("soft-delete Namespace = %#v, %v", deleted, deleteErr)
	}
	value, getErr := service.GetNamespace(ctx, namespaceTestID)
	if getErr != nil || value.Status != accesscontrol.NamespaceStatusDisabled || value.Revision != 2 {
		t.Fatalf("disabled Namespace = %#v, %v", value, getErr)
	}
}

func isolatedNamespaceDatabase(t *testing.T, ctx context.Context, dsn string) *sql.DB {
	t.Helper()
	admin, isolatedNamespaceDatabaseErr := sql.Open("postgres", dsn)
	if isolatedNamespaceDatabaseErr != nil {
		t.Fatal(isolatedNamespaceDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_namespace_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})

	parsed, isolatedNamespaceDatabaseErr := url.Parse(dsn)
	if isolatedNamespaceDatabaseErr != nil {
		t.Fatal(isolatedNamespaceDatabaseErr)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	database, isolatedNamespaceDatabaseErr := sql.Open("postgres", parsed.String())
	if isolatedNamespaceDatabaseErr != nil {
		t.Fatal(isolatedNamespaceDatabaseErr)
	}
	t.Cleanup(func() { _ = database.Close() })
	return database
}
