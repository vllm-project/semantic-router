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

	const (
		namespaceID = "11111111-1111-4111-8111-111111111111"
		principalID = "22222222-2222-4222-8222-222222222222"
		bindingID   = "33333333-3333-4333-8333-333333333333"
		userID      = "44444444-4444-4444-8444-444444444444"
		analystRole = "10000000-0000-5000-8000-000000000006"
		viewerRole  = "10000000-0000-5000-8000-000000000007"
	)
	if _, err := database.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,status) VALUES ($1,'test','namespace-actor','Namespace Actor','active')`, principalID); err != nil {
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
		NewID:          func() string { return namespaceID },
	})
	if testNamespaceManagementPostgresLifecycleErr != nil {
		t.Fatal(testNamespaceManagementPostgresLifecycleErr)
	}
	t.Cleanup(service.Close)
	actor := namespacemanagement.Actor{
		PrincipalID: principalID,
		ActorChain:  []string{principalID},
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
	if _, err := service.CreateNamespace(ctx, create); err == nil {
		t.Fatal("Namespace create unexpectedly survived a companion insert failure")
	}
	var partialResources, partialCommands, partialAudits, partialOutbox int
	if err := database.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_namespaces WHERE id=$1) +
  (SELECT count(*) FROM self_service_policies WHERE namespace_id=$1) +
  (SELECT count(*) FROM management_security_policies WHERE namespace_id=$1) +
  (SELECT count(*) FROM routing_claim_schemas WHERE namespace_id=$1),
  (SELECT count(*) FROM management_idempotency),
  (SELECT count(*) FROM access_audit_events WHERE namespace_id=$1),
  (SELECT count(*) FROM policy_outbox WHERE namespace_id=$1)`, namespaceID).Scan(
		&partialResources,
		&partialCommands,
		&partialAudits,
		&partialOutbox,
	); err != nil {
		t.Fatal(err)
	}
	if partialResources != 0 || partialCommands != 0 || partialAudits != 0 || partialOutbox != 0 {
		t.Fatalf("failed create leaked resources=%d commands=%d audits=%d outbox=%d",
			partialResources, partialCommands, partialAudits, partialOutbox)
	}
	if _, err := database.ExecContext(ctx, `DROP TRIGGER reject_namespace_claim_seed ON routing_claim_schemas;
DROP FUNCTION reject_namespace_claim_seed()`); err != nil {
		t.Fatalf("remove companion failure injection: %v", err)
	}
	created, testNamespaceManagementPostgresLifecycleErr := service.CreateNamespace(ctx, create)
	if testNamespaceManagementPostgresLifecycleErr != nil || created.ID != namespaceID || created.Revision != 1 || created.Replayed {
		t.Fatalf("create Namespace = %#v, %v", created, testNamespaceManagementPostgresLifecycleErr)
	}
	replayed, testNamespaceManagementPostgresLifecycleErr := service.CreateNamespace(ctx, create)
	if testNamespaceManagementPostgresLifecycleErr != nil || !replayed.Replayed || replayed.ID != namespaceID || replayed.Revision != 1 {
		t.Fatalf("replay Namespace = %#v, %v", replayed, testNamespaceManagementPostgresLifecycleErr)
	}
	conflict := create
	conflict.Name = "Different"
	if _, err := service.CreateNamespace(ctx, conflict); !errors.Is(err, namespacemanagement.ErrIdempotencyConflict) {
		t.Fatalf("conflicting Namespace replay error = %v", err)
	}

	var namespaces, selfService, security, claims, commandsCount, audits, outbox int
	if err := database.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM access_namespaces WHERE id=$1),
  (SELECT count(*) FROM self_service_policies WHERE namespace_id=$1),
  (SELECT count(*) FROM management_security_policies WHERE namespace_id=$1),
  (SELECT count(*) FROM routing_claim_schemas WHERE namespace_id=$1),
  (SELECT count(*) FROM management_idempotency),
  (SELECT count(*) FROM access_audit_events WHERE namespace_id=$1),
  (SELECT count(*) FROM policy_outbox WHERE namespace_id=$1)`, namespaceID).Scan(
		&namespaces,
		&selfService,
		&security,
		&claims,
		&commandsCount,
		&audits,
		&outbox,
	); err != nil {
		t.Fatal(err)
	}
	if namespaces != 1 || selfService != 1 || security != 1 || claims != 1 ||
		commandsCount != 1 || audits != 1 || outbox != 1 {
		t.Fatalf("atomic Namespace rows = namespace:%d self:%d security:%d claims:%d command:%d audit:%d outbox:%d",
			namespaces, selfService, security, claims, commandsCount, audits, outbox)
	}
	policy, testNamespaceManagementPostgresLifecycleErr := service.GetSelfServicePolicy(ctx, namespaceID)
	if testNamespaceManagementPostgresLifecycleErr != nil || policy.MaxKeysPerUser != 0 || policy.MaxDelegatedSessions != 0 ||
		policy.AllowTeamKeyDelegation || policy.AutomaticFirstKey ||
		policy.DefaultAccessPolicyID != "" || policy.DefaultRateLimitPolicyID != "" {
		t.Fatalf("durable restrictive self-service policy = %#v, %v", policy, testNamespaceManagementPostgresLifecycleErr)
	}
	if err := service.Ready(ctx); err != nil {
		t.Fatalf("complete active Namespace is not ready: %v", err)
	}
	if _, err := database.ExecContext(ctx, `DELETE FROM routing_claim_schemas WHERE namespace_id=$1`, namespaceID); err != nil {
		t.Fatal(err)
	}
	if err := service.Ready(ctx); !errors.Is(err, namespacemanagement.ErrUnavailable) {
		t.Fatalf("missing active companion readiness error = %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO routing_claim_schemas
  (namespace_id,definitions,revision) VALUES ($1,'{}'::jsonb,1)`, namespaceID); err != nil {
		t.Fatal(err)
	}

	actor.RequestID = "namespace-integration-policy"
	actor.Reason = "Allow one self-service key"
	maximumKeys := 1
	patched, testNamespaceManagementPostgresLifecycleErr := service.PatchSelfServicePolicy(ctx, namespacemanagement.PatchSelfServicePolicyRequest{
		NamespaceID:      namespaceID,
		ExpectedRevision: 1,
		MaxKeysPerUser:   &maximumKeys,
		Actor:            actor,
	})
	if testNamespaceManagementPostgresLifecycleErr != nil || patched.Revision != 2 {
		t.Fatalf("patch self-service policy = %#v, %v", patched, testNamespaceManagementPostgresLifecycleErr)
	}
	if _, err := service.PatchSelfServicePolicy(ctx, namespacemanagement.PatchSelfServicePolicyRequest{
		NamespaceID:      namespaceID,
		ExpectedRevision: 1,
		MaxKeysPerUser:   &maximumKeys,
		Actor:            actor,
	}); !errors.Is(err, namespacemanagement.ErrRevisionConflict) {
		t.Fatalf("stale self-service CAS error = %v", err)
	}

	if _, err := database.ExecContext(ctx, `INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,namespace_id,status)
VALUES ($1,$2,$3,'namespace',$4,'active')`, bindingID, principalID, viewerRole, namespaceID); err != nil {
		t.Fatalf("seed Namespace role binding: %v", err)
	}
	authority, testNamespaceManagementPostgresLifecycleErr := authorizationpostgres.New(database)
	if testNamespaceManagementPostgresLifecycleErr != nil {
		t.Fatal(testNamespaceManagementPostgresLifecycleErr)
	}
	if _, err := authority.ResolveNamespaceResultScope(ctx, principalID); !errors.Is(err, managementauthorization.ErrDenied) {
		t.Fatalf("routing-only viewer Namespace scope error = %v", err)
	}
	if _, err := database.ExecContext(ctx, `UPDATE management_role_bindings SET role_id=$1 WHERE id=$2`, analystRole, bindingID); err != nil {
		t.Fatalf("replace viewer binding with analyst binding: %v", err)
	}
	scope, testNamespaceManagementPostgresLifecycleErr := authority.ResolveNamespaceResultScope(ctx, principalID)
	if testNamespaceManagementPostgresLifecycleErr != nil || scope.All || len(scope.NamespaceIDs) != 1 || scope.NamespaceIDs[0] != namespaceID {
		t.Fatalf("compiled Namespace result scope = %#v, %v", scope, testNamespaceManagementPostgresLifecycleErr)
	}
	page, testNamespaceManagementPostgresLifecycleErr := service.ListNamespaces(ctx, namespacemanagement.ListRequest{Scope: scope, PageSize: 10})
	if testNamespaceManagementPostgresLifecycleErr != nil || len(page.Items) != 1 || page.Items[0].ID != namespaceID {
		t.Fatalf("scope-pushed Namespace list = %#v, %v", page, testNamespaceManagementPostgresLifecycleErr)
	}

	seedTx, testNamespaceManagementPostgresLifecycleErr := database.BeginTx(ctx, nil)
	if testNamespaceManagementPostgresLifecycleErr != nil {
		t.Fatal(testNamespaceManagementPostgresLifecycleErr)
	}
	if _, err := seedTx.ExecContext(ctx, `INSERT INTO access_subjects
  (namespace_id,id,kind) VALUES ($1,$2,'user')`, namespaceID, userID); err != nil {
		_ = seedTx.Rollback()
		t.Fatal(err)
	}
	if _, err := seedTx.ExecContext(ctx, `INSERT INTO access_users
  (id,namespace_id,email,display_name,status) VALUES ($1,$2,'active@example.com','Active User','active')`, userID, namespaceID); err != nil {
		_ = seedTx.Rollback()
		t.Fatal(err)
	}
	if err := seedTx.Commit(); err != nil {
		t.Fatal(err)
	}
	actor.RequestID = "namespace-integration-delete"
	actor.Reason = "Delete Namespace"
	if _, err := service.DeleteNamespace(ctx, namespacemanagement.DeleteNamespaceRequest{
		NamespaceID: namespaceID, ExpectedRevision: 1, Actor: actor,
	}); !errors.Is(err, namespacemanagement.ErrDependency) {
		t.Fatalf("Namespace delete with active dependency error = %v", err)
	}
	if _, err := database.ExecContext(ctx, `UPDATE access_users SET status='disabled' WHERE id=$1`, userID); err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `UPDATE management_role_bindings SET status='disabled' WHERE id=$1`, bindingID); err != nil {
		t.Fatal(err)
	}
	deleted, testNamespaceManagementPostgresLifecycleErr := service.DeleteNamespace(ctx, namespacemanagement.DeleteNamespaceRequest{
		NamespaceID: namespaceID, ExpectedRevision: 1, Actor: actor,
	})
	if testNamespaceManagementPostgresLifecycleErr != nil || deleted.HTTPStatus != 204 || deleted.Revision != 2 {
		t.Fatalf("soft-delete Namespace = %#v, %v", deleted, testNamespaceManagementPostgresLifecycleErr)
	}
	value, testNamespaceManagementPostgresLifecycleErr := service.GetNamespace(ctx, namespaceID)
	if testNamespaceManagementPostgresLifecycleErr != nil || value.Status != accesscontrol.NamespaceStatusDisabled || value.Revision != 2 {
		t.Fatalf("disabled Namespace = %#v, %v", value, testNamespaceManagementPostgresLifecycleErr)
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
