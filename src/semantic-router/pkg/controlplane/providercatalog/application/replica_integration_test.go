package application

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	catalogpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providerdiscovery"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type integrationSessionAuthenticator struct{}

func (integrationSessionAuthenticator) Authenticate(
	_ context.Context,
	_ string,
	namespaceID string,
	_ time.Time,
) (managementauth.AuthenticatedSession, error) {
	return managementauth.AuthenticatedSession{
		NamespaceID: namespaceID,
		Session: managementauth.LiveSession{Session: managementauth.Session{
			PrincipalID: "22222222-2222-4222-8222-222222222222",
		}},
	}, nil
}

type initialStateBarrierCoordinator struct {
	Coordinator
	arrived chan<- struct{}
	release <-chan struct{}
	once    sync.Once
}

func (coordinator *initialStateBarrierCoordinator) State(
	ctx context.Context,
) (catalogpostgres.State, error) {
	state, err := coordinator.Coordinator.State(ctx)
	coordinator.once.Do(func() {
		coordinator.arrived <- struct{}{}
		select {
		case <-coordinator.release:
		case <-ctx.Done():
		}
	})
	return state, err
}

func TestReplicaColdStartCoordinatesTwoReplicasAndRestart(t *testing.T) {
	db := managedCatalogIntegrationDatabase(t)
	registry := managedTestRegistry(t, []string{string(llmprotocol.OpenAIChatV1)})
	groups := testRolloutGroups()
	arrived := make(chan struct{}, 2)
	release := make(chan struct{})
	first := newBarrierReplica(t, db, registry, "router-cold-a", groups[:1], groups, arrived, release)
	second := newBarrierReplica(t, db, registry, "router-cold-b", groups[1:], groups, arrived, release)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	errorsByReplica := make(chan error, 2)
	for _, replica := range []*Replica{first, second} {
		go func(candidate *Replica) {
			if err := candidate.EnsureColdStart(ctx); err != nil {
				errorsByReplica <- err
				return
			}
			errorsByReplica <- nil
		}(replica)
	}
	for range 2 {
		select {
		case <-arrived:
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		}
	}
	close(release)
	for range 2 {
		if err := <-errorsByReplica; err != nil {
			t.Fatal(err)
		}
	}
	if err := first.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	if err := second.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}

	state, err := first.coordinator.State(ctx)
	if err != nil {
		t.Fatal(err)
	}
	wantRevision := registry.Snapshot().Revision()
	if state.DesiredRevision != wantRevision || state.ActiveRevision != wantRevision || state.Generation != 2 {
		t.Fatalf("cold-start state = %+v, want active revision %s at generation 2", state, wantRevision)
	}
	if err := first.Ready(ctx); err != nil {
		t.Fatalf("first replica readiness: %v", err)
	}
	if err := second.Ready(ctx); err != nil {
		t.Fatalf("second replica readiness: %v", err)
	}
	var acknowledgements int
	if err := db.QueryRowContext(ctx, `SELECT count(*)
FROM provider_catalog_replica_acks
WHERE revision = $1 AND replica_id IN ('router-cold-a', 'router-cold-b')`, wantRevision).Scan(&acknowledgements); err != nil {
		t.Fatal(err)
	}
	if acknowledgements != len(groups) {
		t.Fatalf("replica acknowledgements = %d, want %d", acknowledgements, len(groups))
	}

	assertReplicaRestart(t, ctx, db, registry, groups, state)
}

func newBarrierReplica(
	t *testing.T,
	db *sql.DB,
	registry *providercatalog.Registry,
	replicaID string,
	memberships []providercatalog.RolloutGroup,
	required []providercatalog.RolloutGroup,
	arrived chan<- struct{},
	release <-chan struct{},
) *Replica {
	t.Helper()
	coordinator, err := catalogpostgres.New(db, registry)
	if err != nil {
		t.Fatal(err)
	}
	barrier := &initialStateBarrierCoordinator{
		Coordinator: coordinator, arrived: arrived, release: release,
	}
	replica, err := NewReplica(barrier, registry, ReplicaOptions{
		ReplicaID: replicaID, RolloutGroups: memberships, RequiredRolloutGroups: required,
		Lease: 30 * time.Second, RenewInterval: 10 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	return replica
}

func assertReplicaRestart(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	registry *providercatalog.Registry,
	groups []providercatalog.RolloutGroup,
	state catalogpostgres.State,
) {
	t.Helper()
	coordinator, err := catalogpostgres.New(db, registry)
	if err != nil {
		t.Fatal(err)
	}
	restarted, err := NewReplica(coordinator, registry, ReplicaOptions{
		ReplicaID: "router-cold-a", RolloutGroups: groups[:1], RequiredRolloutGroups: groups,
		Lease: 30 * time.Second, RenewInterval: 10 * time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	if coldStartErr := restarted.EnsureColdStart(ctx); coldStartErr != nil {
		t.Fatal(coldStartErr)
	}
	if reconcileErr := restarted.Reconcile(ctx); reconcileErr != nil {
		t.Fatal(reconcileErr)
	}
	restartedState, err := restarted.coordinator.State(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var revisions int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM provider_catalog_revisions`).Scan(&revisions); err != nil {
		t.Fatal(err)
	}
	if restartedState != state || revisions != 1 {
		t.Fatalf("restart changed catalog state: before=%+v after=%+v revisions=%d", state, restartedState, revisions)
	}
}

func TestReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStage(t *testing.T) {
	db := managedCatalogIntegrationDatabase(t)
	limited := managedTestRegistry(t, []string{"openai.chat.v1"})
	all := managedTestRegistry(t, []string{"openai.chat.v1", "anthropic.messages.v1"})
	limitedCoordinator, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr := catalogpostgres.New(db, limited)
	if testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr != nil {
		t.Fatal(testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr)
	}
	allCoordinator, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr := catalogpostgres.New(db, all)
	if testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr != nil {
		t.Fatal(testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr)
	}
	replica, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr := NewReplica(limitedCoordinator, limited, ReplicaOptions{
		ReplicaID: "router-limited", RolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
		RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
		Lease:                 30 * time.Second, RenewInterval: 10 * time.Second,
	})
	if testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr != nil {
		t.Fatal(testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()

	// Ordinary reconciliation is read-only. Runtime cold start calls the
	// separately bounded EnsureColdStart transaction before this lifecycle.
	if err := replica.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	var revisions int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM provider_catalog_revisions`).Scan(&revisions); err != nil {
		t.Fatal(err)
	}
	if revisions != 0 || replica.Readiness().Ready {
		t.Fatalf("startup mutated catalog or became ready: revisions=%d readiness=%+v", revisions, replica.Readiness())
	}

	openAI := limited.Snapshot()
	staged, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr := replica.BootstrapRegistry(ctx, 1)
	if testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr != nil {
		t.Fatal(testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr)
	}
	replayed, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr := replica.BootstrapRegistry(ctx, 1)
	if testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr != nil || replayed != staged {
		t.Fatalf("idempotent bootstrap = %+v, %v; want %+v", replayed, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr, staged)
	}
	if err := replica.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	if replica.Readiness().Ready {
		t.Fatal("replica became ready before the first revision was activated")
	}
	active, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr := replica.Activate(ctx, openAI.Revision(), staged.Generation)
	if testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr != nil {
		t.Fatal(testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr)
	}
	if err := replica.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	if !replica.Readiness().Ready || active.ActiveRevision != openAI.Revision() {
		t.Fatalf("active readiness = %+v, state=%+v", replica.Readiness(), active)
	}

	// Renewal replaces an expired lease using the stable replica identity and
	// capability digest.
	if _, err := db.ExecContext(ctx, `UPDATE provider_catalog_replica_acks
SET acknowledged_at = clock_timestamp() - interval '2 minutes',
    lease_expires_at = clock_timestamp() - interval '1 minute'
WHERE revision = $1 AND replica_id = 'router-limited'`, openAI.Revision()); err != nil {
		t.Fatal(err)
	}
	if err := replica.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	var renewed bool
	if err := db.QueryRowContext(ctx, `SELECT lease_expires_at > clock_timestamp()
FROM provider_catalog_replica_acks
WHERE revision = $1 AND replica_id = 'router-limited'`, openAI.Revision()).Scan(&renewed); err != nil {
		t.Fatal(err)
	}
	if !renewed {
		t.Fatal("replica compatibility lease was not renewed")
	}

	assertIncompatibleStageKeepsActiveReplica(t, ctx, limitedCoordinator, allCoordinator, replica, openAI, active)
}

func assertIncompatibleStageKeepsActiveReplica(
	t *testing.T,
	ctx context.Context,
	limitedCoordinator *catalogpostgres.Coordinator,
	allCoordinator *catalogpostgres.Coordinator,
	replica *Replica,
	openAI *providercatalog.Snapshot,
	active catalogpostgres.State,
) {
	t.Helper()
	// A publisher with an additional adapter can stage a revision this replica
	// cannot restore. The incompatible ACK blocks activation, while the known
	// good active revision continues to satisfy readiness.
	anthropic := managedTestSnapshot(t, "provider-anthropic", "anthropic.messages.v1")
	stagedBad, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr := allCoordinator.Stage(ctx, catalogpostgres.StageRequest{
		Snapshot: anthropic, ExpectedDesiredRevision: openAI.Revision(),
		ExpectedGeneration:    active.Generation,
		RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
	})
	if testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr != nil {
		t.Fatal(testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr)
	}
	if err := replica.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	readiness := replica.Readiness()
	if !readiness.Ready || readiness.ActiveRevision != openAI.Revision() ||
		readiness.DesiredRevision != anthropic.Revision() ||
		readiness.DesiredStatus != string(catalogpostgres.AckIncompatible) {
		t.Fatalf("bad staged revision drained active replica: %+v", readiness)
	}
	_, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr = allCoordinator.Activate(ctx, catalogpostgres.ActivateRequest{
		Revision: anthropic.Revision(), ExpectedGeneration: stagedBad.Generation,
	})
	var blocked *providercatalog.ActivationBlockedError
	if !errors.As(testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr, &blocked) || len(blocked.Blockers.Incompatible) != 1 ||
		blocked.Blockers.Incompatible[0].ReplicaID != "router-limited" {
		t.Fatalf("activation error = %v, blockers=%+v", testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr, blocked)
	}
	loaded, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr := limitedCoordinator.ActiveSnapshot(ctx)
	if testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr != nil || loaded.Revision() != openAI.Revision() {
		t.Fatalf("active snapshot changed after blocked activation: snapshot=%v error=%v", loaded, testReplicaLifecycleUsesDurableActiveSnapshotAndKeepsServingAcrossBadStageErr)
	}
}

func TestApplicationComposesInstalledRegistriesAndManagementServer(t *testing.T) {
	db := managedCatalogIntegrationDatabase(t)
	application := newManagedTestApplication(t, db)
	t.Cleanup(application.Close)
	if application.Catalog == nil || application.Discovery == nil || application.Replica == nil ||
		application.Coordinator == nil ||
		!application.Registry.HasWireFormat(string(llmprotocol.OpenAIChatV1)) {
		t.Fatalf("incomplete managed application = %#v", application)
	}
	server, testApplicationComposesInstalledRegistriesAndManagementServerErr := application.NewManagementServer(ManagementServerOptions{
		Namespaces: managementserver.NamespaceResolverFunc(func(context.Context, *http.Request) (string, error) {
			return "11111111-1111-4111-8111-111111111111", nil
		}),
		Sessions: integrationSessionAuthenticator{},
		Authorization: managementserver.AuthorizerFunc(func(
			context.Context,
			managementserver.AuthorizationRequest,
		) (managementserver.AuthorizationDecision, error) {
			return managementserver.AuthorizationDecision{}, nil
		}),
	})
	if testApplicationComposesInstalledRegistriesAndManagementServerErr != nil || server == nil {
		t.Fatalf("NewManagementServer() = %v, %v", server, testApplicationComposesInstalledRegistriesAndManagementServerErr)
	}
	assertManagedApplicationLifecycle(t, application, server)
}

func newManagedTestApplication(t *testing.T, db *sql.DB) *Application {
	t.Helper()
	discovery, err := providerdiscovery.BuiltinRegistry()
	if err != nil {
		t.Fatal(err)
	}
	policy, err := backendegress.Compile(backendegress.Config{
		Version: "v1", Schemes: []string{"https"},
		Hosts: []backendegress.HostConfig{{Host: "api.example.com", Ports: []uint16{443}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	claims, err := providerdiscovery.NewClaimCodec(providerdiscovery.ClaimKeyset{
		ActiveKeyID: "active", Keys: map[string][]byte{"active": bytes.Repeat([]byte{'c'}, 32)},
	})
	if err != nil {
		t.Fatal(err)
	}
	application, err := NewApplication(ApplicationOptions{
		DB: db, Registry: managedTestRegistry(t, []string{
			string(llmprotocol.OpenAIChatV1), string(llmprotocol.AnthropicMessagesV1),
		}),
		DiscoveryAdapters: discovery,
		EgressPolicy:      policy, CatalogCursorKeys: securitykeyring.Symmetric{
			ActiveVersion: "cursor-v1", Keys: map[string][]byte{"cursor-v1": bytes.Repeat([]byte{'d'}, 32)},
		},
		DiscoveryClaims: claims, DiscoveryClaimTTL: time.Minute,
		Replica: ReplicaOptions{
			ReplicaID: "router-application", RolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
			RequiredRolloutGroups: []providercatalog.RolloutGroup{dataRolloutGroup()},
			Lease:                 30 * time.Second, RenewInterval: 10 * time.Second,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return application
}

func assertManagedApplicationLifecycle(t *testing.T, application *Application, server *managementserver.Server) {
	t.Helper()
	if err := server.Ready(context.Background()); err == nil {
		t.Fatal("Management server was ready without an active Provider Catalog")
	}
	mux := http.NewServeMux()
	server.Register(mux)
	request := httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/provider-catalog:bootstrap",
		strings.NewReader(`{"expectedGeneration":"1"}`))
	request.Header.Set("Authorization", "Bearer management-token")
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set("Accept", managementapi.JSONMediaType)
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("bootstrap while catalog unready = %d, %s", response.Code, response.Body.String())
	}
	state, testApplicationComposesInstalledRegistriesAndManagementServerErr := application.Coordinator.State(context.Background())
	if testApplicationComposesInstalledRegistriesAndManagementServerErr != nil || state.DesiredRevision != application.Registry.Snapshot().Revision() ||
		state.ActiveRevision != "" || state.Generation != 2 {
		t.Fatalf("staged application Registry = %+v, %v", state, testApplicationComposesInstalledRegistriesAndManagementServerErr)
	}
	if _, err := application.Catalog.List(context.Background(), providercatalog.ListRequest{}); !errors.Is(err, catalogpostgres.ErrNoActiveSnapshot) {
		t.Fatalf("catalog read before explicit activation = %v", err)
	}
	if err := application.Replica.Reconcile(context.Background()); err != nil {
		t.Fatalf("acknowledge staged application Registry: %v", err)
	}
	activateBody := fmt.Sprintf(`{"revision":%q,"expectedGeneration":%q}`,
		state.DesiredRevision, fmt.Sprint(state.Generation))
	request = httptest.NewRequest(http.MethodPost, managementapi.BasePath+"/provider-catalog:activate",
		strings.NewReader(activateBody))
	request.Header.Set("Authorization", "Bearer management-token")
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	request.Header.Set("Accept", managementapi.JSONMediaType)
	response = httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("activate acknowledged Registry = %d, %s", response.Code, response.Body.String())
	}
	if err := application.Replica.Reconcile(context.Background()); err != nil {
		t.Fatalf("reconcile active application Registry: %v", err)
	}
	page, testApplicationComposesInstalledRegistriesAndManagementServerErr := application.Catalog.List(context.Background(), providercatalog.ListRequest{})
	if testApplicationComposesInstalledRegistriesAndManagementServerErr != nil || page.CatalogRevision != state.DesiredRevision || len(page.Providers) != 1 {
		t.Fatalf("catalog read after activation = %+v, %v", page, testApplicationComposesInstalledRegistriesAndManagementServerErr)
	}
	if err := server.Ready(context.Background()); err != nil {
		t.Fatalf("Management server not ready after explicit activation: %v", err)
	}
}

func dataRolloutGroup() providercatalog.RolloutGroup {
	return providercatalog.RolloutGroup{Plane: providercatalog.CapabilityPlaneData, ID: "router"}
}

func managedTestRegistry(t testing.TB, protocols []string) *providercatalog.Registry {
	t.Helper()
	definition := managedProviderDefinition("provider-registry", protocols[0])
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations: []providercatalog.Integration{providercatalog.IntegrationFunc(func() providercatalog.Definition {
			return definition
		})},
		BackendCompilers: []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		WireFormats:      protocols, CredentialAdapterIDs: []string{"api-key", "bearer", "x-api-key"},
		DiscoveryAdapterIDs: []string{"anthropic.models.v1", "openai.models.v1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return registry
}

func managedTestSnapshot(
	t testing.TB,
	providerID string,
	protocolID string,
) *providercatalog.Snapshot {
	t.Helper()
	definition := managedProviderDefinition(providerID, protocolID)
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations: []providercatalog.Integration{providercatalog.IntegrationFunc(func() providercatalog.Definition {
			return definition
		})},
		BackendCompilers: []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		WireFormats:      []string{protocolID},
	})
	if err != nil {
		t.Fatal(err)
	}
	return registry.Snapshot()
}

func managedProviderDefinition(providerID, protocolID string) providercatalog.Definition {
	return providercatalog.Definition{
		ID: providerID, Order: 1,
		Display: providercatalog.Display{
			Name: providerID, Description: "A declarative Provider.", Category: "Model APIs",
			Icon: providercatalog.Icon{Source: "lobe", Value: "provider", Color: false},
		},
		Interfaces: []providercatalog.Interface{{
			ID: "messages", Label: "Messages", Default: true,
			WireFormat: llmprotocol.WireFormat(protocolID),
			Compiler: providercatalog.Compiler{
				AdapterID: providercatalog.StaticBackendCompilerID, Config: map[string]any{"path": "/v1/messages"},
			},
		}},
		Credential: providercatalog.Credential{Mode: providercatalog.CredentialNone},
		Origin:     providercatalog.Origin{Mode: providercatalog.OriginFixed, DefaultURL: "https://api.example.com"},
	}
}

func managedCatalogIntegrationDatabase(t *testing.T) *sql.DB {
	t.Helper()
	dsn := os.Getenv("PROVIDERCATALOG_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("PROVIDERCATALOG_POSTGRES_DSN is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	t.Cleanup(cancel)
	admin, managedCatalogIntegrationDatabaseErr := sql.Open("postgres", dsn)
	if managedCatalogIntegrationDatabaseErr != nil {
		t.Fatal(managedCatalogIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatalf("ping PostgreSQL: %v", err)
	}
	schema := "provider_catalog_managed_it_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	scopedDSN, managedCatalogIntegrationDatabaseErr := managedCatalogDSNWithSearchPath(dsn, schema)
	if managedCatalogIntegrationDatabaseErr != nil {
		t.Fatal(managedCatalogIntegrationDatabaseErr)
	}
	db, managedCatalogIntegrationDatabaseErr := sql.Open("postgres", scopedDSN)
	if managedCatalogIntegrationDatabaseErr != nil {
		t.Fatal(managedCatalogIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatalf("apply control-plane migrations: %v", err)
	}
	return db
}

func managedCatalogDSNWithSearchPath(dsn, schema string) (string, error) {
	parsed, err := url.Parse(dsn)
	if err != nil {
		return "", err
	}
	if parsed.Scheme != "postgres" && parsed.Scheme != "postgresql" {
		return "", fmt.Errorf("PostgreSQL DSN must be a URL")
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}
