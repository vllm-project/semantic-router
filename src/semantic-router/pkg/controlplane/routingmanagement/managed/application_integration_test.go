package managed

import (
	"bytes"
	"context"
	"database/sql"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesspublisher"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/providercatalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementserver"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	routingHTTPPrincipalID = "11111111-1111-4111-8111-111111111111"
	routingHTTPAuthority   = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)

func TestRoutingHTTPPublishesExactClosureFromPostgresToRedis(t *testing.T) {
	db, namespaceID, partition := routingHTTPIntegrationDatabase(t)
	redisAddress := os.Getenv("ROUTINGMANAGEMENT_REDIS_ADDR")
	if redisAddress == "" {
		t.Skip("ROUTINGMANAGEMENT_REDIS_ADDR is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	catalog, registry := routingHTTPProviderCatalog(t)
	seedRoutingHTTPCatalogRevision(t, ctx, db, catalog.Revision())
	egress, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := backendegress.Compile(backendegress.Config{
		Version: "v1", Schemes: []string{"https"},
		Hosts: []backendegress.HostConfig{{Host: "models.example.com", Ports: []uint16{443}}},
	})
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil {
		t.Fatal(testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	commandCodec, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": bytes.Repeat([]byte{0x42}, 32)},
	})
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil {
		t.Fatal(testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	application, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := NewApplication(ApplicationOptions{
		DB: db,
		CursorKeyring: securitykeyring.Symmetric{
			ActiveVersion: "cursor-v1",
			Keys:          map[string][]byte{"cursor-v1": bytes.Repeat([]byte{0x24}, 32)},
		},
		ModelCompiler: providercatalog.ModelCompiler{
			Catalog: providercatalog.SnapshotSourceFunc(func(context.Context) (*providercatalog.Snapshot, error) {
				return catalog, nil
			}),
			Registry: registry,
			Egress:   egress,
		},
		ValidatePublication: config.ValidateManagedRoutingSnapshot,
		CommandCodec:        commandCodec,
		IdempotencyTTL:      time.Hour,
		Namespaces:          managementserver.ExplicitNamespaceResolver{},
		Sessions:            routingHTTPSession{},
		Authorization:       routingHTTPAuthorization{},
		BuiltInRecipes:      routingHTTPBuiltInRecipes(t),
	})
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil {
		t.Fatal(testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	if err := application.ReconcileBuiltInRecipes(ctx); err != nil {
		t.Fatal(err)
	}
	if err := application.Ready(ctx); err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	application.Register(mux)
	server := httptest.NewServer(mux)
	defer server.Close()

	routingHTTPMutation(t, server.URL, namespaceID, "/management/v1/routing/models", "model_one", `{
  "id":"model_one","name":"Model One",
  "execution":{"maxRetries":2,"requestTimeout":"30s","streamTimeout":"2m"},
  "pricing":{"inputCostPerMillionTokens":"1","outputCostPerMillionTokens":"2","cacheReadCostPerMillionTokens":null,"cacheWriteCostPerMillionTokens":null},
  "backends":[{"providerId":"provider_one","providerModelId":"upstream/model-one"}]
}`)
	routingHTTPMutation(t, server.URL, namespaceID, "/management/v1/routing/recipes", "recipe_one", `{
  "id":"recipe_one","name":"Recipe One",
  "document":{"signals":{},"projections":{},"decisions":[{"name":"Simple","rules":{}}]}
}`)
	routingHTTPMutation(t, server.URL, namespaceID, "/management/v1/routing/entrypoints", "entrypoint_one", `{
  "id":"entrypoint_one","name":"Entrypoint One","aliases":["vllm-sr/one"],
  "rules":[{"id":"rule_one","name":"Default","recipeId":"recipe_one","assignments":{"decision_one":{"models":[{"modelId":"model_one","weight":"1"}]}}}]
}`)

	publishRequest, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := http.NewRequest(http.MethodPost,
		server.URL+"/management/v1/routing/entrypoints/entrypoint_one:publish", nil)
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil {
		t.Fatal(testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	routingHTTPHeaders(publishRequest, namespaceID, "publish-entrypoint-one")
	publishRequest.Header.Set(managementapi.HeaderIfMatch, `"ep:1"`)
	publishResponse, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := http.DefaultClient.Do(publishRequest)
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil {
		t.Fatal(testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	defer publishResponse.Body.Close()
	publishBody, _ := io.ReadAll(publishResponse.Body)
	if publishResponse.StatusCode != http.StatusAccepted ||
		!strings.Contains(string(publishBody), `"desiredRevision":1`) {
		t.Fatalf("publish status = %d, body = %s", publishResponse.StatusCode, publishBody)
	}

	redisClient := redis.NewClient(&redis.Options{Addr: redisAddress})
	if err := redisClient.Ping(ctx).Err(); err != nil {
		t.Fatal(err)
	}
	prefix := "routing-http-it:" + strings.ReplaceAll(uuid.NewString(), "-", "")
	t.Cleanup(func() {
		deleteRoutingHTTPRedisPrefix(redisClient, prefix+":*")
		_ = redisClient.Close()
	})
	runtimeStore, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := accesspublisher.NewRedisStore(accesspublisher.RedisStoreOptions{
		Client: redisClient, KeyPrefix: prefix, ReplicaLease: 20 * time.Second,
	})
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil {
		t.Fatal(testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	outbox, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := accesspublisher.NewPostgresStore(db, accesspublisher.PostgresStoreOptions{Projector: "routing-http-it"})
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil {
		t.Fatal(testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	desired, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := accesspublisher.NewPostgresDesiredStateReader(db)
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil {
		t.Fatal(testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	engine, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := accesspublisher.NewEngine(accesspublisher.EngineOptions{
		Outbox: outbox, Desired: desired, Runtime: runtimeStore, WorkerID: "routing-http-worker",
		ClaimLease: 20 * time.Second, RetryDelay: time.Millisecond,
	})
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil {
		t.Fatal(testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	result, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := engine.ProcessOnce(ctx)
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil || result.Disposition != accesspublisher.ProcessApplied || result.Revision != 1 {
		t.Fatalf("ProcessOnce() = %+v, %v", result, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}
	readiness, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr := runtimeStore.Readiness(ctx, namespaceID, partition)
	if testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr != nil || !readiness.Ready || readiness.AppliedRevision != 1 ||
		readiness.AccessGate == "" || readiness.AccessGate != readiness.RoutingGate {
		t.Fatalf("published readiness = %+v, %v", readiness, testRoutingHTTPPublishesExactClosureFromPostgresToRedisErr)
	}

	var modelCount, recipeCount, entrypointCount int
	if err := db.QueryRowContext(ctx, `SELECT
  count(*) FILTER (WHERE resource_type='model'),
  count(*) FILTER (WHERE resource_type='recipe'),
  count(*) FILTER (WHERE resource_type='entrypoint')
FROM routing_snapshot_members WHERE namespace_id=$1 AND routing_revision=1`, namespaceID).
		Scan(&modelCount, &recipeCount, &entrypointCount); err != nil {
		t.Fatal(err)
	}
	if modelCount != 1 || recipeCount != 1 || entrypointCount != 1 {
		t.Fatalf("published closure sizes = model %d, recipe %d, entrypoint %d",
			modelCount, recipeCount, entrypointCount)
	}
	var auditCount, commandCount, outboxCount int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM access_audit_events WHERE namespace_id=$1`, namespaceID).
		Scan(&auditCount); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM management_idempotency WHERE namespace_id=$1`, namespaceID).
		Scan(&commandCount); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM policy_outbox WHERE namespace_id=$1`, namespaceID).
		Scan(&outboxCount); err != nil {
		t.Fatal(err)
	}
	if auditCount != 5 || commandCount != 4 || outboxCount != 1 {
		t.Fatalf("durable mutation facts = audit %d, commands %d, outbox %d",
			auditCount, commandCount, outboxCount)
	}
}

func routingHTTPBuiltInRecipes(t *testing.T) routingmanagement.BuiltInRecipeDistribution {
	t.Helper()
	distribution, err := routingmanagement.ParseBuiltInRecipeDistribution(
		[]byte("schema_version: vllm-sr/recipe-metadata/v1\nid: http-recipes\nname: HTTP Recipes\nversion: 1.0.0\n"),
		[]byte(`version: v0.4
recipes:
  - name: Built-in HTTP
    description: Integration fixture.
    document:
      decisions:
        - name: Built-in
          rules: {}
`),
	)
	if err != nil {
		t.Fatal(err)
	}
	return distribution
}

type routingHTTPAuthorization struct{}

func (routingHTTPAuthorization) Authorize(
	context.Context,
	managementserver.AuthorizationRequest,
) (managementserver.AuthorizationDecision, error) {
	return managementserver.AuthorizationDecision{AuthorityDigest: routingHTTPAuthority}, nil
}

func (routingHTTPAuthorization) ResolveResultScope(
	_ context.Context,
	_ accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
	_ accesscontrol.Permission,
) (managementauthorization.ResultScope, error) {
	return managementauthorization.ResultScope{NamespaceID: namespaceID, All: true}, nil
}

type routingHTTPSession struct{}

func (routingHTTPSession) Authenticate(
	_ context.Context, _ string, namespaceID string, _ time.Time,
) (managementauth.AuthenticatedSession, error) {
	return managementauth.AuthenticatedSession{
		NamespaceID: namespaceID,
		Session: managementauth.LiveSession{Session: managementauth.Session{
			PrincipalID: routingHTTPPrincipalID,
		}},
	}, nil
}

func routingHTTPMutation(t *testing.T, baseURL, namespaceID, endpoint, resourceID, body string) {
	t.Helper()
	request, err := http.NewRequest(http.MethodPost, baseURL+endpoint, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", managementapi.JSONMediaType)
	routingHTTPHeaders(request, namespaceID, "create-"+resourceID)
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	payload, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusCreated || !strings.Contains(string(payload), `"id":"`+resourceID+`"`) {
		t.Fatalf("POST %s status = %d, body = %s", endpoint, response.StatusCode, payload)
	}
}

func routingHTTPHeaders(request *http.Request, namespaceID, idempotencyKey string) {
	request.Header.Set("Authorization", "Bearer integration-session")
	request.Header.Set(managementapi.HeaderNamespaceID, namespaceID)
	request.Header.Set(managementapi.HeaderRequestID, uuid.NewString())
	request.Header.Set(managementapi.HeaderIdempotencyKey, idempotencyKey)
}

func routingHTTPProviderCatalog(t *testing.T) (*providercatalog.Snapshot, *providercatalog.Registry) {
	t.Helper()
	definition := providercatalog.Definition{
		ID: "provider_one", Display: providercatalog.Display{
			Name: "Provider One", Description: "Integration provider", Category: "test",
			Icon: providercatalog.Icon{Source: "lobe", Value: "test", Color: false},
		},
		Interfaces: []providercatalog.Interface{{
			ID: "chat", Label: "Chat Completions", Default: true, WireFormat: "openai.chat.v1",
			Compiler: providercatalog.Compiler{
				AdapterID: providercatalog.StaticBackendCompilerID,
				Config:    map[string]any{"path": "/chat/completions"},
			},
		}},
		Credential: providercatalog.Credential{Mode: providercatalog.CredentialNone},
		Origin:     providercatalog.Origin{Mode: providercatalog.OriginFixed, DefaultURL: "https://models.example.com/v1"},
	}
	registry, err := providercatalog.NewRegistry(providercatalog.RegistryOptions{
		Integrations: []providercatalog.Integration{providercatalog.IntegrationFunc(func() providercatalog.Definition {
			return definition
		})},
		BackendCompilers: []providercatalog.BackendCompiler{providercatalog.StaticBackendCompiler{}},
		WireFormats:      []string{"openai.chat.v1"},
	})
	if err != nil {
		t.Fatal(err)
	}
	return registry.Snapshot(), registry
}

func routingHTTPIntegrationDatabase(t *testing.T) (*sql.DB, string, string) {
	t.Helper()
	dsn := os.Getenv("ROUTINGMANAGEMENT_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("ROUTINGMANAGEMENT_POSTGRES_DSN is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	t.Cleanup(cancel)
	admin, routingHTTPIntegrationDatabaseErr := sql.Open("postgres", dsn)
	if routingHTTPIntegrationDatabaseErr != nil {
		t.Fatal(routingHTTPIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	schema := "routing_http_it_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	parsed, routingHTTPIntegrationDatabaseErr := url.Parse(dsn)
	if routingHTTPIntegrationDatabaseErr != nil {
		t.Fatal(routingHTTPIntegrationDatabaseErr)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	db, routingHTTPIntegrationDatabaseErr := sql.Open("postgres", parsed.String())
	if routingHTTPIntegrationDatabaseErr != nil {
		t.Fatal(routingHTTPIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = db.Close() })
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatalf("apply control-plane migrations: %v", err)
	}
	if _, err := db.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,status)
VALUES ($1,'https://issuer.example.com','routing-http-principal','Routing HTTP Principal','active')`,
		routingHTTPPrincipalID); err != nil {
		t.Fatal(err)
	}
	namespaceID := uuid.NewString()
	partition := "quota-" + strings.ReplaceAll(namespaceID, "-", "")
	if _, err := db.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,$2,$3,'USD','active')`, namespaceID, "namespace-"+namespaceID, partition); err != nil {
		t.Fatal(err)
	}
	return db, namespaceID, partition
}

func seedRoutingHTTPCatalogRevision(t *testing.T, ctx context.Context, db *sql.DB, revision string) {
	t.Helper()
	if _, err := db.ExecContext(ctx, `INSERT INTO provider_catalog_revisions
	  (revision,snapshot_bytes,snapshot_digest,integration_references,catalog,
   required_wire_formats,required_credential_adapters,required_discovery_adapters)
VALUES ($1,'x',decode(repeat('aa',32),'hex'),'[]'::jsonb,'{}'::jsonb,
  '["openai.chat.v1"]'::jsonb,'[]'::jsonb,'[]'::jsonb)`, revision); err != nil {
		t.Fatal(err)
	}
}

func deleteRoutingHTTPRedisPrefix(client *redis.Client, pattern string) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	var cursor uint64
	for {
		keys, next, err := client.Scan(ctx, cursor, pattern, 100).Result()
		if err != nil {
			return
		}
		if len(keys) != 0 {
			_ = client.Del(ctx, keys...).Err()
		}
		cursor = next
		if cursor == 0 {
			return
		}
	}
}
