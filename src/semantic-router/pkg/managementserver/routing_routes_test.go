package managementserver

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type routingServiceStub struct {
	models                routingmanagement.Page[routingmanagement.Model]
	recipes               routingmanagement.Page[routingmanagement.Recipe]
	entrypoints           routingmanagement.Page[routingmanagement.Entrypoint]
	model                 routingmanagement.Model
	recipe                routingmanagement.Recipe
	entrypoint            routingmanagement.Entrypoint
	snapshots             routingmanagement.Page[routingmanagement.SnapshotMetadata]
	snapshot              routingmanagement.SnapshotDetail
	resolution            routingsnapshot.Resolution
	modelListRequest      routingmanagement.PageRequest
	recipeListRequest     routingmanagement.PageRequest
	entrypointListRequest routingmanagement.PageRequest
	snapshotListRequest   routingmanagement.SnapshotPageRequest
	snapshotRevision      int64
	createCalls           int
	deleteCalls           int
	modelPatch            routingmanagement.ModelPatch
	manifestCredentials   []string
	manifestRequest       routingmanagement.ManifestImportRequest
	manifestMutation      routingmanagement.MutationContext
	manifestResult        routingmanagement.ManifestImportResult
}

func (stub *routingServiceStub) ManifestCredentialIDs([]byte) ([]string, error) {
	return append([]string(nil), stub.manifestCredentials...), nil
}

func (stub *routingServiceStub) ImportManifest(
	_ context.Context, _ string, request routingmanagement.ManifestImportRequest, mutation routingmanagement.MutationContext,
) (routingmanagement.ManifestImportResult, error) {
	stub.manifestRequest = request
	stub.manifestMutation = mutation
	if stub.manifestResult.Receipt.OperationID == "" {
		stub.manifestResult.Receipt = routingmanagement.RevisionReceipt{
			OperationID: "44444444-4444-4444-8444-444444444444", DesiredRevision: 8,
		}
	}
	return stub.manifestResult, nil
}

func (stub *routingServiceStub) ExportCurrentManifest(context.Context, string) ([]byte, int64, error) {
	return []byte("version: v0.3\n"), 7, nil
}

func (stub *routingServiceStub) ListModels(_ context.Context, _ string, request routingmanagement.PageRequest) (routingmanagement.Page[routingmanagement.Model], error) {
	stub.modelListRequest = request
	return stub.models, nil
}

func (stub *routingServiceStub) GetModel(_ context.Context, _, id string) (routingmanagement.Model, error) {
	if stub.model.ID == id {
		return stub.model, nil
	}
	return routingmanagement.Model{}, routingmanagement.ErrNotFound
}

func (stub *routingServiceStub) CreateModel(context.Context, string, routingmanagement.ModelInput, routingmanagement.MutationContext) (routingmanagement.Model, routingmanagement.RevisionReceipt, error) {
	stub.createCalls++
	return stub.model, routingmanagement.RevisionReceipt{ResourceRevision: 1}, nil
}

func (stub *routingServiceStub) PatchModel(_ context.Context, _, _ string, _ int64, patch routingmanagement.ModelPatch, _ routingmanagement.MutationContext) (routingmanagement.Model, routingmanagement.RevisionReceipt, error) {
	stub.modelPatch = patch
	return stub.model, routingmanagement.RevisionReceipt{ResourceRevision: 2}, nil
}

func (stub *routingServiceStub) DeleteModel(context.Context, string, string, int64, routingmanagement.MutationContext) (routingmanagement.RevisionReceipt, error) {
	stub.deleteCalls++
	return routingmanagement.RevisionReceipt{ResourceRevision: 2}, nil
}

func (stub *routingServiceStub) BulkImport(context.Context, routingmanagement.BulkImportRequest, routingmanagement.MutationContext) ([]routingmanagement.Model, routingmanagement.RevisionReceipt, error) {
	return nil, routingmanagement.RevisionReceipt{OperationID: "44444444-4444-4444-8444-444444444444"}, nil
}

func (stub *routingServiceStub) ProbeModel(context.Context, string, string, time.Duration) (routingmanagement.ProbeResult, error) {
	return routingmanagement.ProbeResult{Available: true, Latency: time.Millisecond, CheckedAt: time.Unix(5, 0)}, nil
}

func (stub *routingServiceStub) ListRecipes(_ context.Context, _ string, request routingmanagement.PageRequest) (routingmanagement.Page[routingmanagement.Recipe], error) {
	stub.recipeListRequest = request
	return stub.recipes, nil
}

func (stub *routingServiceStub) GetRecipe(_ context.Context, _, id string) (routingmanagement.Recipe, error) {
	if stub.recipe.ID == id {
		return stub.recipe, nil
	}
	return routingmanagement.Recipe{}, routingmanagement.ErrNotFound
}

func (stub *routingServiceStub) CreateRecipe(context.Context, string, routingmanagement.RecipeInput, routingmanagement.MutationContext) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error) {
	return stub.recipe, routingmanagement.RevisionReceipt{ResourceRevision: 1}, nil
}

func (stub *routingServiceStub) UpdateRecipe(context.Context, string, string, int64, routingmanagement.RecipeInput, routingmanagement.MutationContext) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error) {
	return stub.recipe, routingmanagement.RevisionReceipt{ResourceRevision: 2}, nil
}

func (stub *routingServiceStub) DeleteRecipe(context.Context, string, string, int64, routingmanagement.MutationContext) (routingmanagement.RevisionReceipt, error) {
	return routingmanagement.RevisionReceipt{ResourceRevision: 2}, nil
}

func (stub *routingServiceStub) ListEntrypoints(_ context.Context, _ string, request routingmanagement.PageRequest) (routingmanagement.Page[routingmanagement.Entrypoint], error) {
	stub.entrypointListRequest = request
	return stub.entrypoints, nil
}

func (stub *routingServiceStub) GetEntrypoint(_ context.Context, _, id string) (routingmanagement.Entrypoint, error) {
	if stub.entrypoint.ID == id {
		return stub.entrypoint, nil
	}
	return routingmanagement.Entrypoint{}, routingmanagement.ErrNotFound
}

func (stub *routingServiceStub) CreateEntrypoint(context.Context, string, routingmanagement.EntrypointInput, routingmanagement.MutationContext) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error) {
	return stub.entrypoint, routingmanagement.RevisionReceipt{ResourceRevision: 1}, nil
}

func (stub *routingServiceStub) UpdateEntrypoint(context.Context, string, string, int64, routingmanagement.EntrypointInput, routingmanagement.MutationContext) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error) {
	return stub.entrypoint, routingmanagement.RevisionReceipt{ResourceRevision: 2}, nil
}

func (stub *routingServiceStub) DeleteEntrypoint(context.Context, string, string, int64, routingmanagement.MutationContext) (routingmanagement.RevisionReceipt, error) {
	return routingmanagement.RevisionReceipt{ResourceRevision: 2}, nil
}

func (stub *routingServiceStub) PublishEntrypoint(context.Context, string, string, int64, routingmanagement.MutationContext) (*routingsnapshot.Snapshot, routingmanagement.RevisionReceipt, error) {
	return nil, routingmanagement.RevisionReceipt{
		ResourceRevision: 2, DesiredRevision: 7, OperationID: "44444444-4444-4444-8444-444444444444",
	}, nil
}

func (stub *routingServiceStub) UnpublishEntrypoint(context.Context, string, string, int64, routingmanagement.MutationContext) (*routingsnapshot.Snapshot, routingmanagement.RevisionReceipt, error) {
	return stub.PublishEntrypoint(context.Background(), "", "", 0, routingmanagement.MutationContext{})
}

func (stub *routingServiceStub) ResolveEntrypoint(context.Context, string, string, string, map[string]routingsnapshot.ClaimValue) (routingsnapshot.Resolution, error) {
	return stub.resolution, nil
}

func (stub *routingServiceStub) ListSnapshots(_ context.Context, _ string, request routingmanagement.SnapshotPageRequest) (routingmanagement.Page[routingmanagement.SnapshotMetadata], error) {
	stub.snapshotListRequest = request
	return stub.snapshots, nil
}

func (stub *routingServiceStub) GetSnapshot(_ context.Context, _ string, revision int64) (routingmanagement.SnapshotDetail, error) {
	stub.snapshotRevision = revision
	return stub.snapshot, nil
}

type routingCommandResultsStub struct {
	stored  managementcommand.StoredResult
	diff    routingmanagement.ManifestDiff
	found   bool
	err     error
	command managementcommand.Command
}

func (stub *routingCommandResultsStub) Lookup(_ context.Context, command managementcommand.Command) (managementcommand.StoredResult, bool, error) {
	stub.command = command
	return stub.stored, stub.found, stub.err
}

func (stub *routingCommandResultsStub) LookupManifestDiff(context.Context, string, string) (routingmanagement.ManifestDiff, error) {
	return stub.diff, stub.err
}

func (*routingCommandResultsStub) Ready(context.Context, *managementcommand.Codec) error { return nil }

type routingAuthorizerStub struct {
	requests  []AuthorizationRequest
	authorize func(AuthorizationRequest) (AuthorizationDecision, error)
}

func (stub *routingAuthorizerStub) Authorize(_ context.Context, request AuthorizationRequest) (AuthorizationDecision, error) {
	stub.requests = append(stub.requests, request)
	if stub.authorize != nil {
		return stub.authorize(request)
	}
	return AuthorizationDecision{AuthorityDigest: testAuthority}, nil
}

func TestRoutingManifestDryRunAuthorizesCredentialsAndReturnsTypedDiff(t *testing.T) {
	credentialID := "11111111-1111-4111-8111-111111111111"
	service := &routingServiceStub{
		manifestCredentials: []string{credentialID},
		manifestResult: routingmanagement.ManifestImportResult{Diff: routingmanagement.ManifestDiff{
			Models: routingmanagement.ManifestResourceDiff{Create: []string{"mdl_new"}},
		}},
	}
	authorizer := &routingAuthorizerStub{}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, authorizer)
	response := serveRoutingRequest(t, routes, http.MethodPost, routingImportsPath,
		strings.NewReader(`{"manifest":"version: v0.3\n","dryRun":true}`), map[string]string{
			managementapi.HeaderIfMatch:        `"routing:7"`,
			managementapi.HeaderIdempotencyKey: "routing-import-preview",
		})
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if service.manifestRequest.ExpectedRevision != 7 || !service.manifestRequest.DryRun ||
		service.manifestMutation.Command != nil {
		t.Fatalf("manifest request = %#v, mutation = %#v", service.manifestRequest, service.manifestMutation)
	}
	if !strings.Contains(response.Body.String(), `"create":["mdl_new"]`) ||
		!strings.Contains(response.Body.String(), `"entrypoints":{"create":[],"update":[],"disable":[]}`) {
		t.Fatalf("typed diff = %s", response.Body.String())
	}
	if len(authorizer.requests) != 1 ||
		!authorizer.requests[0].Conditions["provider_credential_referenced"] ||
		len(authorizer.requests[0].Targets["credential"]) != 1 ||
		string(authorizer.requests[0].Targets["credential"][0].Scope.ResourceID) != credentialID {
		t.Fatalf("authorization request = %#v", authorizer.requests)
	}
}

func TestRoutingManifestExportReturnsPortableYAMLAndRevision(t *testing.T) {
	routes := newTestRoutingRoutes(t, &routingServiceStub{}, &routingCommandResultsStub{}, &routingAuthorizerStub{})
	response := serveRoutingRequest(t, routes, http.MethodGet, routingCurrentExportPath, nil, nil)
	if response.Code != http.StatusOK || response.Body.String() != "version: v0.3\n" {
		t.Fatalf("status = %d, body = %q", response.Code, response.Body.String())
	}
	if got := response.Header().Get("Content-Type"); got != managementapi.YAMLMediaType+"; charset=utf-8" {
		t.Fatalf("Content-Type = %q", got)
	}
	if got := response.Header().Get(managementapi.HeaderETag); got != `"routing:7"` {
		t.Fatalf("ETag = %q", got)
	}
}

func TestRoutingEntrypointListPushesExactScopeBeforePagination(t *testing.T) {
	first := testRoutingEntrypoint("entrypoint_one")
	first.RuleCount, first.AssignedModelCount = 1, 1
	service := &routingServiceStub{entrypoints: routingmanagement.Page[routingmanagement.Entrypoint]{Items: []routingmanagement.Entrypoint{first}}}
	authorizer := &routingAuthorizerStub{authorize: func(AuthorizationRequest) (AuthorizationDecision, error) {
		return AuthorizationDecision{}, managementauthorization.ErrDenied
	}}
	scopes := resultScopeResolverFunc(func(
		_ context.Context,
		_ accesscontrol.ManagementPrincipalID,
		namespaceID accesscontrol.NamespaceID,
		permission accesscontrol.Permission,
	) (managementauthorization.ResultScope, error) {
		if permission != accesscontrol.PermissionRoutingRead {
			t.Fatalf("permission = %q", permission)
		}
		return managementauthorization.ResultScope{NamespaceID: namespaceID, ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceEntrypoint: {"entrypoint_one"},
		}}, nil
	})
	routes := newTestRoutingRoutesWithScopes(t, service, &routingCommandResultsStub{}, authorizer, scopes)

	response := serveRoutingRequest(t, routes, http.MethodGet, routingEntrypointsPath, nil, nil)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	wire := response.Body.String()
	if !strings.Contains(wire, "entrypoint_one") || !strings.Contains(wire, `"ruleCount":1`) ||
		!strings.Contains(wire, `"assignedModelCount":1`) || strings.Contains(wire, "rules") ||
		strings.Contains(wire, "model_one") {
		t.Fatalf("scoped Entrypoint page = %s", wire)
	}
	ids := service.entrypointListRequest.Scope.IDs(accesscontrol.ScopeResourceEntrypoint)
	if len(ids) != 1 || ids[0] != "entrypoint_one" || len(authorizer.requests) != 0 {
		t.Fatalf("repository scope = %#v, per-row authorization calls = %d", service.entrypointListRequest.Scope, len(authorizer.requests))
	}
}

func TestRoutingListsForwardTheAuthorizedTypedScope(t *testing.T) {
	for _, test := range []struct {
		name         string
		path         string
		resourceType accesscontrol.ScopeResourceType
		request      func(*routingServiceStub) routingmanagement.PageRequest
	}{
		{"Models", routingModelsPath, accesscontrol.ScopeResourceModel, func(stub *routingServiceStub) routingmanagement.PageRequest { return stub.modelListRequest }},
		{"ModelCards", routingModelCardsPath, accesscontrol.ScopeResourceModel, func(stub *routingServiceStub) routingmanagement.PageRequest { return stub.modelListRequest }},
		{"Recipes", routingRecipesPath, accesscontrol.ScopeResourceRecipe, func(stub *routingServiceStub) routingmanagement.PageRequest { return stub.recipeListRequest }},
		{"Entrypoints", routingEntrypointsPath, accesscontrol.ScopeResourceEntrypoint, func(stub *routingServiceStub) routingmanagement.PageRequest { return stub.entrypointListRequest }},
	} {
		t.Run(test.name, func(t *testing.T) {
			service := &routingServiceStub{}
			scopes := resultScopeResolverFunc(func(
				_ context.Context,
				_ accesscontrol.ManagementPrincipalID,
				namespaceID accesscontrol.NamespaceID,
				_ accesscontrol.Permission,
			) (managementauthorization.ResultScope, error) {
				return managementauthorization.ResultScope{
					NamespaceID: namespaceID,
					ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
						test.resourceType: {"resource_one"},
					},
				}, nil
			})
			routes := newTestRoutingRoutesWithScopes(
				t, service, &routingCommandResultsStub{}, &routingAuthorizerStub{}, scopes,
			)
			response := serveRoutingRequest(t, routes, http.MethodGet, test.path, nil, nil)
			if response.Code != http.StatusOK {
				t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
			}
			ids := test.request(service).Scope.IDs(test.resourceType)
			if len(ids) != 1 || ids[0] != "resource_one" {
				t.Fatalf("forwarded scope = %#v", test.request(service).Scope)
			}
		})
	}
}

func TestRoutingSnapshotListUsesNamespaceWideReadAndCursorPagination(t *testing.T) {
	createdAt := time.Unix(10, 0).UTC()
	service := &routingServiceStub{snapshots: routingmanagement.Page[routingmanagement.SnapshotMetadata]{
		Items: []routingmanagement.SnapshotMetadata{{
			NamespaceID: testNamespaceID, RoutingRevision: 9,
			ContentDigest: "sha256:" + strings.Repeat("a", 64),
			Status:        routingmanagement.SnapshotStatusActive, MemberCount: 3, CreatedAt: createdAt,
		}},
		NextCursor: "next-snapshot", HasMore: true,
	}}
	authorizer := &routingAuthorizerStub{}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, authorizer)
	path := managementapi.BasePath + "/namespaces/" + testNamespaceID + "/routing/snapshots?pageSize=17&cursor=prior"
	response := serveRoutingRequest(t, routes, http.MethodGet, path, nil, nil)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if service.snapshotListRequest.PageSize != 17 || service.snapshotListRequest.Cursor != "prior" {
		t.Fatalf("snapshot page request = %#v", service.snapshotListRequest)
	}
	if len(authorizer.requests) != 1 || authorizer.requests[0].Operation.Permission.Canonical() != "routing.read@path_namespace" {
		t.Fatalf("snapshot authorization = %#v", authorizer.requests)
	}
	for _, required := range []string{
		`"routingRevision":9`, `"status":"active"`, `"memberCount":3`,
		`"nextCursor":"next-snapshot"`, `"pageSize":17`,
	} {
		if !strings.Contains(response.Body.String(), required) {
			t.Fatalf("snapshot page omitted %s: %s", required, response.Body.String())
		}
	}
}

func TestRoutingSnapshotDetailReturnsImmutableMembersAndExport(t *testing.T) {
	service := &routingServiceStub{snapshot: routingmanagement.SnapshotDetail{
		Metadata: routingmanagement.SnapshotMetadata{
			NamespaceID: testNamespaceID, RoutingRevision: 7,
			ContentDigest: "sha256:" + strings.Repeat("b", 64),
			Status:        routingmanagement.SnapshotStatusRetired, MemberCount: 1, CreatedAt: time.Unix(10, 0).UTC(),
		},
		Members: []routingmanagement.SnapshotMember{{
			ResourceType: "model", ResourceID: "model_one", ResourceRevision: 4,
		}},
		Export: routingsnapshot.Snapshot{Bundle: routingsnapshot.Bundle{
			NamespaceID: testNamespaceID, Revision: 7,
			Models: []routingsnapshot.Model{{
				ID: "model_one", Revision: 4,
				Execution: routingsnapshot.ModelExecution{
					MaxRetries: 2, RetryOn: []string{"unavailable", "timeout"},
					RequestTimeout: "30s", StreamTimeout: "2m",
				},
			}},
		}, Digest: strings.Repeat("b", 64)},
	}}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, &routingAuthorizerStub{})
	path := managementapi.BasePath + "/namespaces/" + testNamespaceID + "/routing/snapshots/7"
	response := serveRoutingRequest(t, routes, http.MethodGet, path, nil, nil)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if service.snapshotRevision != 7 {
		t.Fatalf("snapshot revision = %d", service.snapshotRevision)
	}
	for _, required := range []string{
		`"metadata"`, `"members"`, `"resourceId":"model_one"`,
		`"resourceRevision":4`, `"export"`, `"revision":7`,
		`"control":{"retry":{"count":2,"on":["unavailable","timeout"]},"timeout":{"request":"30s","stream":"2m"}}`,
	} {
		if !strings.Contains(response.Body.String(), required) {
			t.Fatalf("snapshot detail omitted %s: %s", required, response.Body.String())
		}
	}
	if strings.Contains(response.Body.String(), `"execution"`) {
		t.Fatalf("snapshot detail exposed internal execution storage: %s", response.Body.String())
	}
}

func TestRoutingSnapshotDetailAuthorizesBeforeReadingTheExport(t *testing.T) {
	service := &routingServiceStub{}
	authorizer := &routingAuthorizerStub{authorize: func(AuthorizationRequest) (AuthorizationDecision, error) {
		return AuthorizationDecision{}, managementauthorization.ErrDenied
	}}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, authorizer)
	path := managementapi.BasePath + "/namespaces/" + testNamespaceID + "/routing/snapshots/7"
	response := serveRoutingRequest(t, routes, http.MethodGet, path, nil, nil)
	if response.Code != http.StatusForbidden || service.snapshotRevision != 0 {
		t.Fatalf("status = %d, snapshot reads = %d, body = %s", response.Code, service.snapshotRevision, response.Body.String())
	}
}

func TestRoutingModelCardListUsesSemanticProjection(t *testing.T) {
	service := &routingServiceStub{models: routingmanagement.Page[routingmanagement.Model]{
		Items: []routingmanagement.Model{{
			ResourceIdentity: routingmanagement.ResourceIdentity{
				ID: "model_one", Name: "Model One", Status: routingmanagement.StatusActive,
				Revision: 2, CreatedAt: time.Unix(1, 0), UpdatedAt: time.Unix(2, 0),
			},
			Current: routingsnapshot.Model{
				ID: "model_one", Revision: 3, CatalogRevision: "sha256:" + strings.Repeat("a", 64),
				Capabilities: []string{"reasoning"},
				Execution:    routingsnapshot.ModelExecution{RequestTimeout: "5s"},
				Backends: []routingsnapshot.Backend{{
					ProviderID: "private-provider", ProviderModelID: "private/model",
				}},
			},
		}},
	}}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, &routingAuthorizerStub{})
	response := serveRoutingRequest(t, routes, http.MethodGet, routingModelCardsPath, nil, nil)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	wire := response.Body.String()
	for _, required := range []string{`"id":"model_one"`, `"name":"Model One"`, `"card"`, `"capabilities":["reasoning"]`} {
		if !strings.Contains(wire, required) {
			t.Errorf("Model Card page omitted %q: %s", required, wire)
		}
	}
	for _, forbidden := range []string{
		"status", "revision", "catalogRevision", "control", "execution", "backends", "private-provider", "private/model",
	} {
		if strings.Contains(wire, forbidden) {
			t.Errorf("Model Card page leaked %q: %s", forbidden, wire)
		}
	}
}

func TestRoutingEntrypointDetailRequiresEveryDependency(t *testing.T) {
	entrypoint := testRoutingEntrypoint("entrypoint_one")
	service := &routingServiceStub{entrypoint: entrypoint}
	authorizer := &routingAuthorizerStub{}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, authorizer)

	response := serveRoutingRequest(t, routes, http.MethodGet,
		routingEntrypointsPath+"/entrypoint_one?includeTopology=true", nil, nil)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	request := authorizer.requests[len(authorizer.requests)-1]
	dependencies := request.Targets["all_dependencies"]
	if len(dependencies) != 2 || !request.Conditions["entrypoint_topology_requested"] {
		t.Fatalf("topology authorization = %#v", request)
	}
	if !strings.Contains(response.Body.String(), "recipe_one") || !strings.Contains(response.Body.String(), "model_one") {
		t.Fatalf("authorized topology missing: %s", response.Body.String())
	}
}

func TestRoutingModelCreateReplaysBeforeMutableCompilation(t *testing.T) {
	service := &routingServiceStub{}
	commandResults := &routingCommandResultsStub{found: true, stored: managementcommand.StoredResult{
		Resource: &managementcommand.ResourceResult{
			ResourceType: "routing_model", ResourceID: "model_replayed", ResourceRevision: 4,
			ResponseStatus: http.StatusCreated,
		},
		ExpiresAt: time.Now().Add(time.Hour),
	}}
	routes := newTestRoutingRoutes(t, service, commandResults, &routingAuthorizerStub{})
	body := `{"name":"Replay","backends":[{"providerId":"provider","providerModelId":"model","connectionFields":{"token":"must-not-return"}}]}`
	headers := map[string]string{managementapi.HeaderIdempotencyKey: "idempotency-key-0001"}
	response := serveRoutingRequest(t, routes, http.MethodPost, routingModelsPath, strings.NewReader(body), headers)
	if response.Code != http.StatusCreated {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if service.createCalls != 0 {
		t.Fatalf("CreateModel called %d times on durable replay", service.createCalls)
	}
	if commandResults.command.Scope != managementcommand.NamespaceCommandScope(testNamespaceID) ||
		commandResults.command.PrincipalID != testPrincipalID {
		t.Fatalf("command scope = %#v", commandResults.command)
	}
	if response.Header().Get(managementapi.HeaderIdempotencyReplayed) != "true" ||
		!strings.Contains(response.Body.String(), "model_replayed") ||
		strings.Contains(response.Body.String(), "must-not-return") {
		t.Fatalf("replay response = headers %#v, body %s", response.Header(), response.Body.String())
	}
}

func TestRoutingModelDeleteRequiresTypedETag(t *testing.T) {
	service := &routingServiceStub{}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, &routingAuthorizerStub{})

	missing := serveRoutingRequest(t, routes, http.MethodDelete, routingModelsPath+"/model_one", nil, nil)
	if missing.Code != http.StatusPreconditionRequired || service.deleteCalls != 0 {
		t.Fatalf("missing If-Match status = %d, calls = %d", missing.Code, service.deleteCalls)
	}
	response := serveRoutingRequest(t, routes, http.MethodDelete, routingModelsPath+"/model_one", nil,
		map[string]string{managementapi.HeaderIfMatch: `"mdl:1"`})
	if response.Code != http.StatusNoContent || response.Header().Get(managementapi.HeaderETag) != `"mdl:2"` || service.deleteCalls != 1 {
		t.Fatalf("delete = status %d, etag %q, calls %d, body %s", response.Code,
			response.Header().Get(managementapi.HeaderETag), service.deleteCalls, response.Body.String())
	}
}

func TestRoutingModelPatchDoesNotRequireBackendOrCredentialRoundTrip(t *testing.T) {
	service := &routingServiceStub{}
	authorizer := &routingAuthorizerStub{}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, authorizer)

	response := serveRoutingRequest(t, routes, http.MethodPatch, routingModelsPath+"/model_one",
		strings.NewReader(`{"control":{"retry":{"count":4,"on":["unavailable"]},"timeout":{"request":"45s","stream":"5m"}},"pricing":{"inputCostPerMillionTokens":"0.25","outputCostPerMillionTokens":"1.5","cacheReadCostPerMillionTokens":null,"cacheWriteCostPerMillionTokens":null}}`),
		map[string]string{managementapi.HeaderIfMatch: `"mdl:1"`})
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if service.modelPatch.Backends != nil || service.modelPatch.Execution == nil ||
		service.modelPatch.Execution.MaxRetries != 4 || service.modelPatch.Pricing == nil {
		t.Fatalf("sparse Model patch = %#v", service.modelPatch)
	}
	request := authorizer.requests[len(authorizer.requests)-1]
	if request.Conditions["provider_credential_referenced"] || len(request.Targets["credential"]) != 0 {
		t.Fatalf("metadata-only patch requested credential authority: %#v", request)
	}
}

func TestRoutingModelPatchAuthorizesExplicitBackendCredentials(t *testing.T) {
	service := &routingServiceStub{}
	authorizer := &routingAuthorizerStub{}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, authorizer)
	credentialID := "44444444-4444-4444-8444-444444444444"

	response := serveRoutingRequest(t, routes, http.MethodPatch, routingModelsPath+"/model_one",
		strings.NewReader(`{"backends":[{"providerId":"private","providerModelId":"private/model","credentialId":"`+credentialID+`","baseUrl":"https://models.example.com/v1"}]}`),
		map[string]string{managementapi.HeaderIfMatch: `"mdl:1"`})
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	request := authorizer.requests[len(authorizer.requests)-1]
	if !request.Conditions["provider_credential_referenced"] || len(request.Targets["credential"]) != 1 ||
		request.Targets["credential"][0].Scope.ResourceID != accesscontrol.ResourceID(credentialID) {
		t.Fatalf("backend patch credential authority = %#v", request)
	}
}

func TestRoutingResolveAuthorizesOnlySelectedClosure(t *testing.T) {
	entrypoint := testRoutingEntrypoint("entrypoint_one")
	selected := entrypoint.Current.Rules[0]
	recipe := routingsnapshot.Recipe{ID: "recipe_one", Revision: 3, Name: "Recipe"}
	service := &routingServiceStub{resolution: routingsnapshot.Resolution{
		Outcome: routingsnapshot.ResolveMatched, Entrypoint: &entrypoint.Current, Rule: &selected, Recipe: &recipe,
	}}
	authorizer := &routingAuthorizerStub{}
	routes := newTestRoutingRoutes(t, service, &routingCommandResultsStub{}, authorizer)
	response := serveRoutingRequest(t, routes, http.MethodPost, routingEntrypointsPath+"/entrypoint_one:resolve",
		strings.NewReader(`{"path":"/chat"}`), nil)
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if len(authorizer.requests) != 2 {
		t.Fatalf("authorization calls = %d", len(authorizer.requests))
	}
	final := authorizer.requests[1]
	if !final.Conditions["entrypoint_resolution_matched"] || len(final.Targets["all_dependencies"]) != 2 {
		t.Fatalf("resolve authorization = %#v", final)
	}
	if strings.Contains(response.Body.String(), "unselected") {
		t.Fatalf("resolve exposed unrelated topology: %s", response.Body.String())
	}
}

func newTestRoutingRoutes(
	t *testing.T,
	service RoutingManagementService,
	commands RoutingCommandResults,
	authorizer Authorizer,
) *RoutingRoutes {
	return newTestRoutingRoutesWithScopes(t, service, commands, authorizer, allowAllResultScopes())
}

func newTestRoutingRoutesWithScopes(
	t *testing.T,
	service RoutingManagementService,
	commands RoutingCommandResults,
	authorizer Authorizer,
	scopes ResultScopeResolver,
) *RoutingRoutes {
	t.Helper()
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": bytes.Repeat([]byte{0x42}, 32)},
	})
	if err != nil {
		t.Fatal(err)
	}
	routes, err := NewRoutingRoutes(RoutingRoutesOptions{
		Service: service, Commands: codec, CommandResults: commands,
		Namespaces: ExplicitNamespaceResolver{}, Sessions: sessionStub{}, Authorization: authorizer,
		Scopes:         scopes,
		IdempotencyTTL: time.Hour, Now: func() time.Time { return time.Unix(1000, 0) },
	})
	if err != nil {
		t.Fatal(err)
	}
	return routes
}

func serveRoutingRequest(
	t *testing.T,
	routes *RoutingRoutes,
	method, target string,
	body *strings.Reader,
	headers map[string]string,
) *httptest.ResponseRecorder {
	t.Helper()
	var request *http.Request
	if body == nil {
		request = httptest.NewRequest(method, target, nil)
	} else {
		request = httptest.NewRequest(method, target, body)
		request.Header.Set("Content-Type", managementapi.JSONMediaType)
	}
	request.Header.Set("Authorization", "Bearer management-token")
	request.Header.Set(managementapi.HeaderNamespaceID, testNamespaceID)
	request.Header.Set(managementapi.HeaderRequestID, "routing-request")
	for name, value := range headers {
		request.Header.Set(name, value)
	}
	response := httptest.NewRecorder()
	mux := http.NewServeMux()
	routes.Register(mux)
	mux.ServeHTTP(response, request)
	return response
}

func testRoutingEntrypoint(id string) routingmanagement.Entrypoint {
	return routingmanagement.Entrypoint{
		ResourceIdentity: routingmanagement.ResourceIdentity{
			ID: id, Name: id, Status: routingmanagement.StatusActive, Revision: 1,
			CreatedAt: time.Unix(1, 0), UpdatedAt: time.Unix(1, 0),
		},
		Current: routingsnapshot.Entrypoint{
			ID: id, Revision: 2, Name: id, Aliases: []string{id},
			Rules: []routingsnapshot.EntrypointRule{{
				ID: "rule_one", Name: "Rule", RecipeID: "recipe_one", RecipeRevision: 3,
				Assignments: map[string]routingsnapshot.AssignmentSet{
					"decision_one": {Models: []routingsnapshot.Assignment{{ModelID: "model_one", ModelRevision: 4, Weight: "1"}}},
				},
			}},
		},
	}
}
