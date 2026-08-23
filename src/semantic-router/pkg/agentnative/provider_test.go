package agentnative

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

var _ agentruntime.NativeToolProvider = (*Provider)(nil)

const (
	testNamespaceID = "11111111-1111-4111-8111-111111111111"
	testPrincipalID = "22222222-2222-4222-8222-222222222222"
	testSessionID   = "33333333-3333-4333-8333-333333333333"
	testProfileID   = "44444444-4444-4444-8444-444444444444"
	testSkillID     = "55555555-5555-4555-8555-555555555555"
)

func TestProviderDefinitionsMatchNativeProviderContract(t *testing.T) {
	provider := newTestProvider(t, &agentStoreStub{}, &routingReaderStub{}, allowAllScopeStub{})
	registrations, err := provider.Current(context.Background(), testNamespaceID)
	if err != nil {
		t.Fatal(err)
	}
	wantNames := []string{
		toolCatalogDescribe, toolModelsList, toolRecipeGet, toolRecipeValidate,
		toolRecipesExamples, toolSkillsRead,
	}
	gotNames := make([]string, len(registrations))
	for index, registration := range registrations {
		gotNames[index] = registration.Definition.Name
		if registration.Handler == nil || registration.Origin.Kind != agentmanagement.ToolOriginRouter ||
			!json.Valid(registration.Definition.InputSchema) || !json.Valid(registration.Definition.OutputSchema) {
			t.Fatalf("invalid native registration %#v", registration)
		}
		if _, err := provider.Resolve(context.Background(), testNamespaceID, registration.Definition); err != nil {
			t.Fatalf("Resolve(%s): %v", registration.Definition.Name, err)
		}
	}
	if !reflect.DeepEqual(gotNames, wantNames) {
		t.Fatalf("tool order = %v, want %v", gotNames, wantNames)
	}
	tampered := registrations[0].Definition
	tampered.Description += " changed"
	if _, err := provider.Resolve(context.Background(), testNamespaceID, tampered); !errors.Is(err, agentmanagement.ErrConflict) {
		t.Fatalf("Resolve(tampered) error = %v, want ErrConflict", err)
	}
}

func TestModelsListProjectsOnlySemanticModelCard(t *testing.T) {
	routing := &routingReaderStub{models: routingmanagement.Page[routingmanagement.Model]{
		Items: []routingmanagement.Model{{
			ResourceIdentity: routingmanagement.ResourceIdentity{
				ID: "model_one", Name: "Model One", Status: routingmanagement.StatusActive, Revision: 3,
			},
			Current: routingsnapshot.Model{
				ID: "compiled-model-secret", Revision: 9, CatalogRevision: "catalog-secret",
				Name: "Model One", Aliases: []string{"model-one"}, Capabilities: []string{"tools"},
				Backends: []routingsnapshot.Backend{{
					ID: "backend-secret", ProviderID: "provider-secret",
					ProviderModelID: "physical-model-secret", ProviderCredentialID: "credential-secret",
					Connection: routingsnapshot.BackendConnection{
						Path: "/private-path", Headers: map[string]string{"authorization": "secret-value"},
					},
				}},
			},
		}},
	}}
	provider := newTestProvider(t, &agentStoreStub{}, routing, allowAllScopeStub{})
	result, err := provider.byName[toolModelsList].Handler.Invoke(
		context.Background(), testInvocation(), json.RawMessage(`{}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(routing.modelRequests) != 1 || routing.modelRequests[0].PageSize != defaultModelPageSize {
		t.Fatalf("model request = %#v, want page size %d", routing.modelRequests, defaultModelPageSize)
	}
	text := string(result.Value)
	for _, forbidden := range []string{
		"compiled-model-secret", "catalog-secret", "backend-secret", "provider-secret",
		"physical-model-secret", "credential-secret", "/private-path", "secret-value",
		"catalogRevision", "backends", "providerCredentialId", "connection",
	} {
		if strings.Contains(text, forbidden) {
			t.Errorf("Models result exposed %q: %s", forbidden, text)
		}
	}
	var output modelPageOutput
	if err := json.Unmarshal(result.Value, &output); err != nil {
		t.Fatal(err)
	}
	if len(output.Data) != 1 || output.Data[0].ID != "model_one" || output.Data[0].Card.Capabilities[0] != "tools" {
		t.Fatalf("Models result = %#v", output)
	}
}

func TestProviderRejectsCrossNamespaceResultScope(t *testing.T) {
	routing := &routingReaderStub{}
	provider := newTestProvider(t, &agentStoreStub{}, routing, fixedScopeStub{scope: accesscontrol.ResultScope{
		NamespaceID: "99999999-9999-4999-8999-999999999999", All: true,
	}})
	_, err := provider.byName[toolModelsList].Handler.Invoke(
		context.Background(), testInvocation(), json.RawMessage(`{}`),
	)
	if !errors.Is(err, agentmanagement.ErrDenied) || len(routing.modelRequests) != 0 {
		t.Fatalf("cross-namespace result = %v, requests = %d", err, len(routing.modelRequests))
	}
}

func TestSkillsReadUsesOnlyPinnedRevisionAndEncodesEmptyLists(t *testing.T) {
	invocation := testInvocation()
	store := &agentStoreStub{
		session: agentmanagement.Session{
			ID: testSessionID, NamespaceID: testNamespaceID, OwnerPrincipalID: testPrincipalID,
			ProfileID: testProfileID, ProfileRevision: 4, Target: invocation.Target,
			Status: agentmanagement.SessionActive,
		},
		profile: agentmanagement.Profile{Skills: []agentmanagement.SkillReference{{ID: testSkillID, Revision: 7}}},
		skill: agentmanagement.Skill{
			ResourceIdentity: agentmanagement.ResourceIdentity{ID: testSkillID, Name: "Builder"},
			ContentRevision:  7, ContentDigest: "sha256:fixture", Instructions: "Use Router tools.",
		},
	}
	provider := newTestProvider(t, store, &routingReaderStub{}, allowAllScopeStub{})
	result, err := provider.byName[toolSkillsRead].Handler.Invoke(
		context.Background(), invocation, json.RawMessage(`{"skillId":"`+testSkillID+`"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	if store.requestedSkillRevision != 7 {
		t.Fatalf("requested Skill revision = %d, want 7", store.requestedSkillRevision)
	}
	var output map[string]json.RawMessage
	if err := json.Unmarshal(result.Value, &output); err != nil {
		t.Fatal(err)
	}
	if string(output["requiredTools"]) != "[]" || string(output["minimumCapabilities"]) != "[]" {
		t.Fatalf("Skill arrays are not concrete empty arrays: %s", result.Value)
	}
}

func TestRecipeGetHidesCompiledDecisionIdentity(t *testing.T) {
	routing := &routingReaderStub{recipe: routingmanagement.Recipe{
		ResourceIdentity: routingmanagement.ResourceIdentity{
			ID: "recipe_one", Name: "Recipe One", Status: routingmanagement.StatusActive, Revision: 2,
		},
		Current: routingsnapshot.Recipe{
			ID: "recipe_one", Revision: 5, Name: "Recipe One",
			Decisions: []routingsnapshot.Decision{{
				ID: "compiled-decision-secret", Name: "Simple",
				DispatchCardinality: routingsnapshot.DispatchCardinalitySingle,
			}},
			Document: json.RawMessage(`{"decisions":[{"name":"Simple","rules":{}}]}`),
		},
	}}
	scope := accesscontrol.ResultScope{
		NamespaceID: testNamespaceID,
		ResourceIDs: map[accesscontrol.ScopeResourceType][]accesscontrol.ResourceID{
			accesscontrol.ScopeResourceRecipe: {"recipe_one"},
		},
	}
	provider := newTestProvider(t, &agentStoreStub{}, routing, fixedScopeStub{scope: scope})
	result, err := provider.byName[toolRecipeGet].Handler.Invoke(
		context.Background(), testInvocation(), json.RawMessage(`{"recipeId":"recipe_one"}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(result.Value), "compiled-decision-secret") || strings.Contains(string(result.Value), `"id":"compiled`) {
		t.Fatalf("Recipe result exposed compiled identity: %s", result.Value)
	}
}

type agentStoreStub struct {
	session                agentmanagement.Session
	profile                agentmanagement.Profile
	skill                  agentmanagement.Skill
	requestedSkillRevision int64
}

func (store *agentStoreStub) GetSession(context.Context, string, string) (agentmanagement.Session, error) {
	return store.session, nil
}

func (store *agentStoreStub) GetProfileRevision(context.Context, string, string, int64) (agentmanagement.Profile, error) {
	return store.profile, nil
}

func (store *agentStoreStub) GetSkillRevision(
	_ context.Context, _, _ string, revision int64,
) (agentmanagement.Skill, error) {
	store.requestedSkillRevision = revision
	return store.skill, nil
}

type routingReaderStub struct {
	models        routingmanagement.Page[routingmanagement.Model]
	recipe        routingmanagement.Recipe
	modelRequests []routingmanagement.PageRequest
}

func (reader *routingReaderStub) ListModels(
	_ context.Context, _ string, request routingmanagement.PageRequest,
) (routingmanagement.Page[routingmanagement.Model], error) {
	reader.modelRequests = append(reader.modelRequests, request)
	return reader.models, nil
}

func (reader *routingReaderStub) GetRecipe(context.Context, string, string) (routingmanagement.Recipe, error) {
	return reader.recipe, nil
}

type allowAllScopeStub struct{}

func (allowAllScopeStub) ResolveResultScope(
	_ context.Context, _ accesscontrol.ManagementPrincipalID, namespaceID accesscontrol.NamespaceID,
	_ accesscontrol.Permission,
) (accesscontrol.ResultScope, error) {
	return accesscontrol.ResultScope{NamespaceID: namespaceID, All: true}, nil
}

type fixedScopeStub struct{ scope accesscontrol.ResultScope }

func (stub fixedScopeStub) ResolveResultScope(
	context.Context, accesscontrol.ManagementPrincipalID, accesscontrol.NamespaceID, accesscontrol.Permission,
) (accesscontrol.ResultScope, error) {
	return stub.scope, nil
}

type catalogSourceStub struct{}

func (catalogSourceStub) Describe(CatalogQuery) (CatalogPage, error) { return CatalogPage{}, nil }

type exampleSourceStub struct{}

func (exampleSourceStub) List(ExampleQuery) (ExamplePage, error) { return ExamplePage{}, nil }

func newTestProvider(
	t *testing.T, store AgentStore, routing RoutingReader, scopes ScopeResolver,
) *Provider {
	t.Helper()
	provider, err := New(Options{
		Store: store, Routing: routing, Scopes: scopes,
		Catalog: catalogSourceStub{}, Examples: exampleSourceStub{},
	})
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func testInvocation() agentmanagement.ToolInvocationContext {
	return agentmanagement.ToolInvocationContext{
		NamespaceID: testNamespaceID, PrincipalID: testPrincipalID, SessionID: testSessionID,
		AuthorityDigest: "sha256:authority", Target: agentmanagement.Target{
			Kind: agentmanagement.TargetModel, ID: "model_one",
		},
	}
}
