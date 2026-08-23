package agentworkflow

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	workflowNamespaceID  = "11111111-1111-4111-8111-111111111111"
	workflowPrincipalID  = "22222222-2222-4222-8222-222222222222"
	workflowSessionID    = "33333333-3333-4333-8333-333333333333"
	workflowTurnID       = "44444444-4444-4444-8444-444444444444"
	workflowInvocationID = "55555555-5555-4555-8555-555555555555"
)

var workflowTestNow = time.Date(2026, 8, 23, 9, 0, 0, 0, time.UTC)

func TestProviderRegistersExactWorkflowToolSet(t *testing.T) {
	provider := newWorkflowTestProvider(t, &workflowStoreStub{}, &workflowRoutingStub{})
	registrations, err := provider.Current(context.Background(), workflowNamespaceID)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]string, len(registrations))
	for index, registration := range registrations {
		got[index] = registration.Definition.Name
		if registration.Handler == nil || registration.Origin.Kind != agentmanagement.ToolOriginRouter ||
			!json.Valid(registration.Definition.InputSchema) ||
			!json.Valid(registration.Definition.OutputSchema) {
			t.Fatalf("invalid workflow registration %#v", registration)
		}
	}
	want := []string{
		agentmanagement.ToolEntrypointPrepare,
		agentmanagement.ToolPublishPrepare,
		agentmanagement.ToolRecipeEvaluate,
		agentmanagement.ToolRecipePrepare,
		agentmanagement.ToolRecipeProbe,
	}
	sort.Strings(want)
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("workflow tools = %v, want %v", got, want)
	}
}

func TestRecipePrepareCreatesDraftWithoutExistingRecipe(t *testing.T) {
	routing := &workflowRoutingStub{}
	provider := newWorkflowTestProvider(t, &workflowStoreStub{}, routing)
	registration := provider.byName[agentmanagement.ToolRecipePrepare]
	result, err := registration.Handler.Invoke(
		context.Background(), workflowInvocation(),
		json.RawMessage(`{"expectedRevision":0,"name":"Adaptive","document":{"decisions":[]}}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	if routing.createdRecipe.Name != "Adaptive" || string(routing.createdRecipe.Document) != `{"decisions":[]}` {
		t.Fatalf("created Recipe input = %#v", routing.createdRecipe)
	}
	var output recipeMutationOutput
	if err := json.Unmarshal(result.Value, &output); err != nil {
		t.Fatal(err)
	}
	if output.RecipeID != "rcp_adaptive" || output.ResourceRevision != 1 ||
		output.ContentRevision != 1 || output.OperationID == "" {
		t.Fatalf("Recipe prepare output = %#v", output)
	}
}

func newWorkflowTestProvider(
	t *testing.T, store Store, routing RoutingService,
) *Provider {
	t.Helper()
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1", Keys: map[string][]byte{"v1": make([]byte, 32)},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = codec.Close() })
	provider, err := New(Options{
		Store: store, Routing: routing,
		Authorization: managementauthorization.Runtime{Loader: workflowAuthorityLoader{}},
		Commands:      codec, Now: func() time.Time { return workflowTestNow },
	})
	if err != nil {
		t.Fatal(err)
	}
	return provider
}

func workflowInvocation() agentmanagement.ToolInvocationContext {
	return agentmanagement.ToolInvocationContext{
		NamespaceID: workflowNamespaceID, PrincipalID: workflowPrincipalID,
		SessionID: workflowSessionID, TurnID: workflowTurnID,
		InvocationID:     workflowInvocationID,
		AuthorityDigest:  "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		RegistryRevision: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		Target:           agentmanagement.Target{Kind: agentmanagement.TargetEntrypoint, ID: "entrypoint"},
	}
}

type workflowAuthorityLoader struct{}

func (workflowAuthorityLoader) Load(
	_ context.Context,
	principalID accesscontrol.ManagementPrincipalID,
	namespaceID accesscontrol.NamespaceID,
) (managementauthorization.Snapshot, error) {
	role, _ := accesscontrol.BuiltInRole(accesscontrol.BuiltInRolePlatformAdmin)
	return managementauthorization.Snapshot{
		Principal: accesscontrol.ManagementPrincipal{ID: principalID},
		RoleGrants: []managementauthorization.RoleGrant{{
			Binding: accesscontrol.ManagementRoleBinding{
				ID: "workflow-binding", PrincipalID: principalID, RoleID: role.ID,
				Scope:  accesscontrol.NamespaceScope(namespaceID),
				Status: accesscontrol.BindingStatusActive, Revision: 1,
			},
			Role: role,
		}},
		AuthorityDigest: "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
	}, nil
}

type workflowStoreStub struct{}

func (*workflowStoreStub) GetSession(context.Context, string, string) (agentmanagement.Session, error) {
	return agentmanagement.Session{}, agentmanagement.ErrNotFound
}

func (*workflowStoreStub) PutArtifact(
	context.Context, string, agentmanagement.Artifact, json.RawMessage,
) (agentmanagement.Artifact, error) {
	return agentmanagement.Artifact{}, errors.New("unexpected PutArtifact")
}

func (*workflowStoreStub) GetArtifact(context.Context, string, string) (agentmanagement.Artifact, error) {
	return agentmanagement.Artifact{}, agentmanagement.ErrNotFound
}

func (*workflowStoreStub) CreatePublicationPlan(
	context.Context, string, agentmanagement.PublicationPlan, agentmanagement.MutationContext,
) (agentmanagement.PublicationPlan, error) {
	return agentmanagement.PublicationPlan{}, errors.New("unexpected CreatePublicationPlan")
}

func (*workflowStoreStub) GetPublicationPlan(
	context.Context, string, string,
) (agentmanagement.PublicationPlan, error) {
	return agentmanagement.PublicationPlan{}, agentmanagement.ErrNotFound
}

type workflowRoutingStub struct {
	createdRecipe routingmanagement.RecipeInput
}

func (*workflowRoutingStub) GetModel(
	context.Context, string, string,
) (routingmanagement.Model, error) {
	return routingmanagement.Model{}, routingmanagement.ErrNotFound
}

func (*workflowRoutingStub) ProbeModel(
	context.Context, string, string, time.Duration,
) (routingmanagement.ProbeResult, error) {
	return routingmanagement.ProbeResult{}, routingmanagement.ErrProbeUnavailable
}

func (*workflowRoutingStub) GetRecipe(
	context.Context, string, string,
) (routingmanagement.Recipe, error) {
	return routingmanagement.Recipe{}, routingmanagement.ErrNotFound
}

func (stub *workflowRoutingStub) CreateRecipe(
	_ context.Context, _ string, input routingmanagement.RecipeInput,
	_ routingmanagement.MutationContext,
) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error) {
	stub.createdRecipe = input
	return routingmanagement.Recipe{
			ResourceIdentity: routingmanagement.ResourceIdentity{
				ID: "rcp_adaptive", Name: input.Name, Status: routingmanagement.StatusDraft, Revision: 1,
			},
			Current: routingsnapshot.Recipe{
				ID: "rcp_adaptive", Name: input.Name, Revision: 1, Document: input.Document,
			},
		}, routingmanagement.RevisionReceipt{
			ResourceRevision: 1, OperationID: "66666666-6666-4666-8666-666666666666",
		}, nil
}

func (*workflowRoutingStub) UpdateRecipe(
	context.Context, string, string, int64, routingmanagement.RecipeInput,
	routingmanagement.MutationContext,
) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error) {
	return routingmanagement.Recipe{}, routingmanagement.RevisionReceipt{}, errors.New("unexpected UpdateRecipe")
}

func (*workflowRoutingStub) GetEntrypoint(
	context.Context, string, string,
) (routingmanagement.Entrypoint, error) {
	return routingmanagement.Entrypoint{}, routingmanagement.ErrNotFound
}

func (*workflowRoutingStub) CreateEntrypoint(
	context.Context, string, routingmanagement.EntrypointInput, routingmanagement.MutationContext,
) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error) {
	return routingmanagement.Entrypoint{}, routingmanagement.RevisionReceipt{}, errors.New("unexpected CreateEntrypoint")
}

func (*workflowRoutingStub) UpdateEntrypoint(
	context.Context, string, string, int64, routingmanagement.EntrypointInput,
	routingmanagement.MutationContext,
) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error) {
	return routingmanagement.Entrypoint{}, routingmanagement.RevisionReceipt{}, errors.New("unexpected UpdateEntrypoint")
}
