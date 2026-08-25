// Package agentworkflow exposes the state-changing Builder workflow as five
// Router-native tools. It adapts existing Routing and Agent repositories; it
// does not own publication commit, inference dispatch, or HTTP transport.
package agentworkflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"time"

	jsonschema "github.com/invopop/jsonschema"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

const (
	workflowCommandTTL = 24 * time.Hour
	artifactRetention  = 7 * 24 * time.Hour
)

type Store interface {
	GetSession(context.Context, string, string) (agentmanagement.Session, error)
	PutArtifact(context.Context, string, agentmanagement.Artifact, json.RawMessage) (agentmanagement.Artifact, error)
	GetArtifact(context.Context, string, string) (agentmanagement.Artifact, error)
	CreatePublicationPlan(
		context.Context, string, agentmanagement.PublicationPlan, agentmanagement.MutationContext,
	) (agentmanagement.PublicationPlan, error)
	GetPublicationPlan(context.Context, string, string) (agentmanagement.PublicationPlan, error)
}

type RoutingModelService interface {
	GetModel(context.Context, string, string) (routingmanagement.Model, error)
	ProbeModel(context.Context, string, string, time.Duration) (routingmanagement.ProbeResult, error)
}

type RoutingRecipeService interface {
	GetRecipe(context.Context, string, string) (routingmanagement.Recipe, error)
	CreateRecipe(
		context.Context, string, routingmanagement.RecipeInput,
		routingmanagement.MutationContext,
	) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error)
	UpdateRecipe(
		context.Context, string, string, int64, routingmanagement.RecipeInput,
		routingmanagement.MutationContext,
	) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error)
}

type RoutingEntrypointService interface {
	GetEntrypoint(context.Context, string, string) (routingmanagement.Entrypoint, error)
	CreateEntrypoint(
		context.Context, string, routingmanagement.EntrypointInput, routingmanagement.MutationContext,
	) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error)
	UpdateEntrypoint(
		context.Context, string, string, int64, routingmanagement.EntrypointInput,
		routingmanagement.MutationContext,
	) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error)
}

type RoutingService interface {
	RoutingModelService
	RoutingRecipeService
	RoutingEntrypointService
}

type Options struct {
	Store         Store
	Routing       RoutingService
	Authorization managementauthorization.Runtime
	Commands      *managementcommand.Codec
	Now           func() time.Time
}

type Provider struct {
	store         Store
	routing       RoutingService
	authorization managementauthorization.Runtime
	commands      *managementcommand.Codec
	now           func() time.Time
	ordered       []agentmanagement.RegisteredTool
	byName        map[string]agentmanagement.RegisteredTool
}

func New(options Options) (*Provider, error) {
	if options.Store == nil || options.Routing == nil || options.Authorization.Loader == nil ||
		options.Commands == nil {
		return nil, errors.New("agent Builder workflow dependencies are incomplete")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	provider := &Provider{
		store: options.Store, routing: options.Routing, authorization: options.Authorization,
		commands: options.Commands, now: now, byName: make(map[string]agentmanagement.RegisteredTool),
	}
	registrations, err := provider.registrations()
	if err != nil {
		return nil, err
	}
	for _, registration := range registrations {
		canonical, err := agentmanagement.CanonicalizeToolDefinition(registration.Definition)
		if err != nil {
			return nil, fmt.Errorf("register Agent workflow tool %q: %w", registration.Definition.Name, err)
		}
		registration.Definition = canonical
		if _, duplicate := provider.byName[canonical.Name]; duplicate {
			return nil, fmt.Errorf("duplicate Agent workflow tool %q", canonical.Name)
		}
		provider.byName[canonical.Name] = registration
		provider.ordered = append(provider.ordered, registration)
	}
	sort.Slice(provider.ordered, func(left, right int) bool {
		return provider.ordered[left].Definition.Name < provider.ordered[right].Definition.Name
	})
	return provider, nil
}

func (provider *Provider) Current(
	_ context.Context, namespaceID string,
) ([]agentmanagement.RegisteredTool, error) {
	if provider == nil || namespaceID == "" {
		return nil, agentmanagement.ErrInvalid
	}
	result := make([]agentmanagement.RegisteredTool, len(provider.ordered))
	for index, registration := range provider.ordered {
		result[index] = cloneRegistration(registration)
	}
	return result, nil
}

func (provider *Provider) Resolve(
	_ context.Context, namespaceID string, requested agentmanagement.ToolDefinition,
) (agentmanagement.ToolHandler, error) {
	if provider == nil || namespaceID == "" {
		return nil, agentmanagement.ErrInvalid
	}
	registration, found := provider.byName[requested.Name]
	if !found {
		return nil, agentmanagement.ErrToolUnavailable
	}
	canonical, err := agentmanagement.CanonicalizeToolDefinition(requested)
	if err != nil || !reflect.DeepEqual(canonical, registration.Definition) {
		return nil, agentmanagement.ErrConflict
	}
	return registration.Handler, nil
}

func cloneRegistration(source agentmanagement.RegisteredTool) agentmanagement.RegisteredTool {
	result := source
	result.Definition.InputSchema = append(json.RawMessage(nil), source.Definition.InputSchema...)
	result.Definition.OutputSchema = append(json.RawMessage(nil), source.Definition.OutputSchema...)
	result.Definition.RequiredPermissions = append(
		[]accesscontrol.Permission(nil), source.Definition.RequiredPermissions...,
	)
	return result
}

type toolRegistration struct {
	name        string
	description string
	input       any
	output      any
	permissions []accesscontrol.Permission
	class       agentmanagement.ToolClass
	timeout     time.Duration
	handler     agentmanagement.ToolHandlerFunc
}

func (provider *Provider) registrations() ([]agentmanagement.RegisteredTool, error) {
	definitions := []toolRegistration{
		{
			name:        agentmanagement.ToolRecipePrepare,
			description: "Create or update one Recipe draft at an exact revision.",
			input:       recipePrepareSchema{}, output: recipeMutationOutput{},
			permissions: []accesscontrol.Permission{accesscontrol.PermissionRoutingManage},
			class:       agentmanagement.ToolWrite, timeout: 30 * time.Second, handler: provider.prepareRecipe,
		},
		{
			name:        agentmanagement.ToolRecipeProbe,
			description: "Probe every Model assigned to one Recipe path.",
			input:       recipeProbeInput{}, output: recipeProbeOutput{},
			permissions: []accesscontrol.Permission{
				accesscontrol.PermissionRoutingRead, accesscontrol.PermissionEvaluationRun,
			},
			class: agentmanagement.ToolExecute, timeout: 5 * time.Minute, handler: provider.probeRecipe,
		},
		{
			name:        agentmanagement.ToolRecipeEvaluate,
			description: "Evaluate assignment coverage and readiness gates for one Recipe path.",
			input:       recipeEvaluationInput{}, output: recipeEvaluationOutput{},
			permissions: []accesscontrol.Permission{
				accesscontrol.PermissionRoutingRead, accesscontrol.PermissionEvaluationRun,
			},
			class: agentmanagement.ToolExecute, timeout: 2 * time.Minute, handler: provider.evaluateRecipe,
		},
		{
			name:        agentmanagement.ToolEntrypointPrepare,
			description: "Create or update one Entrypoint draft with explicit Model assignments.",
			input:       entrypointPrepareInput{}, output: entrypointPrepareOutput{},
			permissions: []accesscontrol.Permission{accesscontrol.PermissionRoutingManage},
			class:       agentmanagement.ToolWrite, timeout: 2 * time.Minute, handler: provider.prepareEntrypoint,
		},
		{
			name:        agentmanagement.ToolPublishPrepare,
			description: "Prepare the exact immutable publication review for human approval.",
			input:       publishPrepareInput{}, output: publishPrepareOutput{},
			permissions: []accesscontrol.Permission{
				accesscontrol.PermissionRoutingRead, accesscontrol.PermissionRoutingManage,
			},
			class: agentmanagement.ToolWrite, timeout: 30 * time.Second, handler: provider.preparePublication,
		},
	}
	result := make([]agentmanagement.RegisteredTool, 0, len(definitions))
	for _, item := range definitions {
		input, err := reflectToolSchema(item.input)
		if err != nil {
			return nil, fmt.Errorf("reflect %s input: %w", item.name, err)
		}
		output, err := reflectToolSchema(item.output)
		if err != nil {
			return nil, fmt.Errorf("reflect %s output: %w", item.name, err)
		}
		result = append(result, agentmanagement.RegisteredTool{
			Definition: agentmanagement.ToolDefinition{
				Name: item.name, Description: item.description,
				InputSchema: input, OutputSchema: output,
				RequiredPermissions: item.permissions, Class: item.class,
				Idempotency:         agentmanagement.ToolInvocationIdempotent,
				TimeoutMilliseconds: item.timeout.Milliseconds(),
			},
			Handler: item.handler,
			Origin:  agentmanagement.ToolOrigin{Kind: agentmanagement.ToolOriginRouter},
		})
	}
	return result, nil
}

func reflectToolSchema(sample any) (json.RawMessage, error) {
	reflector := jsonschema.Reflector{
		Anonymous: true, DoNotReference: true, FieldNameTag: "json",
		AllowAdditionalProperties: false,
	}
	encoded, err := json.Marshal(reflector.Reflect(sample))
	if err != nil {
		return nil, err
	}
	var root map[string]any
	if err := json.Unmarshal(encoded, &root); err != nil {
		return nil, err
	}
	delete(root, "$id")
	return json.Marshal(root)
}

var _ interface {
	Current(context.Context, string) ([]agentmanagement.RegisteredTool, error)
	Resolve(context.Context, string, agentmanagement.ToolDefinition) (agentmanagement.ToolHandler, error)
} = (*Provider)(nil)
