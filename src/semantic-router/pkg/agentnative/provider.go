package agentnative

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"

	"github.com/google/uuid"
	jsonschema "github.com/invopop/jsonschema"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const (
	defaultModelPageSize     = 25
	maximumModelPageSize     = 25
	maximumInlineResultBytes = 64 << 10
)

type Options struct {
	Store    AgentStore
	Routing  RoutingReader
	Scopes   ScopeResolver
	Catalog  CatalogSource
	Examples ExampleSource
}

// Provider is the immutable Router-native tool adapter. Go's structural
// interfaces let it satisfy agentruntime.NativeToolProvider without importing
// the runtime package or creating a domain cycle.
type Provider struct {
	store    AgentStore
	routing  RoutingReader
	scopes   ScopeResolver
	catalog  CatalogSource
	examples ExampleSource
	ordered  []agentmanagement.RegisteredTool
	byName   map[string]agentmanagement.RegisteredTool
}

func New(options Options) (*Provider, error) {
	if options.Store == nil || options.Routing == nil || options.Scopes == nil ||
		options.Catalog == nil || options.Examples == nil {
		return nil, errors.New("router-native Agent tool dependencies are incomplete")
	}
	provider := &Provider{
		store: options.Store, routing: options.Routing, scopes: options.Scopes,
		catalog: options.Catalog, examples: options.Examples,
		byName: make(map[string]agentmanagement.RegisteredTool),
	}
	registrations, err := provider.registrations()
	if err != nil {
		return nil, err
	}
	for _, registration := range registrations {
		canonical, err := agentmanagement.CanonicalizeToolDefinition(registration.Definition)
		if err != nil {
			return nil, fmt.Errorf("register Router-native tool %q: %w", registration.Definition.Name, err)
		}
		registration.Definition = canonical
		if _, duplicate := provider.byName[canonical.Name]; duplicate {
			return nil, fmt.Errorf("duplicate Router-native tool %q", canonical.Name)
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
	if provider == nil || uuid.Validate(namespaceID) != nil {
		return nil, agentmanagement.ErrInvalid
	}
	result := make([]agentmanagement.RegisteredTool, len(provider.ordered))
	for index := range provider.ordered {
		result[index] = cloneRegistration(provider.ordered[index])
	}
	return result, nil
}

func (provider *Provider) Resolve(
	_ context.Context, namespaceID string, requested agentmanagement.ToolDefinition,
) (agentmanagement.ToolHandler, error) {
	if provider == nil || uuid.Validate(namespaceID) != nil {
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

func (provider *Provider) registrations() ([]agentmanagement.RegisteredTool, error) {
	definitions := []struct {
		name        string
		description string
		input       any
		output      any
		permissions []accesscontrol.Permission
		handler     agentmanagement.ToolHandlerFunc
	}{
		{
			name: toolCatalogDescribe, description: "Read current Router component schemas.",
			input: catalogDescribeInput{}, output: catalogOutputSchema{},
			permissions: []accesscontrol.Permission{accesscontrol.PermissionRoutingRead, accesscontrol.PermissionToolRead},
			handler:     provider.describeCatalog,
		},
		{
			name: toolSkillsRead, description: "Read one Skill revision pinned by this session.",
			input: skillReadInput{}, output: skillReadOutput{},
			permissions: []accesscontrol.Permission{accesscontrol.PermissionAgentRead, accesscontrol.PermissionToolRead},
			handler:     provider.readSkill,
		},
		{
			name: toolModelsList, description: "List authorized connected ModelCards.",
			input: modelsListInput{}, output: modelPageOutput{},
			permissions: []accesscontrol.Permission{accesscontrol.PermissionRoutingRead},
			handler:     provider.listModels,
		},
		{
			name: toolRecipesExamples, description: "Read model-free built-in Recipe examples.",
			input: examplesListInput{}, output: examplesOutputSchema{},
			permissions: []accesscontrol.Permission{accesscontrol.PermissionRoutingRead},
			handler:     provider.listExamples,
		},
		{
			name: toolRecipeGet, description: "Read one authorized Recipe draft and revision.",
			input: recipeGetInput{}, output: recipeGetOutputSchema{},
			permissions: []accesscontrol.Permission{accesscontrol.PermissionRoutingRead},
			handler:     provider.getRecipe,
		},
		{
			name: toolRecipeValidate, description: "Compile a Recipe draft without publishing it.",
			input: recipeValidateInputSchema{}, output: recipeValidateOutputSchema{},
			permissions: []accesscontrol.Permission{accesscontrol.PermissionRoutingRead},
			handler:     provider.validateRecipe,
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
				RequiredPermissions: item.permissions,
				Class:               agentmanagement.ToolRead,
				Idempotency:         agentmanagement.ToolInvocationIdempotent,
				TimeoutMilliseconds: 5000,
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

type catalogDescribeInput struct {
	Kind     ComponentKind `json:"kind,omitempty" jsonschema:"enum=signal,enum=projection,enum=decision,enum=algorithm,enum=plugin"`
	Name     string        `json:"name,omitempty"`
	PageSize int           `json:"pageSize,omitempty" jsonschema:"minimum=1,maximum=20"`
	Cursor   string        `json:"cursor,omitempty"`
}

type catalogDescriptorSchema struct {
	Kind        ComponentKind `json:"kind"`
	Name        string        `json:"name"`
	Description string        `json:"description"`
	Tier        string        `json:"tier,omitempty"`
	Execution   string        `json:"execution,omitempty"`
	Schema      any           `json:"schema,omitempty"`
}

type catalogOutputSchema struct {
	Revision   string                    `json:"revision"`
	Data       []catalogDescriptorSchema `json:"data"`
	NextCursor string                    `json:"nextCursor,omitempty"`
	HasMore    bool                      `json:"hasMore"`
	PageSize   int                       `json:"pageSize"`
}

type skillReadInput struct {
	SkillID string `json:"skillId"`
}

type skillReadOutput struct {
	ID                  string   `json:"id"`
	Name                string   `json:"name"`
	Description         string   `json:"description,omitempty"`
	Revision            int64    `json:"revision"`
	ContentRevision     int64    `json:"contentRevision"`
	ContentDigest       string   `json:"contentDigest"`
	Instructions        string   `json:"instructions"`
	RequiredTools       []string `json:"requiredTools"`
	MinimumCapabilities []string `json:"minimumCapabilities"`
}

type modelsListInput struct {
	Search   string `json:"search,omitempty"`
	Status   string `json:"status,omitempty" jsonschema:"enum=draft,enum=active,enum=disabled"`
	PageSize int    `json:"pageSize,omitempty" jsonschema:"minimum=1,maximum=25"`
	Cursor   string `json:"cursor,omitempty"`
}

type modelReasoning struct {
	Type    string   `json:"type,omitempty"`
	Efforts []string `json:"efforts,omitempty"`
}

type modelCard struct {
	Aliases           []string       `json:"aliases"`
	ParamSize         string         `json:"paramSize,omitempty"`
	ContextWindowSize int            `json:"contextWindowSize,omitempty"`
	Description       string         `json:"description,omitempty"`
	Capabilities      []string       `json:"capabilities"`
	Reasoning         modelReasoning `json:"reasoning,omitempty"`
	LoRAs             []string       `json:"loras"`
	QualityScore      float64        `json:"qualityScore,omitempty"`
	Modality          string         `json:"modality,omitempty"`
	Tags              []string       `json:"tags"`
}

type modelCardView struct {
	ID       string    `json:"id"`
	Name     string    `json:"name"`
	Status   string    `json:"status"`
	Revision int64     `json:"revision"`
	Card     modelCard `json:"card"`
}

type modelPageOutput struct {
	Data       []modelCardView `json:"data"`
	NextCursor string          `json:"nextCursor,omitempty"`
	HasMore    bool            `json:"hasMore"`
	PageSize   int             `json:"pageSize"`
}

type examplesListInput struct {
	Search   string `json:"search,omitempty"`
	Name     string `json:"name,omitempty"`
	PageSize int    `json:"pageSize,omitempty" jsonschema:"minimum=1,maximum=10"`
	Cursor   string `json:"cursor,omitempty"`
}

type recipeExampleSchema struct {
	SourceID       string `json:"sourceId"`
	SourceRevision int64  `json:"sourceRevision"`
	Name           string `json:"name"`
	Description    string `json:"description,omitempty"`
	RecipeDigest   string `json:"recipeDigest"`
	Document       any    `json:"document,omitempty"`
}

type examplesOutputSchema struct {
	Revision   string                `json:"revision"`
	Data       []recipeExampleSchema `json:"data"`
	NextCursor string                `json:"nextCursor,omitempty"`
	HasMore    bool                  `json:"hasMore"`
	PageSize   int                   `json:"pageSize"`
}

type recipeGetInput struct {
	RecipeID string `json:"recipeId"`
}

type recipeDecisionView struct {
	Name                string `json:"name"`
	DispatchCardinality string `json:"dispatchCardinality"`
}

type recipeView struct {
	ID             string               `json:"id"`
	Name           string               `json:"name"`
	Description    string               `json:"description,omitempty"`
	Status         string               `json:"status"`
	Revision       int64                `json:"revision"`
	RecipeRevision int64                `json:"recipeRevision"`
	Origin         string               `json:"origin"`
	Immutable      bool                 `json:"immutable"`
	Decisions      []recipeDecisionView `json:"decisions"`
	Document       json.RawMessage      `json:"document"`
}

type recipeViewSchema struct {
	ID             string               `json:"id"`
	Name           string               `json:"name"`
	Description    string               `json:"description,omitempty"`
	Status         string               `json:"status"`
	Revision       int64                `json:"revision"`
	RecipeRevision int64                `json:"recipeRevision"`
	Origin         string               `json:"origin"`
	Immutable      bool                 `json:"immutable"`
	Decisions      []recipeDecisionView `json:"decisions"`
	Document       any                  `json:"document"`
}

type recipeGetOutput struct {
	Data recipeView `json:"data"`
	ETag string     `json:"etag"`
}

type recipeGetOutputSchema struct {
	Data recipeViewSchema `json:"data"`
	ETag string           `json:"etag"`
}

type recipeValidateInput struct {
	RecipeID         string          `json:"recipeId"`
	ExpectedRevision int64           `json:"expectedRevision,omitempty"`
	Document         json.RawMessage `json:"document,omitempty"`
}

type recipeValidateInputSchema struct {
	RecipeID         string `json:"recipeId"`
	ExpectedRevision int64  `json:"expectedRevision,omitempty" jsonschema:"minimum=1"`
	Document         any    `json:"document,omitempty"`
}

type recipeDiagnostic struct {
	Severity string `json:"severity"`
	Code     string `json:"code"`
	Message  string `json:"message"`
}

type recipeTopology struct {
	SignalCounts     map[string]int       `json:"signalCounts"`
	ProjectionCounts map[string]int       `json:"projectionCounts"`
	Decisions        []recipeDecisionView `json:"decisions"`
}

type recipeValidateOutput struct {
	Valid             bool               `json:"valid"`
	ETag              string             `json:"etag"`
	CanonicalDocument json.RawMessage    `json:"canonicalDocument,omitempty"`
	Diagnostics       []recipeDiagnostic `json:"diagnostics"`
	Topology          recipeTopology     `json:"topology"`
}

type recipeValidateOutputSchema struct {
	Valid             bool               `json:"valid"`
	ETag              string             `json:"etag"`
	CanonicalDocument any                `json:"canonicalDocument,omitempty"`
	Diagnostics       []recipeDiagnostic `json:"diagnostics"`
	Topology          recipeTopology     `json:"topology"`
}

func (provider *Provider) describeCatalog(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext, raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input catalogDescribeInput
	if err := decodeInput(raw, &input); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	if _, err := provider.routingScope(ctx, invocation); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	page, err := provider.catalog.Describe(CatalogQuery(input))
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	return encodeResult(page)
}

func (provider *Provider) readSkill(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext, raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input skillReadInput
	if err := decodeInput(raw, &input); err != nil || uuid.Validate(input.SkillID) != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	session, err := provider.store.GetSession(ctx, invocation.NamespaceID, invocation.SessionID)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	if session.Status != agentmanagement.SessionActive || session.OwnerPrincipalID != invocation.PrincipalID ||
		session.Target != invocation.Target {
		return agentmanagement.ToolResult{}, agentmanagement.ErrDenied
	}
	agentScope, err := provider.resolveScope(ctx, invocation, accesscontrol.PermissionAgentRead)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	if !sessionVisible(agentScope, session) {
		return agentmanagement.ToolResult{}, agentmanagement.ErrNotFound
	}
	profile, err := provider.store.GetProfileRevision(
		ctx, invocation.NamespaceID, session.ProfileID, session.ProfileRevision,
	)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	revision := int64(0)
	for _, reference := range profile.Skills {
		if reference.ID == input.SkillID {
			revision = reference.Revision
			break
		}
	}
	if revision == 0 {
		return agentmanagement.ToolResult{}, agentmanagement.ErrNotFound
	}
	skill, err := provider.store.GetSkillRevision(ctx, invocation.NamespaceID, input.SkillID, revision)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	return encodeResult(skillReadOutput{
		ID: skill.ID, Name: skill.Name, Description: skill.Description,
		Revision: revision, ContentRevision: skill.ContentRevision,
		ContentDigest: skill.ContentDigest, Instructions: skill.Instructions,
		RequiredTools:       append([]string{}, skill.RequiredTools...),
		MinimumCapabilities: append([]string{}, skill.MinimumCapabilities...),
	})
}

func (provider *Provider) listModels(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext, raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input modelsListInput
	if err := decodeInput(raw, &input); err != nil || input.PageSize < 0 || input.PageSize > maximumModelPageSize {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	pageSize := input.PageSize
	if pageSize == 0 {
		pageSize = defaultModelPageSize
	}
	scope, err := provider.routingScope(ctx, invocation)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	page, err := provider.routing.ListModels(ctx, invocation.NamespaceID, routingmanagement.PageRequest{
		PageSize: pageSize, Cursor: input.Cursor, Search: input.Search,
		Status: routingmanagement.Status(input.Status), Scope: scope,
	})
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	items := make([]modelCardView, len(page.Items))
	for index, model := range page.Items {
		items[index] = modelCardView{
			ID: model.ID, Name: model.Name, Status: string(model.Status), Revision: model.Revision,
			Card: modelCard{
				Aliases: append([]string{}, model.Current.Aliases...), ParamSize: model.Current.ParamSize,
				ContextWindowSize: model.Current.ContextWindowSize, Description: model.Current.Description,
				Capabilities: append([]string{}, model.Current.Capabilities...),
				Reasoning:    modelReasoning{Type: model.Current.Reasoning.Type, Efforts: append([]string{}, model.Current.Reasoning.Efforts...)},
				LoRAs:        append([]string{}, model.Current.LoRAs...), QualityScore: model.Current.QualityScore,
				Modality: model.Current.Modality, Tags: append([]string{}, model.Current.Tags...),
			},
		}
	}
	return encodeResult(modelPageOutput{
		Data: items, NextCursor: page.NextCursor, HasMore: page.HasMore, PageSize: pageSize,
	})
}

func (provider *Provider) listExamples(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext, raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input examplesListInput
	if err := decodeInput(raw, &input); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	if _, err := provider.routingScope(ctx, invocation); err != nil {
		return agentmanagement.ToolResult{}, err
	}
	page, err := provider.examples.List(ExampleQuery(input))
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	return encodeResult(page)
}

func (provider *Provider) getRecipe(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext, raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input recipeGetInput
	if err := decodeInput(raw, &input); err != nil || routingmanagement.ValidateResourceID(input.RecipeID) != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	recipe, err := provider.authorizedRecipe(ctx, invocation, input.RecipeID)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	return encodeResult(recipeGetOutput{Data: recipeDTO(recipe), ETag: recipeETag(recipe.Revision)})
}

func (provider *Provider) validateRecipe(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext, raw json.RawMessage,
) (agentmanagement.ToolResult, error) {
	var input recipeValidateInput
	if err := decodeInput(raw, &input); err != nil || routingmanagement.ValidateResourceID(input.RecipeID) != nil ||
		input.ExpectedRevision < 0 {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	recipe, err := provider.authorizedRecipe(ctx, invocation, input.RecipeID)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	if input.ExpectedRevision > 0 && input.ExpectedRevision != recipe.Revision {
		return agentmanagement.ToolResult{}, agentmanagement.ErrConflict
	}
	document := input.Document
	if len(document) == 0 {
		document = recipe.Current.Document
	}
	canonical, decisions, compileErr := routingmanagement.CompileRecipeDocument(recipe.ID, document)
	result := recipeValidateOutput{
		ETag: recipeETag(recipe.Revision), Diagnostics: []recipeDiagnostic{},
		Topology: recipeTopology{
			SignalCounts: map[string]int{}, ProjectionCounts: map[string]int{},
			Decisions: []recipeDecisionView{},
		},
	}
	if compileErr != nil {
		result.Diagnostics = append(result.Diagnostics, recipeDiagnostic{
			Severity: "error", Code: "invalid_recipe", Message: safeDiagnostic(compileErr.Error()),
		})
		return encodeResult(result)
	}
	result.Valid = true
	result.CanonicalDocument = canonical
	result.Topology.Decisions = decisionViews(decisions)
	result.Topology.SignalCounts, result.Topology.ProjectionCounts = componentCounts(canonical)
	return encodeResult(result)
}

func (provider *Provider) routingScope(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext,
) (accesscontrol.ResultScope, error) {
	return provider.resolveScope(ctx, invocation, accesscontrol.PermissionRoutingRead)
}

func (provider *Provider) resolveScope(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext, permission accesscontrol.Permission,
) (accesscontrol.ResultScope, error) {
	if uuid.Validate(invocation.NamespaceID) != nil || uuid.Validate(invocation.PrincipalID) != nil ||
		uuid.Validate(invocation.SessionID) != nil || invocation.AuthorityDigest == "" {
		return accesscontrol.ResultScope{}, agentmanagement.ErrDenied
	}
	scope, err := provider.scopes.ResolveResultScope(
		ctx, accesscontrol.ManagementPrincipalID(invocation.PrincipalID),
		accesscontrol.NamespaceID(invocation.NamespaceID), permission,
	)
	if err != nil {
		return accesscontrol.ResultScope{}, err
	}
	canonical, err := scope.Canonical()
	if err != nil || canonical.NamespaceID != accesscontrol.NamespaceID(invocation.NamespaceID) {
		return accesscontrol.ResultScope{}, agentmanagement.ErrDenied
	}
	return canonical, nil
}

func (provider *Provider) authorizedRecipe(
	ctx context.Context, invocation agentmanagement.ToolInvocationContext, recipeID string,
) (routingmanagement.Recipe, error) {
	scope, err := provider.routingScope(ctx, invocation)
	if err != nil {
		return routingmanagement.Recipe{}, err
	}
	if !scopeContains(scope, accesscontrol.ScopeResourceRecipe, recipeID) {
		return routingmanagement.Recipe{}, routingmanagement.ErrNotFound
	}
	return provider.routing.GetRecipe(ctx, invocation.NamespaceID, recipeID)
}

func scopeContains(scope accesscontrol.ResultScope, kind accesscontrol.ScopeResourceType, id string) bool {
	canonical, err := scope.Canonical()
	if err != nil {
		return false
	}
	if canonical.All {
		return true
	}
	for _, candidate := range canonical.IDs(kind) {
		if string(candidate) == id {
			return true
		}
	}
	return false
}

func sessionVisible(scope accesscontrol.ResultScope, session agentmanagement.Session) bool {
	canonical, err := scope.Canonical()
	if err != nil {
		return false
	}
	if canonical.All || scopeContains(canonical, accesscontrol.ScopeResourceAgentSession, session.ID) {
		return true
	}
	for _, userID := range canonical.UserIDs {
		if string(userID) == session.EffectiveUserID {
			return true
		}
	}
	for _, teamID := range canonical.TeamIDs {
		if string(teamID) == session.EffectiveTeamID {
			return true
		}
	}
	return false
}

func recipeDTO(recipe routingmanagement.Recipe) recipeView {
	origin := recipe.Origin
	if origin == "" {
		origin = routingmanagement.RecipeOriginCustom
	}
	return recipeView{
		ID: recipe.ID, Name: recipe.Name, Description: recipe.Description,
		Status: string(recipe.Status), Revision: recipe.Revision,
		RecipeRevision: recipe.Current.Revision, Origin: string(origin),
		Immutable: origin == routingmanagement.RecipeOriginDistribution,
		Decisions: decisionViews(recipe.Current.Decisions),
		Document:  append(json.RawMessage(nil), recipe.Current.Document...),
	}
}

func decisionViews(decisions []routingsnapshot.Decision) []recipeDecisionView {
	result := make([]recipeDecisionView, len(decisions))
	for index, decision := range decisions {
		result[index] = recipeDecisionView{
			Name: decision.Name, DispatchCardinality: string(decision.DispatchCardinality),
		}
	}
	return result
}

func componentCounts(document json.RawMessage) (map[string]int, map[string]int) {
	var root struct {
		Signals     map[string][]json.RawMessage `json:"signals"`
		Projections map[string][]json.RawMessage `json:"projections"`
	}
	if err := json.Unmarshal(document, &root); err != nil {
		return map[string]int{}, map[string]int{}
	}
	signals := make(map[string]int, len(root.Signals))
	for name, values := range root.Signals {
		signals[name] = len(values)
	}
	projections := make(map[string]int, len(root.Projections))
	for name, values := range root.Projections {
		projections[name] = len(values)
	}
	return signals, projections
}

func recipeETag(revision int64) string {
	return fmt.Sprintf(`"rcp:%d"`, revision)
}

func safeDiagnostic(value string) string {
	value = strings.Map(func(character rune) rune {
		if character < 0x20 || character == 0x7f {
			return ' '
		}
		return character
	}, value)
	value = strings.TrimSpace(value)
	runes := []rune(value)
	if len(runes) > 1024 {
		value = string(runes[:1024])
	}
	return value
}

func decodeInput(raw json.RawMessage, target any) error {
	if len(raw) == 0 || len(raw) > 1<<20 || !json.Valid(raw) {
		return agentmanagement.ErrInvalid
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return agentmanagement.ErrInvalid
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return agentmanagement.ErrInvalid
	}
	return nil
}

func encodeResult(value any) (agentmanagement.ToolResult, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return agentmanagement.ToolResult{}, err
	}
	if len(encoded) == 0 || len(encoded) > maximumInlineResultBytes {
		return agentmanagement.ToolResult{}, agentmanagement.ErrToolUnavailable
	}
	return agentmanagement.ToolResult{Value: encoded}, nil
}

var _ interface {
	Current(context.Context, string) ([]agentmanagement.RegisteredTool, error)
	Resolve(context.Context, string, agentmanagement.ToolDefinition) (agentmanagement.ToolHandler, error)
} = (*Provider)(nil)
