package config

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelauthoring"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// fileAuthoringBundle is the private bridge from public v0.3 names to the
// immutable Model/Recipe/Entrypoint compiler shared with Management.
type fileAuthoringBundle struct {
	Models      []AuthoringModel
	Recipes     []AuthoringRecipe
	Entrypoints []AuthoringEntrypoint
	Credentials map[string]BackendCredentialConfig
}

func buildFileAuthoringBundle(canonical *CanonicalConfig) (fileAuthoringBundle, error) {
	if canonical == nil {
		return fileAuthoringBundle{}, fmt.Errorf("canonical config is required")
	}
	models, modelsByName, credentials, err := publicModelsToAuthoring(canonical)
	if err != nil {
		return fileAuthoringBundle{}, err
	}
	recipes, recipesByName, err := publicRecipesToAuthoring(canonical)
	if err != nil {
		return fileAuthoringBundle{}, err
	}
	entrypoints, err := publicEntrypointsToAuthoring(canonical.Entrypoints, recipesByName)
	if err != nil {
		return fileAuthoringBundle{}, err
	}
	implicitNames := canonicalImplicitAutoModelNames(canonical)
	if len(implicitNames) != 0 {
		defaultRecipe, found := recipesByName[string(DefaultRecipeName)]
		if !found {
			return fileAuthoringBundle{}, fmt.Errorf(
				"global.router.auto_model_names require a top-level routing profile",
			)
		}
		assignments, assignmentErr := embeddedRecipeAssignments(defaultRecipe)
		if assignmentErr != nil {
			return fileAuthoringBundle{}, fmt.Errorf("implicit auto Entrypoint: %w", assignmentErr)
		}
		entrypoints = append(entrypoints, AuthoringEntrypoint{
			Name: implicitNames[0], Aliases: append([]string(nil), implicitNames[1:]...),
			Recipe: string(DefaultRecipeName), Assignments: assignments,
		})
	}
	if err := validateAuthoringRecipes(recipes); err != nil {
		return fileAuthoringBundle{}, err
	}
	if err := validateAuthoringEntrypoints(entrypoints, recipes, modelsByName); err != nil {
		return fileAuthoringBundle{}, err
	}
	return fileAuthoringBundle{
		Models: models, Recipes: recipes, Entrypoints: entrypoints, Credentials: credentials,
	}, nil
}

// canonicalImplicitAutoModelNames preserves the established v0.3 automatic
// routing surface for a top-level routing profile. An explicitly present empty
// auto_model_names list disables it so an authored Entrypoint can own those
// names without a second routing authority.
func canonicalImplicitAutoModelNames(canonical *CanonicalConfig) []string {
	if canonical == nil || !canonicalRoutingHasProfile(canonical.Routing) {
		return nil
	}
	if canonical.Global != nil && canonical.Global.Router.AutoModelNames != nil {
		return normalizeAutoModelNames(*canonical.Global.Router.AutoModelNames)
	}
	primary := DefaultAutoModelName
	if canonical.Global != nil && strings.TrimSpace(canonical.Global.Router.AutoModelName) != "" {
		primary = canonical.Global.Router.AutoModelName
	}
	names := normalizeAutoModelNames([]string{
		DefaultVSRAutoModelName,
		LegacyAutoModelAlias,
		primary,
	})
	// An explicit Entrypoint that claims an established automatic-routing name
	// is already the single authority for that request surface. Preserve that
	// v0.3 form without synthesizing a competing default Entrypoint.
	claimed := make(map[string]struct{})
	for _, entrypoint := range canonical.Entrypoints {
		for _, name := range normalizeEntrypointModelNames(entrypoint.ModelNames) {
			claimed[name] = struct{}{}
		}
	}
	for _, name := range names {
		if _, found := claimed[name]; found {
			return nil
		}
	}
	return names
}

func publicModelsToAuthoring(
	canonical *CanonicalConfig,
) ([]AuthoringModel, map[string]AuthoringModel, map[string]BackendCredentialConfig, error) {
	cards := canonicalRoutingModels(canonical.Routing)
	cardsByName := make(map[string]RoutingModel, len(cards))
	for index, card := range cards {
		name := strings.TrimSpace(card.Name)
		if name == "" || name != card.Name {
			return nil, nil, nil, fmt.Errorf("routing.modelCards[%d].name must be non-empty without surrounding whitespace", index)
		}
		if _, duplicate := cardsByName[name]; duplicate {
			return nil, nil, nil, fmt.Errorf("routing.modelCards[%s]: duplicate model name", name)
		}
		cardsByName[name] = card
	}

	models := make([]AuthoringModel, 0, len(canonical.Providers.Models))
	modelsByName := make(map[string]AuthoringModel, len(canonical.Providers.Models))
	credentials := make(map[string]BackendCredentialConfig)
	for modelIndex := range canonical.Providers.Models {
		source := &canonical.Providers.Models[modelIndex]
		name := strings.TrimSpace(source.Name)
		if name == "" || name != source.Name {
			return nil, nil, nil, fmt.Errorf("providers.models[%d].name must be non-empty without surrounding whitespace", modelIndex)
		}
		if _, duplicate := modelsByName[name]; duplicate {
			return nil, nil, nil, fmt.Errorf("providers.models[%s]: duplicate model name", name)
		}
		if strings.TrimSpace(source.APIFormat) != source.APIFormat {
			return nil, nil, nil, fmt.Errorf(
				"providers.models[%s].api_format cannot contain surrounding whitespace", name,
			)
		}
		card, found := cardsByName[name]
		if !found {
			return nil, nil, nil, fmt.Errorf("providers.models[%s] does not match any routing.modelCards entry", name)
		}
		execution, effectivePricing, err := compileModelControl(name, source.Control, source.Pricing)
		if err != nil {
			return nil, nil, nil, err
		}
		connections, generated, err := publicBackendRefsToConnections(*source)
		if err != nil {
			return nil, nil, nil, err
		}
		for credentialName, credential := range generated {
			credentials[credentialName] = credential
		}
		if len(connections) == 0 {
			return nil, nil, nil, fmt.Errorf("providers.models[%s].backend_refs must contain at least one backend", name)
		}
		reasoning, err := publicReasoningFamily(canonical.Providers.Defaults, *source)
		if err != nil {
			return nil, nil, nil, err
		}
		reasoning, err = mergePublicModelReasoning(name, reasoning, card.Reasoning)
		if err != nil {
			return nil, nil, nil, err
		}
		loras := make([]string, 0, len(card.LoRAs))
		for _, adapter := range card.LoRAs {
			loras = append(loras, adapter.Name)
		}
		model := AuthoringModel{
			Name: name,
			Card: AuthoringModelCard{
				ParamSize: card.ParamSize, ContextWindowSize: card.ContextWindowSize,
				Description: card.Description, Capabilities: append([]string(nil), card.Capabilities...),
				Reasoning: reasoning, LoRAs: loras, QualityScore: card.QualityScore,
				Modality: card.Modality, Tags: append([]string(nil), card.Tags...),
			},
			Connections:    connections,
			Execution:      execution,
			RuntimePricing: effectivePricing,
		}
		models = append(models, model)
		modelsByName[name] = model
	}
	if len(cardsByName) != len(modelsByName) {
		for name := range cardsByName {
			if _, found := modelsByName[name]; !found {
				return nil, nil, nil, fmt.Errorf("routing.modelCards[%s] does not match any providers.models entry", name)
			}
		}
	}
	return models, modelsByName, credentials, nil
}

func publicReasoningFamily(
	defaults CanonicalProviderDefaults,
	model CanonicalProviderModel,
) (routingsnapshot.ReasoningFamily, error) {
	name := strings.TrimSpace(model.ReasoningFamily)
	if name != model.ReasoningFamily {
		return routingsnapshot.ReasoningFamily{}, fmt.Errorf(
			"providers.models[%s].reasoning_family cannot contain surrounding whitespace", model.Name,
		)
	}
	if name == "" {
		return routingsnapshot.ReasoningFamily{}, nil
	}
	family, found := defaults.ReasoningFamilies[name]
	if !found {
		return routingsnapshot.ReasoningFamily{}, fmt.Errorf(
			"providers.models[%s].reasoning_family %q not found in providers.defaults.reasoning_families",
			model.Name, name,
		)
	}
	return routingsnapshot.ReasoningFamily{Type: family.Type}, nil
}

func mergePublicModelReasoning(
	modelName string,
	providerFamily routingsnapshot.ReasoningFamily,
	card ModelReasoning,
) (routingsnapshot.ReasoningFamily, error) {
	cardType := strings.TrimSpace(card.Type)
	if cardType != card.Type {
		return routingsnapshot.ReasoningFamily{}, fmt.Errorf(
			"routing.modelCards[%s].reasoning.type cannot contain surrounding whitespace", modelName,
		)
	}
	if providerFamily.Type == "" && (cardType != "" || len(card.Efforts) > 0) {
		return routingsnapshot.ReasoningFamily{}, fmt.Errorf(
			"routing.modelCards[%s].reasoning requires providers.models[%s].reasoning_family", modelName, modelName,
		)
	}
	if cardType != "" && cardType != providerFamily.Type {
		return routingsnapshot.ReasoningFamily{}, fmt.Errorf(
			"routing.modelCards[%s].reasoning.type %q does not match providers.models[%s].reasoning_family type %q",
			modelName, cardType, modelName, providerFamily.Type,
		)
	}
	providerFamily.Efforts = append([]string(nil), card.Efforts...)
	return providerFamily, nil
}

func publicBackendRefsToConnections(
	model CanonicalProviderModel,
) ([]modelauthoring.Connection, map[string]BackendCredentialConfig, error) {
	connections := make([]modelauthoring.Connection, 0, len(model.BackendRefs))
	credentials := make(map[string]BackendCredentialConfig)
	providerModelID := strings.TrimSpace(model.ProviderModelID)
	if providerModelID != model.ProviderModelID {
		return nil, nil, fmt.Errorf("providers.models[%s].provider_model_id cannot contain surrounding whitespace", model.Name)
	}
	if providerModelID == "" {
		providerModelID = model.Name
	}
	for index, backend := range model.BackendRefs {
		backendName := strings.TrimSpace(backend.Name)
		if backendName != backend.Name {
			return nil, nil, fmt.Errorf(
				"providers.models[%s].backend_refs[%d].name cannot contain surrounding whitespace", model.Name, index,
			)
		}
		providerID := strings.TrimSpace(backend.Provider)
		if providerID != backend.Provider {
			return nil, nil, fmt.Errorf(
				"providers.models[%s].backend_refs[%d].provider cannot contain surrounding whitespace", model.Name, index,
			)
		}
		if providerID == "" {
			providerID = "vllm"
		}
		endpoint, endpointErr := publicBackendOrigin(backend)
		if endpointErr != nil {
			return nil, nil, fmt.Errorf(
				"providers.models[%s].backend_refs[%d]: %w", model.Name, index, endpointErr,
			)
		}
		credentialName, credential, credentialErr := publicBackendCredential(
			model.Name, index, backendName, providerID, endpoint, backend,
		)
		if credentialErr != nil {
			return nil, nil, credentialErr
		}
		if credential != nil {
			credentials[credentialName] = *credential
		}
		weight := ""
		if backend.Weight < 0 {
			return nil, nil, fmt.Errorf(
				"providers.models[%s].backend_refs[%d].weight cannot be negative", model.Name, index,
			)
		}
		if backend.Weight > 0 {
			weight = strconv.Itoa(backend.Weight)
		}
		providerInterface := strings.TrimSpace(backend.Type)
		if providerInterface != backend.Type {
			return nil, nil, fmt.Errorf(
				"providers.models[%s].backend_refs[%d].type cannot contain surrounding whitespace", model.Name, index,
			)
		}
		if providerInterface == "" {
			providerInterface = publicProviderInterface(model.APIFormat)
		}
		connectionFields := make(map[string]any)
		if backend.APIVersion != "" {
			connectionFields["api_version"] = backend.APIVersion
		}
		if len(connectionFields) == 0 {
			connectionFields = nil
		}
		connections = append(connections, modelauthoring.Connection{
			Name: backendName, Provider: providerID,
			Interface: providerInterface, Endpoint: endpoint,
			Model: providerModelID, Credential: credentialName, Weight: weight,
			ConnectionFields: connectionFields,
			Transport: modelauthoring.TransportOverrides{
				Path: backend.ChatPath, Headers: clonePublicBackendHeaders(backend.ExtraHeaders),
			},
		})
	}
	return connections, credentials, nil
}

func publicBackendCredential(
	modelName string,
	index int,
	backendName string,
	providerID string,
	endpoint string,
	backend CanonicalBackendRef,
) (string, *BackendCredentialConfig, error) {
	credentialRef := strings.TrimSpace(backend.Credential)
	if credentialRef != backend.Credential ||
		(credentialRef != "" && (backend.APIKey != "" || backend.APIKeyEnv != "")) {
		return "", nil, fmt.Errorf(
			"providers.models[%s].backend_refs[%d] credential must be canonical and mutually exclusive with api_key/api_key_env",
			modelName, index,
		)
	}
	if backend.APIKey != "" && backend.APIKeyEnv != "" {
		return "", nil, fmt.Errorf(
			"providers.models[%s].backend_refs[%d] must configure only one of api_key or api_key_env",
			modelName, index,
		)
	}
	if backend.APIKey == "" && backend.APIKeyEnv == "" {
		if backend.AuthHeader != "" || backend.AuthPrefix != "" {
			return "", nil, fmt.Errorf(
				"providers.models[%s].backend_refs[%d] auth_header/auth_prefix require api_key or api_key_env",
				modelName, index,
			)
		}
		return credentialRef, nil, nil
	}

	adapterID, err := publicCredentialAdapter(backend.AuthHeader, backend.AuthPrefix)
	if err != nil {
		return "", nil, fmt.Errorf(
			"providers.models[%s].backend_refs[%d]: %w", modelName, index, err,
		)
	}
	backendIdentity := backendName
	if backendIdentity == "" {
		backendIdentity = strconv.Itoa(index)
	}
	credentialName := stableProviderCredentialID(modelName, backendIdentity, providerID, endpoint)
	return credentialName, &BackendCredentialConfig{
		CredentialAdapterID: adapterID,
		SecretEnv:           backend.APIKeyEnv,
		SecretValue:         backend.APIKey,
	}, nil
}

func stableProviderCredentialID(modelName, backendIdentity, providerID, origin string) string {
	identity := strings.Join(
		[]string{"vllm-sr/provider-credential/v1", modelName, backendIdentity, providerID, origin},
		"\x00",
	)
	return uuid.NewSHA1(uuid.NameSpaceOID, []byte(identity)).String()
}

func clonePublicBackendHeaders(input map[string]string) map[string]string {
	if len(input) == 0 {
		return nil
	}
	result := make(map[string]string, len(input))
	for name, value := range input {
		result[name] = value
	}
	return result
}

func copyStringMap(input map[string]string) map[string]string {
	return clonePublicBackendHeaders(input)
}

func copyReasoningFamilies(
	input map[string]ReasoningFamilyConfig,
) map[string]ReasoningFamilyConfig {
	if len(input) == 0 {
		return nil
	}
	result := make(map[string]ReasoningFamilyConfig, len(input))
	for name, family := range input {
		result[name] = family
	}
	return result
}

func publicBackendOrigin(backend CanonicalBackendRef) (string, error) {
	baseURL := strings.TrimSpace(backend.BaseURL)
	endpoint := strings.TrimSpace(backend.Endpoint)
	if baseURL != backend.BaseURL || endpoint != backend.Endpoint {
		return "", fmt.Errorf("endpoint and base_url cannot contain surrounding whitespace")
	}
	if baseURL != "" && endpoint != "" {
		return "", fmt.Errorf("endpoint and base_url are mutually exclusive")
	}
	origin := baseURL
	if origin == "" {
		origin = endpoint
	}
	protocol := strings.ToLower(strings.TrimSpace(backend.Protocol))
	if strings.TrimSpace(backend.Protocol) != backend.Protocol {
		return "", fmt.Errorf("protocol cannot contain surrounding whitespace")
	}
	if origin == "" {
		return "", nil
	}
	if strings.Contains(origin, "://") {
		if protocol != "" && !strings.HasPrefix(strings.ToLower(origin), protocol+"://") {
			return "", fmt.Errorf("protocol %q conflicts with the endpoint scheme", backend.Protocol)
		}
		return origin, nil
	}
	if protocol == "" {
		protocol = "http"
	}
	return protocol + "://" + origin, nil
}

func publicProviderInterface(apiFormat string) string {
	switch strings.TrimSpace(apiFormat) {
	case "", "openai", "openai-chat", "openai.chat.v1":
		return ""
	case "openai-responses", "openai.responses.v1":
		return "responses"
	case "anthropic", "anthropic.messages.v1":
		return "messages"
	default:
		return strings.TrimSpace(apiFormat)
	}
}

func publicCredentialAdapter(authHeader, authPrefix string) (string, error) {
	header := strings.ToLower(strings.TrimSpace(authHeader))
	prefix := strings.TrimSpace(authPrefix)
	if strings.TrimSpace(authHeader) != authHeader ||
		(strings.TrimSpace(authPrefix) != authPrefix && !strings.EqualFold(authPrefix, "Bearer ")) {
		return "", fmt.Errorf("auth_header and auth_prefix cannot contain surrounding whitespace")
	}
	if header == "" {
		header = "authorization"
	}
	switch header {
	case "authorization":
		if prefix == "" || strings.EqualFold(prefix, "bearer") {
			return "bearer", nil
		}
	case "x-api-key":
		if prefix == "" {
			return "x-api-key", nil
		}
	case "api-key":
		if prefix == "" {
			return "api-key", nil
		}
	}
	return "", fmt.Errorf(
		"auth_header %q with auth_prefix %q cannot be materialized by an installed credential adapter",
		authHeader, authPrefix,
	)
}

func publicRecipesToAuthoring(
	canonical *CanonicalConfig,
) ([]AuthoringRecipe, map[string]CanonicalRecipe, error) {
	sources := append([]CanonicalRecipe(nil), canonical.Recipes...)
	if canonicalRoutingHasProfile(canonical.Routing) {
		defaultRouting := canonical.Routing
		defaultRouting.ModelCards = nil
		sources = append([]CanonicalRecipe{{Name: string(DefaultRecipeName), Routing: defaultRouting}}, sources...)
	}
	result := make([]AuthoringRecipe, 0, len(sources))
	byName := make(map[string]CanonicalRecipe, len(sources))
	for index, source := range sources {
		name := strings.TrimSpace(source.Name)
		if name == "" || name != source.Name {
			return nil, nil, fmt.Errorf("recipes[%d].name must be non-empty without surrounding whitespace", index)
		}
		if _, duplicate := byName[name]; duplicate {
			return nil, nil, fmt.Errorf("recipes[%s]: duplicate recipe name", name)
		}
		if len(source.Routing.ModelCards) != 0 {
			return nil, nil, fmt.Errorf("recipes[%s].routing.modelCards is forbidden; use top-level routing.modelCards", name)
		}
		document := source.Routing
		document.Decisions = cloneEntrypointDecisions(source.Routing.Decisions)
		for decisionIndex := range document.Decisions {
			document.Decisions[decisionIndex].ID = ""
			stripRoutingRecipeModelSelection(&document.Decisions[decisionIndex])
		}
		result = append(result, AuthoringRecipe{Name: name, Description: source.Description, Document: document})
		byName[name] = source
	}
	return result, byName, nil
}

func publicEntrypointsToAuthoring(
	entrypoints []CanonicalEntrypoint,
	recipes map[string]CanonicalRecipe,
) ([]AuthoringEntrypoint, error) {
	result := make([]AuthoringEntrypoint, 0, len(entrypoints))
	for index, source := range entrypoints {
		names := normalizeEntrypointModelNames(source.ModelNames)
		if len(names) == 0 {
			return nil, fmt.Errorf("entrypoints[%d].model_names cannot be empty", index)
		}
		recipe, found := recipes[strings.TrimSpace(source.Recipe)]
		if !found {
			return nil, fmt.Errorf("entrypoints[%d].recipe references unknown Recipe %q", index, source.Recipe)
		}
		assignments := publicAssignmentsToAuthoring(source.Assignments)
		if len(assignments) == 0 {
			var err error
			assignments, err = embeddedRecipeAssignments(recipe)
			if err != nil {
				return nil, fmt.Errorf("entrypoints[%d]: %w", index, err)
			}
		}
		result = append(result, AuthoringEntrypoint{
			Name: names[0], Aliases: append([]string(nil), names[1:]...),
			Recipe: recipe.Name, Assignments: assignments,
		})
	}
	return result, nil
}

// normalizeEntrypointModelNames keeps the first declared name as the primary
// request-facing name while removing blank and repeated aliases.
func normalizeEntrypointModelNames(names []string) []string {
	if len(names) == 0 {
		return nil
	}
	result := make([]string, 0, len(names))
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		normalized := strings.TrimSpace(name)
		if normalized == "" {
			continue
		}
		if _, exists := seen[normalized]; exists {
			continue
		}
		seen[normalized] = struct{}{}
		result = append(result, normalized)
	}
	return result
}

// embeddedRecipeAssignments preserves the original v0.3 convenience form for
// Decisions whose complete physical candidate set is expressed in modelRefs.
// Algorithm-local role-specific bindings cannot be flattened losslessly and
// therefore require explicit Entrypoint assignments.
func embeddedRecipeAssignments(recipe CanonicalRecipe) (map[string]AuthoringAssignmentSet, error) {
	result := make(map[string]AuthoringAssignmentSet, len(recipe.Routing.Decisions))
	for _, decision := range recipe.Routing.Decisions {
		if len(decision.ModelRefs) == 0 {
			return nil, fmt.Errorf(
				"recipe %q Decision %q has no complete modelRefs candidate set; add entrypoint assignments",
				recipe.Name, decision.Name,
			)
		}
		models := make([]AuthoringModelAssignment, 0, len(decision.ModelRefs))
		for _, ref := range decision.ModelRefs {
			weight := "1"
			if ref.Weight > 0 {
				weight = strconv.FormatFloat(ref.Weight, 'f', -1, 64)
			}
			var reasoning *AuthoringAssignmentReasoning
			if ref.UseReasoning != nil || ref.ReasoningEffort != "" || ref.ReasoningDescription != "" {
				reasoning = &AuthoringAssignmentReasoning{
					Enabled: ref.UseReasoning != nil && *ref.UseReasoning,
					Effort:  ref.ReasoningEffort, Description: ref.ReasoningDescription,
				}
			}
			models = append(models, AuthoringModelAssignment{
				Model: ref.Model, Weight: weight, LoRAName: ref.LoRAName, Reasoning: reasoning,
			})
		}
		result[decision.Name] = AuthoringAssignmentSet{Models: models}
	}
	return result, nil
}

func canonicalRoutingHasProfile(routing CanonicalRouting) bool {
	if routing.Strategy != "" || len(routing.Decisions) != 0 ||
		len(routing.Projections.Partitions) != 0 || len(routing.Projections.Scores) != 0 ||
		len(routing.Projections.Mappings) != 0 {
		return true
	}
	return len(routing.Signals.Keywords) != 0 || len(routing.Signals.Embeddings) != 0 ||
		len(routing.Signals.Domains) != 0 || len(routing.Signals.FactCheck) != 0 ||
		len(routing.Signals.UserFeedbacks) != 0 || len(routing.Signals.Reasks) != 0 ||
		len(routing.Signals.Preferences) != 0 || len(routing.Signals.Language) != 0 ||
		len(routing.Signals.Context) != 0 || len(routing.Signals.Structure) != 0 ||
		len(routing.Signals.Complexity) != 0 || len(routing.Signals.Modality) != 0 ||
		len(routing.Signals.RoleBindings) != 0 || len(routing.Signals.Jailbreak) != 0 ||
		len(routing.Signals.PII) != 0 || len(routing.Signals.KB) != 0 ||
		len(routing.Signals.Conversation) != 0 || len(routing.Signals.EventRules) != 0 ||
		len(routing.Signals.Metadata) != 0 || len(routing.Signals.Classifiers) != 0
}

func mergeGeneratedBackendCredentials(
	target *BackendCredentialsConfig,
	generated map[string]BackendCredentialConfig,
) error {
	if len(generated) == 0 {
		return nil
	}
	if target.File == nil {
		target.File = make(map[string]BackendCredentialConfig, len(generated))
	}
	for name, credential := range generated {
		if existing, duplicate := target.File[name]; duplicate && existing != credential {
			return fmt.Errorf("generated backend credential %q conflicts with global.services.backend_credentials", name)
		}
		target.File[name] = credential
	}
	return nil
}
