package config

import (
	"fmt"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// canonicalCatalogInput adapts the existing v0.3 hierarchy to the catalog's
// normalized resource graph. This is the only place where public config joins
// aliases, cards, provider definitions, and operator evaluations.
func canonicalCatalogInput(canonical *CanonicalConfig) (modelcatalog.CompileInput, error) {
	builder, err := newCatalogInputBuilder(canonical)
	if err != nil {
		return modelcatalog.CompileInput{}, err
	}
	for modelIndex, model := range canonical.Providers.Models {
		if err := builder.addModel(model, modelIndex); err != nil {
			return builder.input, err
		}
	}
	if err := builder.finishCards(canonical); err != nil {
		return builder.input, err
	}
	return builder.input, nil
}

type catalogInputBuilder struct {
	input             modelcatalog.CompileInput
	builtIn           *modelcatalog.Registry
	cards             map[string]RoutingModel
	boundCards        map[string]struct{}
	materializedCards map[string]struct{}
}

func newCatalogInputBuilder(canonical *CanonicalConfig) (*catalogInputBuilder, error) {
	input := modelcatalog.CompileInput{Defaults: modelcatalog.Defaults{
		Model:           canonical.Providers.Defaults.DefaultModel,
		ReasoningEffort: canonical.Providers.Defaults.DefaultReasoningEffort,
	}}
	builtIn, err := modelcatalog.BuiltIn()
	if err != nil {
		return nil, fmt.Errorf("load built-in model catalog: %w", err)
	}
	cards := make(map[string]RoutingModel, len(canonical.Routing.ModelCards))
	for cardIndex, card := range canonical.Routing.ModelCards {
		if _, exists := cards[card.Name]; exists {
			return nil, fmt.Errorf("routing.modelCards[%d].name %q is duplicated", cardIndex, card.Name)
		}
		cards[card.Name] = card
	}
	return &catalogInputBuilder{
		input: input, builtIn: builtIn, cards: cards,
		boundCards:        make(map[string]struct{}, len(canonical.Providers.Models)),
		materializedCards: make(map[string]struct{}, len(canonical.Providers.Models)),
	}, nil
}

func (builder *catalogInputBuilder) addModel(model CanonicalProviderModel, modelIndex int) error {
	if strings.TrimSpace(model.Catalog) != "" && model.Reasoning != nil {
		return fmt.Errorf("providers.models[%d].reasoning is only valid for a custom model without catalog", modelIndex)
	}
	cardID := effectiveCanonicalCardID(model)
	alias := catalogModelAlias(model, cardID)
	for backendIndex, backend := range model.BackendRefs {
		instance, binding := catalogProviderBinding(model, backend, backendIndex)
		builder.input.Providers = append(builder.input.Providers, instance)
		alias.Providers = append(alias.Providers, binding)
	}
	builder.input.Models = append(builder.input.Models, alias)
	builder.boundCards[cardID] = struct{}{}
	card, shouldMaterialize := builder.cardForModel(model, cardID)
	if !shouldMaterialize {
		return nil
	}
	builder.markLoRAAliasesBound(card)
	return builder.materializeCard(card, model, modelIndex)
}

func catalogModelAlias(model CanonicalProviderModel, cardID string) modelcatalog.ModelAlias {
	return modelcatalog.ModelAlias{
		Name: model.Name, Catalog: cardID,
		BindingDefaults: modelcatalog.ModelProviderBinding{
			ModelID: model.ProviderModelID, Protocol: catalogProtocolForAPIFormat(model.APIFormat),
			Pricing: catalogPricing(model.Pricing), Reliability: catalogReliability(model.Reliability),
			ExternalModelIDs: copyStringMap(model.ExternalModelIDs),
		},
	}
}

func catalogProviderBinding(
	model CanonicalProviderModel,
	backend CanonicalBackendRef,
	backendIndex int,
) (modelcatalog.ProviderInstance, modelcatalog.ModelProviderBinding) {
	instanceName := canonicalEndpointName(model.Name, backend, backendIndex)
	providerID := strings.TrimSpace(backend.Provider)
	if providerID == "" {
		providerID = "vllm"
	}
	instance := modelcatalog.ProviderInstance{
		Name: instanceName, Catalog: providerID, BaseURL: canonicalBackendURL(backend),
		Credentials: modelcatalog.CredentialsRef{APIKey: backend.APIKey, APIKeyEnv: backend.APIKeyEnv},
		Headers:     copyStringMap(backend.ExtraHeaders), APIVersion: backend.APIVersion,
		AuthHeader: backend.AuthHeader, AuthPrefix: backend.AuthPrefix, ChatPath: backend.ChatPath,
	}
	binding := modelcatalog.ModelProviderBinding{
		Name: instanceName, ModelID: catalogProviderModelID(model, providerID),
		Protocol: catalogProtocolForAPIFormat(model.APIFormat), Pricing: catalogPricing(model.Pricing),
		Reliability: catalogReliability(model.Reliability), ExternalModelIDs: copyStringMap(model.ExternalModelIDs),
	}
	return instance, binding
}

func catalogProviderModelID(model CanonicalProviderModel, providerID string) string {
	if external := model.ExternalModelIDs[providerID]; external != "" {
		return external
	}
	if model.ProviderModelID != "" {
		return model.ProviderModelID
	}
	if model.Catalog == "" {
		return model.Name
	}
	return ""
}

func (builder *catalogInputBuilder) cardForModel(
	model CanonicalProviderModel,
	cardID string,
) (RoutingModel, bool) {
	card, hasCard := builder.cards[cardID]
	if strings.TrimSpace(model.Catalog) == "" {
		// An operator-owned model remains custom even when its request-facing
		// name happens to match a built-in catalog identity. Materialize the
		// sparse custom card so it cannot inherit built-in metadata or evidence.
		if !hasCard {
			card.Name = cardID
		}
		return card, true
	}
	_, isBuiltIn := builder.builtIn.Model(cardID)
	if !hasCard && !isBuiltIn {
		return RoutingModel{Name: cardID}, true
	}
	if model.Reasoning != nil && card.Name == "" {
		card.Name = cardID
	}
	return card, hasCard || model.Reasoning != nil
}

func (builder *catalogInputBuilder) markLoRAAliasesBound(card RoutingModel) {
	// A LoRA request alias can have a metadata-only card and inherits the base
	// model's provider binding instead of declaring another provider model.
	for _, lora := range card.LoRAs {
		if name := strings.TrimSpace(lora.Name); name != "" {
			builder.boundCards[name] = struct{}{}
		}
	}
}

func (builder *catalogInputBuilder) materializeCard(
	card RoutingModel,
	model CanonicalProviderModel,
	modelIndex int,
) error {
	if _, alreadyMaterialized := builder.materializedCards[card.Name]; alreadyMaterialized {
		return nil
	}
	overlay, records, err := catalogCardOverlay(card, model, modelIndex, builder.builtIn)
	if err != nil {
		return err
	}
	builder.input.ModelCards = append(builder.input.ModelCards, overlay)
	builder.input.Evaluations.Records = append(builder.input.Evaluations.Records, records...)
	builder.materializedCards[card.Name] = struct{}{}
	return nil
}

func (builder *catalogInputBuilder) finishCards(canonical *CanonicalConfig) error {
	if len(canonical.Providers.Models) == 0 {
		return builder.materializeRoutingOnlyCards(canonical.Routing.ModelCards)
	}
	for cardIndex, card := range canonical.Routing.ModelCards {
		if _, bound := builder.boundCards[card.Name]; !bound {
			return fmt.Errorf(
				"routing.modelCards[%d].name %q does not match a providers.models catalog identity",
				cardIndex, card.Name,
			)
		}
	}
	return nil
}

func (builder *catalogInputBuilder) materializeRoutingOnlyCards(cards []RoutingModel) error {
	// Routing fragments intentionally contain cards without provider bindings.
	// Materialize each as a metadata-only local alias for offline consumers.
	for cardIndex, card := range cards {
		if _, alreadyBound := builder.boundCards[card.Name]; alreadyBound {
			continue
		}
		if err := builder.materializeCard(
			card, CanonicalProviderModel{Name: card.Name}, cardIndex,
		); err != nil {
			return err
		}
		builder.input.Models = append(builder.input.Models, modelcatalog.ModelAlias{
			Name: card.Name, Catalog: card.Name,
		})
	}
	return nil
}

func canonicalBackendURL(backend CanonicalBackendRef) string {
	if strings.TrimSpace(backend.BaseURL) != "" {
		return strings.TrimSpace(backend.BaseURL)
	}
	endpoint := strings.TrimSpace(backend.Endpoint)
	if endpoint == "" || strings.Contains(endpoint, "://") {
		return endpoint
	}
	return defaultProtocol(backend.Protocol) + "://" + endpoint
}

func catalogProtocolForAPIFormat(apiFormat string) string {
	switch apiFormat {
	case APIFormatResponses:
		return "openai/responses@1"
	case APIFormatAnthropic:
		return "anthropic/messages@1"
	case APIFormatOpenAI:
		return "openai/chat-completions@1"
	default:
		return ""
	}
}

func catalogPricing(value ModelPricing) modelcatalog.Pricing {
	return modelcatalog.Pricing{
		Currency: value.Currency, PromptPer1M: value.PromptPer1M,
		CompletionPer1M: value.CompletionPer1M, CachedInputPer1M: value.CachedInputPer1M,
		CacheWritePer1M: value.CacheWritePer1M,
	}
}

func catalogReliability(value ProviderReliability) modelcatalog.Reliability {
	return modelcatalog.Reliability{
		LBPolicy: value.LBPolicy, RetryCount: value.RetryCount, RetryOn: value.RetryOn,
		Consecutive5xx: value.Consecutive5xx, BaseEjectionTime: value.BaseEjectionTime,
		MaxEjectionPercent: value.MaxEjectionPercent, HealthCheckPath: value.HealthCheckPath,
		HealthCheckInterval: value.HealthCheckInterval, HealthCheckTimeout: value.HealthCheckTimeout,
	}
}

func catalogCardOverlay(
	card RoutingModel,
	model CanonicalProviderModel,
	modelIndex int,
	builtIn *modelcatalog.Registry,
) (modelcatalog.ModelCardOverlay, []modelcatalog.EvaluationRecord, error) {
	catalogBacked := strings.TrimSpace(model.Catalog) != ""
	overlay := modelcatalog.ModelCardOverlay{Name: card.Name, BuiltIn: &catalogBacked}
	applyCatalogCardStrings(card, &overlay)
	applyCatalogCardLimits(card, &overlay)
	applyCatalogCardLists(card, &overlay)
	applyCatalogCardReasoning(model, &overlay)
	overlay.Evaluations = cloneUserEvaluations(card.Evaluations)
	records, err := catalogEvaluationRecords(card, modelIndex, builtIn)
	if err != nil {
		return overlay, nil, err
	}
	return overlay, records, nil
}

func applyCatalogCardStrings(card RoutingModel, overlay *modelcatalog.ModelCardOverlay) {
	setStringPointer(card.DisplayName, &overlay.DisplayName)
	setStringPointer(card.Description, &overlay.Description)
	setStringPointer(card.Publisher, &overlay.Publisher)
	setStringPointer(card.Family, &overlay.Family)
	setStringPointer(card.ParamSize, &overlay.ParameterSize)
	setStringPointer(card.Revision, &overlay.Revision)
	setStringPointer(card.ReleasedAt, &overlay.ReleasedAt)
	setStringPointer(card.KnowledgeCutoff, &overlay.KnowledgeCutoff)
	setStringPointer(card.Lifecycle, &overlay.Lifecycle)
}

func applyCatalogCardLimits(card RoutingModel, overlay *modelcatalog.ModelCardOverlay) {
	if card.ContextWindowSize > 0 {
		value := card.ContextWindowSize
		overlay.ContextWindowSize = &value
	}
	if card.MaxOutputTokens > 0 {
		value := card.MaxOutputTokens
		overlay.MaxOutputTokens = &value
	}
}

func applyCatalogCardLists(card RoutingModel, overlay *modelcatalog.ModelCardOverlay) {
	if card.Presentation != nil {
		value := *card.Presentation
		overlay.Presentation = &value
	}
	if card.Distribution != nil {
		value := *card.Distribution
		overlay.Distribution = &value
	}
	if card.Capabilities != nil {
		value := append([]string(nil), card.Capabilities...)
		overlay.Capabilities = &value
	}
	if card.Modalities != nil {
		value := *card.Modalities
		overlay.Modalities = &value
	}
	if card.Modality != "" {
		value := card.Modality
		overlay.RuntimeModality = &value
	}
	if card.Tags != nil {
		value := append([]string(nil), card.Tags...)
		overlay.Tags = &value
	}
	applyCatalogCardLoRAs(card.LoRAs, overlay)
}

func applyCatalogCardLoRAs(loras []LoRAAdapter, overlay *modelcatalog.ModelCardOverlay) {
	if loras != nil {
		value := make([]modelcatalog.LoRAAdapter, 0, len(loras))
		for _, lora := range loras {
			value = append(value, modelcatalog.LoRAAdapter{Name: lora.Name, Description: lora.Description})
		}
		overlay.LoRAs = &value
	}
}

func applyCatalogCardReasoning(model CanonicalProviderModel, overlay *modelcatalog.ModelCardOverlay) {
	if model.Reasoning != nil {
		if model.Reasoning.Family != "" {
			value := model.Reasoning.Family
			overlay.ReasoningFamily = &value
		} else {
			overlay.Reasoning = &modelcatalog.ReasoningFamilyDefinition{
				Type: model.Reasoning.Type, Parameter: model.Reasoning.Parameter,
				Levels: append([]string(nil), model.Reasoning.Levels...), Default: model.Reasoning.Default,
			}
		}
	}
}

func setStringPointer(value string, target **string) {
	if value == "" {
		return
	}
	copy := value
	*target = &copy
}
