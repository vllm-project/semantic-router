package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

const maximumManagedRecipeDocumentBytes = 2 << 20

// ManagedRecipeDocument is the strict model-free Recipe source stored beside
// immutable publication metadata. It carries readable Decision names only;
// publication metadata owns compiled Decision identities.
type ManagedRecipeDocument struct {
	Signals     CanonicalSignals     `yaml:"signals,omitempty"`
	Projections CanonicalProjections `yaml:"projections,omitempty"`
	Decisions   []Decision           `yaml:"decisions"`
	Strategy    RoutingStrategy      `yaml:"strategy,omitempty"`
}

func ParseManagedRecipeDocument(document json.RawMessage) (ManagedRecipeDocument, json.RawMessage, error) {
	if len(document) == 0 || len(document) > maximumManagedRecipeDocumentBytes || !json.Valid(document) {
		return ManagedRecipeDocument{}, nil, fmt.Errorf("managed Recipe document must be valid JSON no larger than %d bytes", maximumManagedRecipeDocumentBytes)
	}
	decoder := json.NewDecoder(bytes.NewReader(document))
	decoder.UseNumber()
	var root map[string]any
	if err := decoder.Decode(&root); err != nil || root == nil {
		return ManagedRecipeDocument{}, nil, fmt.Errorf("managed Recipe document must be a JSON object")
	}
	if err := requireManagedRecipeEOF(decoder); err != nil {
		return ManagedRecipeDocument{}, nil, err
	}
	allowed := map[string]struct{}{
		"signals": {}, "projections": {}, "decisions": {}, "strategy": {},
	}
	for field := range root {
		if _, ok := allowed[field]; !ok {
			return ManagedRecipeDocument{}, nil, fmt.Errorf("managed Recipe document contains unsupported field %q", field)
		}
	}
	var parsed ManagedRecipeDocument
	if err := yaml.UnmarshalStrict(document, &parsed); err != nil {
		return ManagedRecipeDocument{}, nil, fmt.Errorf("decode managed Recipe document: %w", err)
	}
	if len(parsed.Decisions) > 256 {
		return ManagedRecipeDocument{}, nil, fmt.Errorf("managed Recipe must define at most 256 decisions")
	}
	for index := range parsed.Decisions {
		if parsed.Decisions[index].ID != "" {
			return ManagedRecipeDocument{}, nil, fmt.Errorf(
				"managed Recipe decisions[%d].id is compiler-owned; use the Decision name in authoring documents",
				index,
			)
		}
	}
	if err := validateRecipeDocumentModelFree(CanonicalRouting(parsed)); err != nil {
		return ManagedRecipeDocument{}, nil, fmt.Errorf("managed Recipe document: %w", err)
	}
	canonical, err := json.Marshal(root)
	if err != nil {
		return ManagedRecipeDocument{}, nil, fmt.Errorf("canonicalize managed Recipe document: %w", err)
	}
	return parsed, canonical, nil
}

func requireManagedRecipeEOF(decoder *json.Decoder) error {
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("managed Recipe document contains trailing JSON")
		}
		return fmt.Errorf("decode managed Recipe document trailer: %w", err)
	}
	return nil
}

// CompileManagedRoutingSnapshot applies one already-compiled publication to a
// bootstrap config. It never converts strict resources back through the human
// authoring DTOs.
func CompileManagedRoutingSnapshot(
	base *RouterConfig,
	snapshot *routingsnapshot.Snapshot,
) (*RouterConfig, error) {
	if base == nil {
		return nil, fmt.Errorf("managed routing bootstrap configuration is required")
	}
	if base.ControlPlane.Mode != ControlPlaneModeManaged {
		return nil, fmt.Errorf("managed routing snapshot requires managed control-plane mode")
	}
	if snapshot == nil {
		return nil, fmt.Errorf("managed routing snapshot is required")
	}
	verified, err := verifyManagedRoutingSnapshot(snapshot)
	if err != nil {
		return nil, err
	}

	bootstrap := CanonicalConfig{
		Version: "v0.4", BillingCurrency: base.BillingCurrency,
		Listeners: append([]Listener(nil), base.Listeners...),
		Global:    CanonicalGlobalFromRouterConfig(base),
	}
	compiled, err := normalizeCanonicalConfig(&bootstrap, nil)
	if err != nil {
		return nil, fmt.Errorf("clone managed routing bootstrap: %w", err)
	}
	if err := applyRoutingSnapshotState(compiled, verified); err != nil {
		return nil, fmt.Errorf("compile managed routing snapshot: %w", err)
	}
	compiled.ConfigBaseDir = base.ConfigBaseDir
	compiled.DocumentHash = verified.Digest
	compiled.SkipExternalAssetValidation = base.SkipExternalAssetValidation
	if err := finalizeParsedConfig(compiled); err != nil {
		return nil, fmt.Errorf("validate managed routing snapshot: %w", err)
	}
	return compiled, nil
}

// ValidateManagedRoutingSnapshot validates the complete routing contract
// without deployment-specific bootstrap state. Publication transactions use
// this validator in tests and embedding applications may use it when their
// bootstrap compiler is unavailable. Production managed composition validates
// with CompileManagedRoutingSnapshot so bootstrap-specific contracts are also
// enforced before a desired revision is committed.
func ValidateManagedRoutingSnapshot(snapshot *routingsnapshot.Snapshot) error {
	verified, err := verifyManagedRoutingSnapshot(snapshot)
	if err != nil {
		return err
	}
	validation := DefaultGlobalConfig()
	validation.SkipExternalAssetValidation = true
	if err := applyRoutingSnapshotState(&validation, verified); err != nil {
		return fmt.Errorf("materialize managed routing snapshot: %w", err)
	}
	if err := validateConfigStructure(&validation); err != nil {
		return fmt.Errorf("validate managed routing snapshot: %w", err)
	}
	return nil
}

func verifyManagedRoutingSnapshot(snapshot *routingsnapshot.Snapshot) (*routingsnapshot.Snapshot, error) {
	if snapshot == nil {
		return nil, fmt.Errorf("managed routing snapshot is required")
	}
	verified, err := routingsnapshot.Compile(snapshot.Bundle)
	if err != nil {
		return nil, fmt.Errorf("verify managed routing snapshot: %w", err)
	}
	if snapshot.Digest == "" || snapshot.Digest != verified.Digest {
		return nil, fmt.Errorf("managed routing snapshot digest mismatch")
	}
	return verified, nil
}

func applyRoutingSnapshotState(cfg *RouterConfig, snapshot *routingsnapshot.Snapshot) error {
	if cfg == nil || snapshot == nil {
		return fmt.Errorf("router config and routing snapshot are required")
	}
	cfg.BillingCurrency = snapshot.Currency
	cfg.RoutingSnapshot = snapshot
	applySnapshotModels(cfg, snapshot.Models)
	if err := applySnapshotRecipes(cfg, snapshot.Recipes); err != nil {
		return err
	}
	if err := applySnapshotEntrypoints(cfg, snapshot.Entrypoints, snapshot.Models); err != nil {
		return err
	}
	return cfg.PrepareEntrypointRecipes()
}

func applySnapshotModels(cfg *RouterConfig, models []routingsnapshot.Model) {
	cfg.ModelConfig = make(map[string]ModelParams, len(models))
	cfg.ReasoningFamilies = make(map[string]ReasoningFamilyConfig)
	cfg.VLLMEndpoints = nil
	for _, model := range models {
		loras := make([]LoRAAdapter, 0, len(model.LoRAs))
		for _, name := range model.LoRAs {
			loras = append(loras, LoRAAdapter{Name: name})
		}
		reasoningFamily := ""
		if model.Reasoning.Type != "" {
			reasoningFamily = model.ID
			cfg.ReasoningFamilies[reasoningFamily] = ReasoningFamilyConfig{
				Type: model.Reasoning.Type, Parameter: reasoningParameter(model.Reasoning.Type),
			}
		}
		cfg.ModelConfig[model.Name] = ModelParams{
			ResourceID: model.ID, ResourceRevision: model.Revision,
			Aliases: append([]string(nil), model.Aliases...), Reasoning: model.Reasoning,
			ParamSize: model.ParamSize, ContextWindowSize: model.ContextWindowSize,
			Description: model.Description, Capabilities: append([]string(nil), model.Capabilities...),
			LoRAs: loras, Tags: append([]string(nil), model.Tags...),
			QualityScore: model.QualityScore, Modality: model.Modality,
			Execution: ModelExecutionSettings{
				MaxRetries: model.Execution.MaxRetries, RequestTimeout: model.Execution.RequestTimeout,
				StreamTimeout: model.Execution.StreamTimeout,
			},
			RuntimePricing: ModelRuntimePricing{
				InputCostPerMillionTokens:      cloneStringPointer(model.Pricing.InputCostPerMillionTokens),
				OutputCostPerMillionTokens:     cloneStringPointer(model.Pricing.OutputCostPerMillionTokens),
				CacheReadCostPerMillionTokens:  cloneStringPointer(model.Pricing.CacheReadCostPerMillionTokens),
				CacheWriteCostPerMillionTokens: cloneStringPointer(model.Pricing.CacheWriteCostPerMillionTokens),
			},
			ReasoningFamily: reasoningFamily,
		}
	}
}

func applySnapshotRecipes(cfg *RouterConfig, recipes []routingsnapshot.Recipe) error {
	cfg.Recipes = make([]RoutingRecipe, 0, len(recipes))
	for _, recipe := range recipes {
		document, _, err := ParseManagedRecipeDocument(recipe.Document)
		if err != nil {
			return fmt.Errorf("recipe %s@%d: %w", recipe.ID, recipe.Revision, err)
		}
		decisions, err := materializeManagedRecipeDecisions(recipe, document.Decisions)
		if err != nil {
			return err
		}
		strategy := document.Strategy
		if strategy == "" {
			strategy = cfg.Strategy
		}
		ensureModelRefDefaults(decisions)
		cfg.Recipes = append(cfg.Recipes, RoutingRecipe{
			ID: recipe.ID, Revision: recipe.Revision, Name: RecipeName(recipe.Name),
			Description: recipe.Description,
			Profile: RoutingProfile{
				Signals:     normalizeSignals(document.Signals, decisions),
				Projections: normalizeProjections(document.Projections),
				Decisions:   decisions, Strategy: strategy,
			},
		})
	}
	return nil
}

func materializeManagedRecipeDecisions(
	recipe routingsnapshot.Recipe,
	authored []Decision,
) ([]Decision, error) {
	if len(recipe.Decisions) != len(authored) {
		return nil, fmt.Errorf("recipe %s@%d decision metadata does not match its document", recipe.ID, recipe.Revision)
	}
	metadataByName := make(map[string]routingsnapshot.Decision, len(recipe.Decisions))
	for _, metadata := range recipe.Decisions {
		if _, duplicate := metadataByName[metadata.Name]; duplicate {
			return nil, fmt.Errorf("recipe %s@%d has duplicate decision metadata name %q", recipe.ID, recipe.Revision, metadata.Name)
		}
		metadataByName[metadata.Name] = metadata
	}

	decisions := copyDecisions(authored)
	for index := range decisions {
		decision := &decisions[index]
		metadata, found := metadataByName[decision.Name]
		if !found {
			return nil, fmt.Errorf("recipe %s@%d decision metadata does not match its document", recipe.ID, recipe.Revision)
		}
		algorithmType := ""
		if decision.Algorithm != nil {
			algorithmType = decision.Algorithm.Type
		}
		cardinality, known := DecisionAlgorithmDispatchCardinality(algorithmType)
		if !known || routingsnapshot.DispatchCardinality(cardinality) != metadata.DispatchCardinality {
			return nil, fmt.Errorf("recipe %s@%d decision %q cardinality does not match its document", recipe.ID, recipe.Revision, decision.Name)
		}
		decision.ID = metadata.ID
		delete(metadataByName, decision.Name)
	}
	if len(metadataByName) != 0 {
		return nil, fmt.Errorf("recipe %s@%d decision metadata does not match its document", recipe.ID, recipe.Revision)
	}
	return decisions, nil
}

func applySnapshotEntrypoints(
	cfg *RouterConfig,
	entrypoints []routingsnapshot.Entrypoint,
	models []routingsnapshot.Model,
) error {
	modelsByID := make(map[string]routingsnapshot.Model, len(models))
	for _, model := range models {
		modelsByID[model.ID] = model
	}
	cfg.Entrypoints = make([]EntrypointMapping, 0, len(entrypoints))
	for _, entrypoint := range entrypoints {
		mapping := EntrypointMapping{
			ID: entrypoint.ID, Revision: entrypoint.Revision, Name: entrypoint.Name,
			ModelNames: append([]string(nil), entrypoint.Aliases...),
			Rules:      make([]EntrypointRule, 0, len(entrypoint.Rules)),
		}
		for _, rule := range entrypoint.Rules {
			recipe, found := cfg.RecipeByID(rule.RecipeID)
			if !found || recipe.Revision != rule.RecipeRevision {
				return fmt.Errorf("entrypoint %s rule %s references unavailable recipe %s@%d", entrypoint.ID, rule.ID, rule.RecipeID, rule.RecipeRevision)
			}
			assignments, err := runtimeAssignmentsFromSnapshot(rule.Assignments, modelsByID)
			if err != nil {
				return fmt.Errorf("entrypoint %s rule %s: %w", entrypoint.ID, rule.ID, err)
			}
			mapping.Rules = append(mapping.Rules, EntrypointRule{
				ID: rule.ID, Name: rule.Name, Matches: runtimeMatchesFromSnapshot(rule.Matchers),
				Action: EntrypointRuleAction{
					RecipeID: rule.RecipeID, RecipeRevision: rule.RecipeRevision,
					Recipe: recipe.Name, Assignments: assignments,
				},
			})
		}
		cfg.Entrypoints = append(cfg.Entrypoints, mapping)
	}
	return nil
}

func runtimeAssignmentsFromSnapshot(
	input map[string]routingsnapshot.AssignmentSet,
	modelsByID map[string]routingsnapshot.Model,
) (map[string]RoutingAssignmentSet, error) {
	result := make(map[string]RoutingAssignmentSet, len(input))
	for decisionID, set := range input {
		runtimeSet := RoutingAssignmentSet{Models: make([]RoutingModelAssignment, 0, len(set.Models))}
		for _, assignment := range set.Models {
			model, found := modelsByID[assignment.ModelID]
			if !found || model.Revision != assignment.ModelRevision {
				return nil, fmt.Errorf("assignment references unavailable Model %s@%d", assignment.ModelID, assignment.ModelRevision)
			}
			var reasoning *RoutingAssignmentReasoning
			if assignment.Reasoning != nil {
				reasoning = &RoutingAssignmentReasoning{
					Enabled: assignment.Reasoning.Enabled, Effort: assignment.Reasoning.Effort,
					Description: assignment.Reasoning.Description,
				}
			}
			runtimeSet.Models = append(runtimeSet.Models, RoutingModelAssignment{
				ModelID: assignment.ModelID, ModelRevision: assignment.ModelRevision,
				ModelName: model.Name, Priority: assignment.Priority, Weight: assignment.Weight,
				LoRAName: assignment.LoRAName, Reasoning: reasoning,
			})
		}
		if set.Fallback != nil {
			runtimeSet.Fallback = &RoutingFallbackPolicy{
				Strategy: set.Fallback.Strategy, On: append([]string(nil), set.Fallback.On...),
			}
		}
		result[decisionID] = runtimeSet
	}
	return result, nil
}

func runtimeMatchesFromSnapshot(input []routingsnapshot.Matcher) []EntrypointMatch {
	result := make([]EntrypointMatch, 0, len(input))
	for _, matcher := range input {
		switch {
		case matcher.Claim != nil:
			result = append(result, EntrypointMatch{Claim: &EntrypointClaimMatch{
				Name: matcher.Claim.Name,
				Value: EntrypointClaimValue{
					Kind: matcher.Claim.Value.Kind, String: matcher.Claim.Value.String,
					Boolean: matcher.Claim.Value.Boolean, Integer: matcher.Claim.Value.Integer,
				},
			}})
		case matcher.ExactPath != "":
			result = append(result, EntrypointMatch{Path: &EntrypointPathMatch{Exact: matcher.ExactPath}})
		case matcher.PathPrefix != "":
			result = append(result, EntrypointMatch{Path: &EntrypointPathMatch{Prefix: matcher.PathPrefix}})
		}
	}
	return result
}

func entrypointAliases(name string, aliases []string) []string {
	result := make([]string, 0, len(aliases))
	for _, alias := range aliases {
		if alias != name {
			result = append(result, alias)
		}
	}
	return stableUniqueStrings(result)
}
