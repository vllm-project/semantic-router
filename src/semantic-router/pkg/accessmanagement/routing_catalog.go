package accessmanagement

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// GetRoutingCatalog returns only resources discoverable by one applied API-key
// policy. The immutable routing revision is read from that policy's publication
// pin, so a Management read cannot mix old grants with a newer topology.
func (service *Service) GetRoutingCatalog(
	ctx context.Context,
	namespaceID string,
	subject Subject,
) (RoutingCatalog, error) {
	if subject.Kind != accesscontrol.SubjectKindAPIKey {
		return RoutingCatalog{}, ErrInvalidRequest
	}
	desired, err := service.load(ctx, namespaceID, subject)
	if err != nil {
		return RoutingCatalog{}, err
	}
	applied, err := service.appliedKeyPolicy(ctx, desired)
	if err != nil {
		return RoutingCatalog{}, err
	}
	if applied.Active.RoutingRevision <= 0 || !validDigest(applied.Active.RoutingDocumentDigest) ||
		strings.TrimSpace(applied.Active.PublicationID) == "" || applied.Active.RuntimeEpoch == 0 {
		return RoutingCatalog{}, fmt.Errorf("%w: applied key policy has no routing publication pin", ErrUnavailable)
	}
	publication, err := service.routing.ReadRoutingPublication(ctx, RoutingPublicationPin{
		NamespaceID: namespaceID, QuotaPartition: desired.QuotaPartition,
		PublicationID: applied.Active.PublicationID, RuntimeEpoch: applied.Active.RuntimeEpoch,
		RoutingRevision:       applied.Active.RoutingRevision,
		RoutingDocumentDigest: applied.Active.RoutingDocumentDigest,
	})
	if err != nil {
		return RoutingCatalog{}, fmt.Errorf("%w: read applied routing publication: %w", ErrUnavailable, err)
	}
	if publication == nil || publication.Snapshot.NamespaceID != namespaceID ||
		publication.Snapshot.Revision != applied.Active.RoutingRevision ||
		publication.RoutingDocumentDigest != applied.Active.RoutingDocumentDigest ||
		!validDigest(publication.Snapshot.Digest) {
		return RoutingCatalog{}, fmt.Errorf("%w: applied routing publication does not match the key policy pin", ErrUnavailable)
	}
	catalog, err := compileRoutingCatalog(subject, applied, *publication)
	if err != nil {
		return RoutingCatalog{}, fmt.Errorf("%w: compile key-scoped routing catalog: %w", ErrUnavailable, err)
	}
	return catalog, nil
}

func (service *Service) appliedKeyPolicy(
	ctx context.Context,
	snapshot PolicySnapshot,
) (accessruntime.AppliedPolicy, error) {
	if service == nil || service.applied == nil || snapshot.Subject.Kind != accesscontrol.SubjectKindAPIKey {
		return accessruntime.AppliedPolicy{}, ErrInvalidRequest
	}
	applied, err := service.applied.ReadAppliedPolicy(
		ctx, snapshot.NamespaceID, snapshot.QuotaPartition, snapshot.Subject.ID,
	)
	if err != nil {
		if errors.Is(err, accessruntime.ErrProjectionNotFound) {
			return accessruntime.AppliedPolicy{}, ErrNotFound
		}
		return accessruntime.AppliedPolicy{}, fmt.Errorf("%w: read applied key policy: %w", ErrUnavailable, err)
	}
	if applied.Active.KeyID != snapshot.Subject.ID || applied.Active.Revision == 0 ||
		applied.Active.Revision != applied.Projection.Revision ||
		applied.Active.Digest != applied.Projection.Digest ||
		applied.Projection.KeyID != snapshot.Subject.ID ||
		applied.Projection.NamespaceID != snapshot.NamespaceID ||
		applied.Projection.QuotaPartition != snapshot.QuotaPartition ||
		!validDigest(applied.Active.Digest) {
		return accessruntime.AppliedPolicy{}, fmt.Errorf("%w: applied key policy identity is inconsistent", ErrUnavailable)
	}
	return applied, nil
}

func validDigest(value string) bool {
	if len(value) != 64 {
		return false
	}
	for _, character := range value {
		if !strings.ContainsRune("0123456789abcdef", character) {
			return false
		}
	}
	return true
}

func compileRoutingCatalog(
	subject Subject,
	applied accessruntime.AppliedPolicy,
	publication RoutingPublication,
) (RoutingCatalog, error) {
	snapshot := publication.Snapshot
	result := RoutingCatalog{
		Subject: subject, PolicyRevision: applied.Active.Revision,
		PolicyDigest: applied.Active.Digest, RoutingRevision: snapshot.Revision,
		RoutingDigest: publication.RoutingDocumentDigest,
	}
	visibleModels := make(map[string]struct{})
	for _, model := range snapshot.Models {
		if !discoverable(applied.Projection, accesscontrol.GrantResourceModel, model.ID) {
			continue
		}
		visibleModels[model.ID] = struct{}{}
		result.Models = append(result.Models, catalogModel(model))
	}
	visibleRecipes := make(map[string]struct{})
	for _, entrypoint := range snapshot.Entrypoints {
		if !discoverable(applied.Projection, accesscontrol.GrantResourceEntrypoint, entrypoint.ID) {
			continue
		}
		view := RoutingCatalogEntrypoint{
			ID: entrypoint.ID, Revision: entrypoint.Revision, Name: entrypoint.Name,
			Aliases: append([]string(nil), entrypoint.Aliases...),
		}
		for _, rule := range entrypoint.Rules {
			visibleRecipes[rule.RecipeID] = struct{}{}
			view.Rules = append(view.Rules, catalogRule(rule, visibleModels))
		}
		result.Entrypoints = append(result.Entrypoints, view)
	}
	for _, recipe := range snapshot.Recipes {
		if _, visible := visibleRecipes[recipe.ID]; !visible {
			continue
		}
		topology, err := catalogRecipeTopology(recipe.Document)
		if err != nil {
			return RoutingCatalog{}, fmt.Errorf("recipe %q topology: %w", recipe.ID, err)
		}
		result.Recipes = append(result.Recipes, RoutingCatalogRecipe{
			ID: recipe.ID, Revision: recipe.Revision, Name: recipe.Name,
			Description: recipe.Description,
			Decisions:   append([]routingsnapshot.Decision(nil), recipe.Decisions...),
			Signals:     topology.Signals,
			Projections: topology.Projections,
		})
	}
	return result, nil
}

var catalogSignalTypes = []string{
	"keywords", "embeddings", "domains", "fact_check", "user_feedbacks", "reasks",
	"preferences", "language", "context", "structure", "complexity", "modality",
	"role_bindings", "jailbreak", "pii", "kb", "conversation", "events", "metadata",
	"classifiers",
}

type catalogRecipeDocument struct {
	Signals map[string][]struct {
		Name string `json:"name"`
	} `json:"signals"`
	Projections struct {
		Partitions []struct {
			Name    string   `json:"name"`
			Members []string `json:"members"`
		} `json:"partitions"`
		Scores []struct {
			Name   string `json:"name"`
			Inputs []struct {
				Type   string `json:"type"`
				Name   string `json:"name"`
				KB     string `json:"kb"`
				Metric string `json:"metric"`
			} `json:"inputs"`
		} `json:"scores"`
		Mappings []struct {
			Name    string `json:"name"`
			Source  string `json:"source"`
			Outputs []struct {
				Name string `json:"name"`
			} `json:"outputs"`
		} `json:"mappings"`
	} `json:"projections"`
}

type catalogRecipeTopologyView struct {
	Signals     []RoutingCatalogSignal
	Projections []RoutingCatalogProjection
}

func catalogRecipeTopology(document json.RawMessage) (catalogRecipeTopologyView, error) {
	var source catalogRecipeDocument
	if len(document) == 0 || !json.Valid(document) {
		return catalogRecipeTopologyView{}, fmt.Errorf("document is not valid JSON")
	}
	if err := json.Unmarshal(document, &source); err != nil {
		return catalogRecipeTopologyView{}, fmt.Errorf("decode document: %w", err)
	}
	result := catalogRecipeTopologyView{
		Signals:     make([]RoutingCatalogSignal, 0),
		Projections: make([]RoutingCatalogProjection, 0),
	}
	// Deliberately whitelist signal families. A new family remains private until
	// this security projection explicitly chooses how to represent it.
	for _, signalType := range catalogSignalTypes {
		for _, signal := range source.Signals[signalType] {
			if name := strings.TrimSpace(signal.Name); name != "" {
				result.Signals = append(result.Signals, RoutingCatalogSignal{Type: signalType, Name: name})
			}
		}
	}
	for _, projection := range source.Projections.Partitions {
		result.Projections = append(result.Projections, RoutingCatalogProjection{
			Type: "partition", Name: projection.Name,
			Members: append([]string{}, projection.Members...), Inputs: []RoutingCatalogProjectionReference{}, Outputs: []string{},
		})
	}
	for _, projection := range source.Projections.Scores {
		view := RoutingCatalogProjection{
			Type: "score", Name: projection.Name, Members: []string{},
			Inputs: make([]RoutingCatalogProjectionReference, 0, len(projection.Inputs)), Outputs: []string{},
		}
		for _, input := range projection.Inputs {
			view.Inputs = append(view.Inputs, RoutingCatalogProjectionReference{
				Type: input.Type, Name: input.Name, KB: input.KB, Metric: input.Metric,
			})
		}
		result.Projections = append(result.Projections, view)
	}
	for _, projection := range source.Projections.Mappings {
		view := RoutingCatalogProjection{
			Type: "mapping", Name: projection.Name, Members: []string{},
			Inputs: []RoutingCatalogProjectionReference{}, Source: projection.Source,
			Outputs: make([]string, 0, len(projection.Outputs)),
		}
		for _, output := range projection.Outputs {
			view.Outputs = append(view.Outputs, output.Name)
		}
		result.Projections = append(result.Projections, view)
	}
	return result, nil
}

func discoverable(
	projection accessprojection.Projection,
	resourceType accesscontrol.GrantResourceType,
	resourceID string,
) bool {
	return projection.Evaluate(resourceType, resourceID, accesscontrol.GrantPermissionDiscover) ==
		accesscontrol.AccessDecisionAllow
}

func catalogModel(model routingsnapshot.Model) RoutingCatalogModel {
	return RoutingCatalogModel{
		ID: model.ID, Revision: model.Revision, Name: model.Name,
		Aliases: append([]string(nil), model.Aliases...), ParamSize: model.ParamSize,
		ContextWindowSize: model.ContextWindowSize, Description: model.Description,
		Capabilities: append([]string(nil), model.Capabilities...),
		Reasoning: routingsnapshot.ReasoningFamily{
			Type: model.Reasoning.Type, Efforts: append([]string(nil), model.Reasoning.Efforts...),
		},
		LoRAs: append([]string(nil), model.LoRAs...), QualityScore: model.QualityScore,
		Modality: model.Modality, Tags: append([]string(nil), model.Tags...),
		Pricing: cloneCatalogPricing(model.Pricing),
	}
}

func catalogRule(
	rule routingsnapshot.EntrypointRule,
	visibleModels map[string]struct{},
) RoutingCatalogRule {
	result := RoutingCatalogRule{
		ID: rule.ID, Name: rule.Name, Matchers: cloneCatalogMatchers(rule.Matchers),
		RecipeID: rule.RecipeID, RecipeRevision: rule.RecipeRevision,
		Assignments: make(map[string]RoutingCatalogAssignmentSet, len(rule.Assignments)),
	}
	for decisionID, set := range rule.Assignments {
		filtered := RoutingCatalogAssignmentSet{Fallback: cloneCatalogFallback(set.Fallback)}
		for _, assignment := range set.Models {
			if _, visible := visibleModels[assignment.ModelID]; !visible {
				continue
			}
			copy := assignment
			if assignment.Reasoning != nil {
				reasoning := *assignment.Reasoning
				copy.Reasoning = &reasoning
			}
			filtered.Models = append(filtered.Models, copy)
		}
		result.Assignments[decisionID] = filtered
	}
	return result
}

func cloneCatalogMatchers(values []routingsnapshot.Matcher) []routingsnapshot.Matcher {
	result := make([]routingsnapshot.Matcher, len(values))
	for index, matcher := range values {
		result[index] = matcher
		if matcher.Claim != nil {
			claim := *matcher.Claim
			result[index].Claim = &claim
		}
	}
	return result
}

func cloneCatalogFallback(value *routingsnapshot.FallbackPolicy) *routingsnapshot.FallbackPolicy {
	if value == nil {
		return nil
	}
	return &routingsnapshot.FallbackPolicy{
		Strategy: value.Strategy,
		On:       append([]string(nil), value.On...),
	}
}

func cloneCatalogPricing(value routingsnapshot.ModelPricing) routingsnapshot.ModelPricing {
	clone := func(source *string) *string {
		if source == nil {
			return nil
		}
		copy := *source
		return &copy
	}
	return routingsnapshot.ModelPricing{
		InputCostPerMillionTokens:      clone(value.InputCostPerMillionTokens),
		OutputCostPerMillionTokens:     clone(value.OutputCostPerMillionTokens),
		CacheReadCostPerMillionTokens:  clone(value.CacheReadCostPerMillionTokens),
		CacheWriteCostPerMillionTokens: clone(value.CacheWriteCostPerMillionTokens),
	}
}
