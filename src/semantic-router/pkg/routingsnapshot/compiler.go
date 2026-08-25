package routingsnapshot

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

var (
	decimalPattern         = regexp.MustCompile(`^(0|[1-9][0-9]*)(\.[0-9]+)?$`)
	wireFormatPattern      = regexp.MustCompile(`^[a-z][a-z0-9._-]{0,127}$`)
	catalogRevisionPattern = regexp.MustCompile(`^sha256:[a-f0-9]{64}$`)
)

func Compile(input Bundle) (*Snapshot, error) {
	if strings.TrimSpace(input.NamespaceID) == "" {
		return nil, fmt.Errorf("namespaceId is required")
	}
	if input.Revision <= 0 {
		return nil, fmt.Errorf("revision must be positive")
	}
	bundle := cloneBundle(input)
	if err := normalizeBundle(&bundle); err != nil {
		return nil, err
	}
	snapshot := &Snapshot{Bundle: bundle}
	if err := snapshot.buildIndexes(); err != nil {
		return nil, err
	}
	if err := snapshot.validateReferences(); err != nil {
		return nil, err
	}
	payload, err := json.Marshal(bundle)
	if err != nil {
		return nil, fmt.Errorf("marshal canonical routing snapshot: %w", err)
	}
	digest := sha256.Sum256(payload)
	snapshot.Digest = hex.EncodeToString(digest[:])
	semanticPayload, err := json.Marshal(routingSemantics{
		NamespaceID: bundle.NamespaceID,
		Currency:    bundle.Currency,
		Models:      bundle.Models,
		Recipes:     bundle.Recipes,
		Entrypoints: bundle.Entrypoints,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal canonical routing semantics: %w", err)
	}
	semanticDigest := sha256.Sum256(semanticPayload)
	snapshot.SemanticDigest = hex.EncodeToString(semanticDigest[:])
	return snapshot, nil
}

// routingSemantics is the executable routing value. The aggregate publication
// revision is deliberately absent: resource revisions and every field that can
// change request routing remain in the digest, while an access-only publication
// can prove that its already-warmed router runtime is reusable.
type routingSemantics struct {
	NamespaceID string       `json:"namespaceId"`
	Currency    string       `json:"currency,omitempty"`
	Models      []Model      `json:"models"`
	Recipes     []Recipe     `json:"recipes"`
	Entrypoints []Entrypoint `json:"entrypoints"`
}

func cloneBundle(input Bundle) Bundle {
	payload, _ := json.Marshal(input)
	var output Bundle
	_ = json.Unmarshal(payload, &output)
	return output
}

func normalizeBundle(bundle *Bundle) error {
	if bundle.Currency != "" && !regexp.MustCompile(`^[A-Z]{3}$`).MatchString(bundle.Currency) {
		return fmt.Errorf("currency must be an ISO-4217 code")
	}
	priced := false
	for i := range bundle.Models {
		if err := normalizeModel(&bundle.Models[i]); err != nil {
			return fmt.Errorf("models[%d]: %w", i, err)
		}
		if modelHasPrice(bundle.Models[i]) {
			priced = true
		}
	}
	if priced && bundle.Currency == "" {
		return fmt.Errorf("currency is required when a model has pricing")
	}
	for i := range bundle.Recipes {
		if err := normalizeRecipe(&bundle.Recipes[i]); err != nil {
			return fmt.Errorf("recipes[%d]: %w", i, err)
		}
	}
	for i := range bundle.Entrypoints {
		if err := normalizeEntrypoint(&bundle.Entrypoints[i]); err != nil {
			return fmt.Errorf("entrypoints[%d]: %w", i, err)
		}
	}
	sort.Slice(bundle.Models, func(i, j int) bool { return bundle.Models[i].ID < bundle.Models[j].ID })
	sort.Slice(bundle.Recipes, func(i, j int) bool { return bundle.Recipes[i].ID < bundle.Recipes[j].ID })
	sort.Slice(bundle.Entrypoints, func(i, j int) bool { return bundle.Entrypoints[i].ID < bundle.Entrypoints[j].ID })
	return nil
}

func normalizeModel(model *Model) error {
	if strings.TrimSpace(model.ID) == "" || strings.TrimSpace(model.Name) == "" {
		return fmt.Errorf("id and name are required")
	}
	if model.Revision <= 0 {
		return fmt.Errorf("revision must be positive")
	}
	if !catalogRevisionPattern.MatchString(model.CatalogRevision) {
		return fmt.Errorf("catalogRevision must be an immutable sha256 digest")
	}
	model.Aliases = uniqueSorted(model.Aliases)
	model.Capabilities = uniqueSorted(model.Capabilities)
	model.LoRAs = uniqueSorted(model.LoRAs)
	model.Tags = uniqueSorted(model.Tags)
	model.Reasoning.Efforts = uniqueSorted(model.Reasoning.Efforts)
	if !canonicalOptionalText(model.ParamSize, 128) {
		return fmt.Errorf("paramSize is invalid")
	}
	if model.ContextWindowSize < 0 || model.ContextWindowSize > 100_000_000 {
		return fmt.Errorf("contextWindowSize must be between 0 and 100000000")
	}
	if !canonicalOptionalText(model.Description, 4096) {
		return fmt.Errorf("description is invalid")
	}
	if math.IsNaN(model.QualityScore) || math.IsInf(model.QualityScore, 0) || model.QualityScore < 0 || model.QualityScore > 1 {
		return fmt.Errorf("qualityScore must be between 0 and 1")
	}
	if !canonicalOptionalText(model.Modality, 128) {
		return fmt.Errorf("modality is invalid")
	}
	if model.Execution.MaxRetries < 0 || model.Execution.MaxRetries > 5 {
		return fmt.Errorf("execution.maxRetries must be between 0 and 5")
	}
	if err := normalizeRetryTriggers(&model.Execution); err != nil {
		return err
	}
	if model.Execution.RequestTimeout == "" {
		model.Execution.RequestTimeout = "300s"
	}
	if model.Execution.StreamTimeout == "" {
		model.Execution.StreamTimeout = "300s"
	}
	for field, value := range map[string]string{
		"execution.requestTimeout": model.Execution.RequestTimeout,
		"execution.streamTimeout":  model.Execution.StreamTimeout,
	} {
		if err := validateDuration(value); err != nil {
			return fmt.Errorf("%s: %w", field, err)
		}
	}
	if model.Pricing.CacheReadCostPerMillionTokens == nil || *model.Pricing.CacheReadCostPerMillionTokens == "" {
		model.Pricing.CacheReadCostPerMillionTokens = clonePrice(model.Pricing.InputCostPerMillionTokens)
	}
	if model.Pricing.CacheWriteCostPerMillionTokens == nil || *model.Pricing.CacheWriteCostPerMillionTokens == "" {
		model.Pricing.CacheWriteCostPerMillionTokens = clonePrice(model.Pricing.InputCostPerMillionTokens)
	}
	prices := []struct {
		field string
		value *string
	}{
		{"pricing.inputCostPerMillionTokens", model.Pricing.InputCostPerMillionTokens},
		{"pricing.outputCostPerMillionTokens", model.Pricing.OutputCostPerMillionTokens},
		{"pricing.cacheReadCostPerMillionTokens", model.Pricing.CacheReadCostPerMillionTokens},
		{"pricing.cacheWriteCostPerMillionTokens", model.Pricing.CacheWriteCostPerMillionTokens},
	}
	for _, price := range prices {
		field, value := price.field, price.value
		if err := validatePrice(value); err != nil {
			return fmt.Errorf("%s: %w", field, err)
		}
	}
	if len(model.Backends) == 0 {
		return fmt.Errorf("at least one backend is required")
	}
	for i := range model.Backends {
		backend := &model.Backends[i]
		if strings.TrimSpace(backend.ID) == "" || strings.TrimSpace(backend.ProviderID) == "" ||
			strings.TrimSpace(string(backend.WireFormat)) == "" || strings.TrimSpace(backend.Origin) == "" ||
			strings.TrimSpace(backend.ProviderModelID) == "" {
			return fmt.Errorf("backends[%d] requires id, providerId, wireFormat, origin, and providerModelId", i)
		}
		if !wireFormatPattern.MatchString(string(backend.WireFormat)) {
			return fmt.Errorf("backends[%d].wireFormat must be a canonical wire format identifier", i)
		}
		parsed, normalizeModelErr := url.Parse(backend.Origin)
		if normalizeModelErr != nil || parsed.Scheme == "" || parsed.Host == "" || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
			return fmt.Errorf("backends[%d].origin must be an absolute credential-free origin", i)
		}
		if err := normalizeBackendConnection(&backend.Connection); err != nil {
			return fmt.Errorf("backends[%d].%w", i, err)
		}
		weight, normalizeModelErr := canonicalPositiveDecimal(backend.Weight, 9)
		if normalizeModelErr != nil {
			return fmt.Errorf("backends[%d].weight: %w", i, normalizeModelErr)
		}
		backend.Weight = weight
	}
	sort.Slice(model.Backends, func(i, j int) bool { return model.Backends[i].ID < model.Backends[j].ID })
	return nil
}

func normalizeRetryTriggers(execution *ModelExecution) error {
	if execution.MaxRetries == 0 {
		if len(execution.RetryOn) != 0 {
			return fmt.Errorf("execution.retryOn must be empty when maxRetries is 0")
		}
		return nil
	}
	if len(execution.RetryOn) == 0 {
		execution.RetryOn = []string{"unavailable"}
		return nil
	}
	if len(execution.RetryOn) > 2 {
		return fmt.Errorf("execution.retryOn must contain at most 2 failure classes")
	}
	order := map[string]int{"unavailable": 0, "timeout": 1}
	seen := make(map[string]struct{}, len(execution.RetryOn))
	for _, trigger := range execution.RetryOn {
		if _, ok := order[trigger]; !ok {
			return fmt.Errorf("execution.retryOn contains unsupported failure class %q", trigger)
		}
		if _, duplicate := seen[trigger]; duplicate {
			return fmt.Errorf("execution.retryOn contains duplicate failure class %q", trigger)
		}
		seen[trigger] = struct{}{}
	}
	sort.Slice(execution.RetryOn, func(i, j int) bool {
		return order[execution.RetryOn[i]] < order[execution.RetryOn[j]]
	})
	return nil
}

func clonePrice(value *string) *string {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

func modelHasPrice(model Model) bool {
	return model.Pricing.InputCostPerMillionTokens != nil ||
		model.Pricing.OutputCostPerMillionTokens != nil ||
		model.Pricing.CacheReadCostPerMillionTokens != nil ||
		model.Pricing.CacheWriteCostPerMillionTokens != nil
}

func validateDuration(value string) error {
	duration, err := time.ParseDuration(value)
	if err != nil {
		return fmt.Errorf("must be a duration")
	}
	if duration < time.Second || duration > 24*time.Hour {
		return fmt.Errorf("must be between 1s and 24h")
	}
	return nil
}

func validatePrice(value *string) error {
	if value == nil {
		return nil
	}
	canonical, err := canonicalNonNegativeDecimal(*value, 9)
	if err != nil {
		return err
	}
	parts := strings.SplitN(canonical, ".", 2)
	whole, _ := strconv.ParseUint(parts[0], 10, 64)
	if whole > 1_000_000 || (whole == 1_000_000 && len(parts) == 2 && strings.Trim(parts[1], "0") != "") {
		return fmt.Errorf("must not exceed 1000000")
	}
	*value = canonical
	return nil
}

func normalizeRecipe(recipe *Recipe) error {
	if strings.TrimSpace(recipe.ID) == "" || strings.TrimSpace(recipe.Name) == "" {
		return fmt.Errorf("id and name are required")
	}
	if recipe.Revision <= 0 || len(recipe.Decisions) == 0 || len(recipe.Document) == 0 || !json.Valid(recipe.Document) {
		return fmt.Errorf("positive revision, decisions, and valid document are required")
	}
	if !canonicalOptionalText(recipe.Description, 4096) {
		return fmt.Errorf("description is invalid")
	}
	seen := make(map[string]struct{}, len(recipe.Decisions))
	for _, decision := range recipe.Decisions {
		if strings.TrimSpace(decision.ID) == "" || strings.TrimSpace(decision.Name) == "" {
			return fmt.Errorf("decision id and name are required")
		}
		if decision.DispatchCardinality != DispatchCardinalitySingle && decision.DispatchCardinality != DispatchCardinalityMulti {
			return fmt.Errorf("decision %s has unknown dispatch cardinality", decision.ID)
		}
		if _, exists := seen[decision.ID]; exists {
			return fmt.Errorf("duplicate decision id %q", decision.ID)
		}
		seen[decision.ID] = struct{}{}
	}
	sort.Slice(recipe.Decisions, func(i, j int) bool { return recipe.Decisions[i].ID < recipe.Decisions[j].ID })
	var canonical any
	if err := json.Unmarshal(recipe.Document, &canonical); err != nil {
		return fmt.Errorf("invalid recipe document: %w", err)
	}
	recipe.Document, _ = json.Marshal(canonical)
	return nil
}

func normalizeEntrypoint(entrypoint *Entrypoint) error {
	if strings.TrimSpace(entrypoint.ID) == "" || strings.TrimSpace(entrypoint.Name) == "" {
		return fmt.Errorf("id and name are required")
	}
	if entrypoint.Revision <= 0 {
		return fmt.Errorf("revision must be positive")
	}
	entrypoint.Aliases = uniqueSorted(entrypoint.Aliases)
	if len(entrypoint.Aliases) == 0 || len(entrypoint.Rules) == 0 {
		return fmt.Errorf("aliases and rules are required")
	}
	seenRules := make(map[string]struct{}, len(entrypoint.Rules))
	for i := range entrypoint.Rules {
		if err := normalizeEntrypointRule(&entrypoint.Rules[i], i, seenRules); err != nil {
			return err
		}
	}
	sort.Slice(entrypoint.Rules, func(i, j int) bool { return entrypoint.Rules[i].ID < entrypoint.Rules[j].ID })
	return validateRuleAmbiguity(entrypoint.Rules)
}

func normalizeEntrypointRule(rule *EntrypointRule, ruleIndex int, seenRules map[string]struct{}) error {
	if strings.TrimSpace(rule.ID) == "" || strings.TrimSpace(rule.Name) == "" || strings.TrimSpace(rule.RecipeID) == "" || rule.RecipeRevision <= 0 {
		return fmt.Errorf("rules[%d] requires id, name, recipeId, and recipeRevision", ruleIndex)
	}
	if _, exists := seenRules[rule.ID]; exists {
		return fmt.Errorf("duplicate rule id %q", rule.ID)
	}
	seenRules[rule.ID] = struct{}{}
	if err := normalizeMatchers(rule.Matchers); err != nil {
		return fmt.Errorf("rules[%d].matchers: %w", ruleIndex, err)
	}
	for decisionID, assignmentSet := range rule.Assignments {
		normalized, err := normalizeAssignmentSet(assignmentSet, ruleIndex, decisionID)
		if err != nil {
			return err
		}
		rule.Assignments[decisionID] = normalized
	}
	return nil
}

func normalizeAssignmentSet(assignmentSet AssignmentSet, ruleIndex int, decisionID string) (AssignmentSet, error) {
	if len(assignmentSet.Models) == 0 || len(assignmentSet.Models) > 32 {
		return AssignmentSet{}, fmt.Errorf("rules[%d].assignments[%s].models must contain between 1 and 32 Models", ruleIndex, decisionID)
	}
	priorities := make(map[int]struct{})
	for assignmentIndex := range assignmentSet.Models {
		assignment := &assignmentSet.Models[assignmentIndex]
		if err := normalizeModelAssignment(assignment, ruleIndex, decisionID, assignmentIndex); err != nil {
			return AssignmentSet{}, err
		}
		priorities[assignment.Priority] = struct{}{}
	}
	if assignmentSet.Fallback == nil {
		if len(priorities) != 1 {
			return AssignmentSet{}, fmt.Errorf("rules[%d].assignments[%s] requires fallback when a Model priority is greater than zero", ruleIndex, decisionID)
		}
		if _, ok := priorities[0]; !ok {
			return AssignmentSet{}, fmt.Errorf("rules[%d].assignments[%s] requires priority zero when fallback is absent", ruleIndex, decisionID)
		}
	} else if err := normalizeFallbackPolicy(assignmentSet.Fallback, priorities); err != nil {
		return AssignmentSet{}, fmt.Errorf("rules[%d].assignments[%s].fallback: %w", ruleIndex, decisionID, err)
	}
	sort.SliceStable(assignmentSet.Models, func(i, j int) bool {
		return assignmentSet.Models[i].Priority < assignmentSet.Models[j].Priority
	})
	return assignmentSet, nil
}

func normalizeModelAssignment(assignment *Assignment, ruleIndex int, decisionID string, assignmentIndex int) error {
	if assignment.Priority < 0 || assignment.Priority > 31 {
		return fmt.Errorf("rules[%d].assignments[%s].models[%d].priority must be between 0 and 31", ruleIndex, decisionID, assignmentIndex)
	}
	weight, err := canonicalPositiveDecimal(assignment.Weight, 9)
	if err != nil {
		return fmt.Errorf("rules[%d].assignments[%s].models[%d].weight: %w", ruleIndex, decisionID, assignmentIndex, err)
	}
	assignment.Weight = weight
	if assignment.Reasoning == nil || assignment.Reasoning.Enabled {
		return nil
	}
	if assignment.Reasoning.Effort != "" || assignment.Reasoning.Description != "" {
		return fmt.Errorf("rules[%d].assignments[%s].models[%d].reasoning cannot set effort or description when disabled", ruleIndex, decisionID, assignmentIndex)
	}
	// Explicit disabled and omission have identical execution semantics.
	// Canonicalize them before duplicate detection and digesting.
	assignment.Reasoning = nil
	return nil
}

func normalizeFallbackPolicy(policy *FallbackPolicy, priorities map[int]struct{}) error {
	if policy.Strategy != "priority" {
		return fmt.Errorf("strategy must be priority")
	}
	if len(priorities) < 2 {
		return fmt.Errorf("at least two priority tiers are required")
	}
	for priority := 0; priority < len(priorities); priority++ {
		if _, ok := priorities[priority]; !ok {
			return fmt.Errorf("priority tiers must be contiguous from zero")
		}
	}
	if len(policy.On) == 0 || len(policy.On) > 2 {
		return fmt.Errorf("on must contain between 1 and 2 failure classes")
	}
	allowed := map[string]struct{}{"unavailable": {}, "timeout": {}}
	seen := make(map[string]struct{}, len(policy.On))
	for _, trigger := range policy.On {
		if _, ok := allowed[trigger]; !ok {
			return fmt.Errorf("on contains unsupported failure class %q", trigger)
		}
		if _, duplicate := seen[trigger]; duplicate {
			return fmt.Errorf("on contains duplicate failure class %q", trigger)
		}
		seen[trigger] = struct{}{}
	}
	order := map[string]int{"unavailable": 0, "timeout": 1}
	sort.Slice(policy.On, func(i, j int) bool { return order[policy.On[i]] < order[policy.On[j]] })
	return nil
}

func normalizeMatchers(matchers []Matcher) error {
	claimNames := make(map[string]struct{})
	pathCount := 0
	for i, matcher := range matchers {
		fields := 0
		if matcher.Claim != nil {
			fields++
		}
		if matcher.ExactPath != "" {
			fields++
			pathCount++
		}
		if matcher.PathPrefix != "" {
			fields++
			pathCount++
		}
		if fields != 1 {
			return fmt.Errorf("matcher %d must set exactly one matcher kind", i)
		}
		if matcher.Claim != nil {
			if strings.TrimSpace(matcher.Claim.Name) == "" {
				return fmt.Errorf("claim matcher %d requires a name", i)
			}
			if _, exists := claimNames[matcher.Claim.Name]; exists {
				return fmt.Errorf("duplicate claim matcher %q", matcher.Claim.Name)
			}
			claimNames[matcher.Claim.Name] = struct{}{}
			if err := matcher.Claim.Value.Validate(); err != nil {
				return fmt.Errorf("claim matcher %q: %w", matcher.Claim.Name, err)
			}
		}
		path := matcher.ExactPath
		if path == "" {
			path = matcher.PathPrefix
		}
		if path != "" && (!strings.HasPrefix(path, "/") || strings.Contains(path, "?")) {
			return fmt.Errorf("path matcher must be an absolute path without a query")
		}
	}
	if pathCount > 1 {
		return fmt.Errorf("at most one path matcher is allowed")
	}
	sort.Slice(matchers, func(i, j int) bool { return matcherKey(matchers[i]) < matcherKey(matchers[j]) })
	return nil
}

func (v ClaimValue) Validate() error {
	switch v.Kind {
	case "string":
		return nil
	case "boolean":
		if v.String != "" || v.Integer != 0 {
			return fmt.Errorf("boolean claim contains another value kind")
		}
		return nil
	case "integer":
		if v.String != "" || v.Boolean {
			return fmt.Errorf("integer claim contains another value kind")
		}
		return nil
	default:
		return fmt.Errorf("kind must be string, boolean, or integer")
	}
}

func (snapshot *Snapshot) buildIndexes() error {
	snapshot.modelsByID = make(map[string]Model, len(snapshot.Models))
	snapshot.recipesByID = make(map[string]Recipe, len(snapshot.Recipes))
	snapshot.entrypointsByID = make(map[string]Entrypoint, len(snapshot.Entrypoints))
	snapshot.aliases = make(map[string]string)
	for _, model := range snapshot.Models {
		if _, exists := snapshot.modelsByID[model.ID]; exists {
			return fmt.Errorf("duplicate model id %q", model.ID)
		}
		snapshot.modelsByID[model.ID] = model
	}
	for _, recipe := range snapshot.Recipes {
		if _, exists := snapshot.recipesByID[recipe.ID]; exists {
			return fmt.Errorf("duplicate recipe id %q", recipe.ID)
		}
		snapshot.recipesByID[recipe.ID] = recipe
	}
	for _, entrypoint := range snapshot.Entrypoints {
		if _, exists := snapshot.entrypointsByID[entrypoint.ID]; exists {
			return fmt.Errorf("duplicate entrypoint id %q", entrypoint.ID)
		}
		snapshot.entrypointsByID[entrypoint.ID] = entrypoint
		for _, alias := range entrypoint.Aliases {
			if owner, exists := snapshot.aliases[alias]; exists {
				return fmt.Errorf("entrypoint alias %q is owned by both %s and %s", alias, owner, entrypoint.ID)
			}
			snapshot.aliases[alias] = entrypoint.ID
		}
	}
	return nil
}

func (snapshot *Snapshot) validateReferences() error {
	for _, entrypoint := range snapshot.Entrypoints {
		for _, rule := range entrypoint.Rules {
			if err := snapshot.validateEntrypointRuleReferences(entrypoint.ID, rule); err != nil {
				return err
			}
		}
	}
	return nil
}

func (snapshot *Snapshot) validateEntrypointRuleReferences(entrypointID string, rule EntrypointRule) error {
	recipe, exists := snapshot.recipesByID[rule.RecipeID]
	if !exists || recipe.Revision != rule.RecipeRevision {
		return fmt.Errorf("entrypoint %s rule %s references unavailable recipe revision %s@%d", entrypointID, rule.ID, rule.RecipeID, rule.RecipeRevision)
	}
	decisions := make(map[string]Decision, len(recipe.Decisions))
	for _, decision := range recipe.Decisions {
		decisions[decision.ID] = decision
	}
	if len(rule.Assignments) != len(decisions) {
		return fmt.Errorf("entrypoint %s rule %s must assign every recipe decision", entrypointID, rule.ID)
	}
	for decisionID, assignmentSet := range rule.Assignments {
		decision, found := decisions[decisionID]
		if !found {
			return fmt.Errorf("entrypoint %s rule %s assigns unknown decision %s", entrypointID, rule.ID, decisionID)
		}
		if err := snapshot.validateAssignmentReferences(entrypointID, rule.ID, decisionID, decision, assignmentSet); err != nil {
			return err
		}
	}
	return nil
}

func (snapshot *Snapshot) validateAssignmentReferences(
	entrypointID string,
	ruleID string,
	decisionID string,
	decision Decision,
	assignmentSet AssignmentSet,
) error {
	if len(assignmentSet.Models) == 0 {
		return fmt.Errorf("entrypoint %s rule %s decision %s has no model", entrypointID, ruleID, decisionID)
	}
	if assignmentSet.Fallback != nil && decision.DispatchCardinality != DispatchCardinalitySingle {
		return fmt.Errorf("entrypoint %s rule %s decision %s cannot use fallback with %s dispatch cardinality", entrypointID, ruleID, decisionID, decision.DispatchCardinality)
	}
	seenTargets := make(map[string]struct{}, len(assignmentSet.Models))
	for i := range assignmentSet.Models {
		assignment := &assignmentSet.Models[i]
		model, exists := snapshot.modelsByID[assignment.ModelID]
		if !exists || model.Revision != assignment.ModelRevision {
			return fmt.Errorf("entrypoint %s rule %s references unavailable model revision %s@%d", entrypointID, ruleID, assignment.ModelID, assignment.ModelRevision)
		}
		if err := validateAssignmentTarget(model, assignment, entrypointID, ruleID, decisionID, seenTargets); err != nil {
			return err
		}
	}
	return nil
}

func validateAssignmentTarget(
	model Model,
	assignment *Assignment,
	entrypointID string,
	ruleID string,
	decisionID string,
	seenTargets map[string]struct{},
) error {
	reasoningKey, _ := json.Marshal(assignment.Reasoning)
	targetKey := assignment.ModelID + "\x00" + assignment.LoRAName + "\x00" + string(reasoningKey)
	if _, duplicate := seenTargets[targetKey]; duplicate {
		return fmt.Errorf("entrypoint %s rule %s decision %s repeats the same model, LoRA, and reasoning target", entrypointID, ruleID, decisionID)
	}
	seenTargets[targetKey] = struct{}{}
	if assignment.LoRAName != "" && !slices.Contains(model.LoRAs, assignment.LoRAName) {
		return fmt.Errorf("model %s does not declare LoRA %q", model.ID, assignment.LoRAName)
	}
	if assignment.Reasoning == nil {
		return nil
	}
	if model.Reasoning.Type == "" {
		return fmt.Errorf("model %s does not support reasoning controls", model.ID)
	}
	if assignment.Reasoning.Effort != "" && !slices.Contains(model.Reasoning.Efforts, assignment.Reasoning.Effort) {
		return fmt.Errorf("model %s does not support reasoning effort %q", model.ID, assignment.Reasoning.Effort)
	}
	if len(assignment.Reasoning.Description) > 1024 {
		return fmt.Errorf("reasoning description exceeds 1024 bytes")
	}
	return nil
}

func uniqueSorted(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value == "" {
			continue
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	sort.Strings(result)
	return result
}

func canonicalOptionalText(value string, maximum int) bool {
	return utf8.ValidString(value) && len(value) <= maximum &&
		(value == "" || value == strings.TrimSpace(value)) && !strings.ContainsRune(value, '\x00')
}

func canonicalPositiveDecimal(value string, scale int) (string, error) {
	if value == "" {
		value = "1"
	}
	canonical, err := canonicalNonNegativeDecimal(value, scale)
	if err != nil {
		return "", err
	}
	if canonical == "0" {
		return "", fmt.Errorf("must be positive")
	}
	return canonical, nil
}

func canonicalNonNegativeDecimal(value string, scale int) (string, error) {
	if value == "" {
		return "", fmt.Errorf("must not be empty")
	}
	if !decimalPattern.MatchString(value) {
		return "", fmt.Errorf("must be a canonical non-negative decimal")
	}
	parts := strings.SplitN(value, ".", 2)
	if len(parts) == 2 {
		if len(parts[1]) > scale {
			return "", fmt.Errorf("supports at most %d fractional digits", scale)
		}
		parts[1] = strings.TrimRight(parts[1], "0")
		if parts[1] == "" {
			return parts[0], nil
		}
		return parts[0] + "." + parts[1], nil
	}
	return parts[0], nil
}

func matcherKey(m Matcher) string {
	if m.Claim != nil {
		b, _ := json.Marshal(m.Claim)
		return "claim:" + string(b)
	}
	if m.ExactPath != "" {
		return "exact:" + m.ExactPath
	}
	return "prefix:" + m.PathPrefix
}
