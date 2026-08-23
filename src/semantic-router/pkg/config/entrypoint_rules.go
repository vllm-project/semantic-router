package config

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// PrepareEntrypointRecipes compiles immutable per-rule routing views. Public
// input has exactly one shape: Rules with complete Actions.
func (c *RouterConfig) PrepareEntrypointRecipes() error {
	if c == nil {
		return nil
	}
	for i := range c.Entrypoints {
		if err := c.Entrypoints[i].prepareRules(c); err != nil {
			return fmt.Errorf("entrypoints[%d] (%s): %w", i, strings.Join(c.Entrypoints[i].ModelNames, ", "), err)
		}
	}
	return nil
}

func (e *EntrypointMapping) prepareRules(cfg *RouterConfig) error {
	if e == nil {
		return fmt.Errorf("entrypoint is required")
	}
	if len(e.Rules) == 0 {
		return fmt.Errorf("rules must contain at least one complete action")
	}
	e.Recipe = ""
	e.derivedRecipe = nil
	defaultRules := 0
	seenMatcherActions := make(map[string]string, len(e.Rules))
	for i := range e.Rules {
		rule := &e.Rules[i]
		base, ok := cfg.RecipeByID(rule.Action.RecipeID)
		if !ok || base.Revision != rule.Action.RecipeRevision {
			return fmt.Errorf("rule %q references unavailable recipe revision %s@%d", rule.ID, rule.Action.RecipeID, rule.Action.RecipeRevision)
		}
		derived, err := deriveEntrypointRecipe(cfg, base, e.ID, rule.ID, rule.Action.Assignments)
		if err != nil {
			return fmt.Errorf("rule %q: %w", rule.ID, err)
		}
		rule.derivedRecipe = derived
		if len(rule.Matches) == 0 {
			defaultRules++
			e.Recipe = base.Name
			e.derivedRecipe = derived
		}
		matcherBytes, _ := json.Marshal(rule.Matches)
		actionBytes, _ := json.Marshal(rule.Action)
		key := string(matcherBytes)
		if previous, exists := seenMatcherActions[key]; exists && previous != string(actionBytes) {
			return fmt.Errorf("rules with identical matchers have different actions")
		}
		seenMatcherActions[key] = string(actionBytes)
	}
	if defaultRules > 1 {
		return fmt.Errorf("at most one rule may have no matches")
	}
	return validateEntrypointRuleAmbiguity(e.Rules)
}

type EntrypointResolveOutcome string

const (
	EntrypointResolveMatched        EntrypointResolveOutcome = "matched"
	EntrypointResolveClaimedNoMatch EntrypointResolveOutcome = "claimed_no_match"
	EntrypointResolveUnclaimed      EntrypointResolveOutcome = "unclaimed"
)

type EntrypointResolution struct {
	Outcome    EntrypointResolveOutcome
	Entrypoint *EntrypointMapping
	Rule       *EntrypointRule
	Recipe     *RoutingRecipe
}

// ResolveEntrypoint is the single resolver used by discovery and invocation.
// It never falls through from a claimed alias to a concrete Model.
func (c *RouterConfig) ResolveEntrypoint(alias, path string, claims map[string]EntrypointClaimValue) (EntrypointResolution, error) {
	if c == nil {
		return EntrypointResolution{}, fmt.Errorf("router config is required")
	}
	for entrypointIndex := range c.Entrypoints {
		entrypoint := &c.Entrypoints[entrypointIndex]
		if !containsString(entrypoint.ModelNames, strings.TrimSpace(alias)) {
			continue
		}
		var selected *EntrypointRule
		var selectedSpecificity entrypointRuleSpecificity
		for ruleIndex := range entrypoint.Rules {
			rule := &entrypoint.Rules[ruleIndex]
			matched, specificity := entrypointRuleMatches(*rule, path, claims)
			if !matched {
				continue
			}
			if selected == nil || compareEntrypointSpecificity(specificity, selectedSpecificity) > 0 {
				selected = rule
				selectedSpecificity = specificity
				continue
			}
			if compareEntrypointSpecificity(specificity, selectedSpecificity) == 0 && selected.ID != rule.ID {
				return EntrypointResolution{}, fmt.Errorf("ambiguous routing rules in compiled entrypoint %s", entrypoint.ID)
			}
		}
		if selected == nil {
			return EntrypointResolution{Outcome: EntrypointResolveClaimedNoMatch, Entrypoint: entrypoint}, nil
		}
		return EntrypointResolution{Outcome: EntrypointResolveMatched, Entrypoint: entrypoint, Rule: selected, Recipe: selected.derivedRecipe}, nil
	}
	return EntrypointResolution{Outcome: EntrypointResolveUnclaimed}, nil
}

type entrypointRuleSpecificity struct {
	claims    int
	exactPath int
	prefixLen int
}

func entrypointRuleMatches(rule EntrypointRule, path string, claims map[string]EntrypointClaimValue) (bool, entrypointRuleSpecificity) {
	var specificity entrypointRuleSpecificity
	for _, matcher := range rule.Matches {
		switch {
		case matcher.Claim != nil:
			actual, exists := claims[matcher.Claim.Name]
			if !exists || actual != matcher.Claim.Value {
				return false, entrypointRuleSpecificity{}
			}
			specificity.claims++
		case matcher.Path != nil && matcher.Path.Exact != "":
			if path != matcher.Path.Exact {
				return false, entrypointRuleSpecificity{}
			}
			specificity.exactPath = 1
		case matcher.Path != nil:
			if !segmentAwarePathPrefix(path, matcher.Path.Prefix) {
				return false, entrypointRuleSpecificity{}
			}
			specificity.prefixLen = len(matcher.Path.Prefix)
		}
	}
	return true, specificity
}

func compareEntrypointSpecificity(left, right entrypointRuleSpecificity) int {
	if left.claims != right.claims {
		if left.claims > right.claims {
			return 1
		}
		return -1
	}
	if left.exactPath != right.exactPath {
		if left.exactPath > right.exactPath {
			return 1
		}
		return -1
	}
	if left.prefixLen != right.prefixLen {
		if left.prefixLen > right.prefixLen {
			return 1
		}
		return -1
	}
	return 0
}

func segmentAwarePathPrefix(path, prefix string) bool {
	if path == prefix {
		return true
	}
	if prefix == "/" {
		return strings.HasPrefix(path, "/")
	}
	return strings.HasPrefix(path, strings.TrimSuffix(prefix, "/")+"/")
}

func validateEntrypointRuleAmbiguity(rules []EntrypointRule) error {
	for i := range rules {
		for j := i + 1; j < len(rules); j++ {
			leftSpecificity := specificityForMatchers(rules[i].Matches)
			rightSpecificity := specificityForMatchers(rules[j].Matches)
			if compareEntrypointSpecificity(leftSpecificity, rightSpecificity) != 0 || !entrypointMatchersCanOverlap(rules[i].Matches, rules[j].Matches) {
				continue
			}
			leftAction, _ := json.Marshal(rules[i].Action)
			rightAction, _ := json.Marshal(rules[j].Action)
			if string(leftAction) != string(rightAction) {
				return fmt.Errorf("equally specific rules %q and %q can match the same request with different actions", rules[i].ID, rules[j].ID)
			}
		}
	}
	return nil
}

func specificityForMatchers(matchers []EntrypointMatch) entrypointRuleSpecificity {
	var result entrypointRuleSpecificity
	for _, matcher := range matchers {
		if matcher.Claim != nil {
			result.claims++
		}
		if matcher.Path != nil && matcher.Path.Exact != "" {
			result.exactPath = 1
		}
		if matcher.Path != nil && matcher.Path.Prefix != "" {
			result.prefixLen = len(matcher.Path.Prefix)
		}
	}
	return result
}

func entrypointMatchersCanOverlap(left, right []EntrypointMatch) bool {
	leftClaims := claimMatchersByName(left)
	rightClaims := claimMatchersByName(right)
	for name, value := range leftClaims {
		if other, exists := rightClaims[name]; exists && other != value {
			return false
		}
	}
	leftPath := pathMatcherFrom(left)
	rightPath := pathMatcherFrom(right)
	return pathsCanOverlap(leftPath, rightPath)
}

func claimMatchersByName(matchers []EntrypointMatch) map[string]EntrypointClaimValue {
	result := make(map[string]EntrypointClaimValue)
	for _, matcher := range matchers {
		if matcher.Claim != nil {
			result[matcher.Claim.Name] = matcher.Claim.Value
		}
	}
	return result
}

func pathMatcherFrom(matchers []EntrypointMatch) *EntrypointPathMatch {
	for _, matcher := range matchers {
		if matcher.Path != nil {
			return matcher.Path
		}
	}
	return nil
}

func pathsCanOverlap(left, right *EntrypointPathMatch) bool {
	if left == nil || right == nil {
		return true
	}
	if left.Exact != "" && right.Exact != "" {
		return left.Exact == right.Exact
	}
	if left.Exact != "" {
		return segmentAwarePathPrefix(left.Exact, right.Prefix)
	}
	if right.Exact != "" {
		return segmentAwarePathPrefix(right.Exact, left.Prefix)
	}
	return segmentAwarePathPrefix(left.Prefix, right.Prefix) || segmentAwarePathPrefix(right.Prefix, left.Prefix)
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func (c *RouterConfig) RecipeByID(id string) (*RoutingRecipe, bool) {
	if c == nil || strings.TrimSpace(id) == "" {
		return nil, false
	}
	for i := range c.Recipes {
		if c.Recipes[i].ID == id {
			return &c.Recipes[i], true
		}
	}
	return nil, false
}

func normalizeEntrypointMatches(path string, input []AuthoringEntrypointMatch) ([]EntrypointMatch, error) {
	result := make([]EntrypointMatch, 0, len(input))
	seenClaims := make(map[string]struct{})
	pathMatchers := 0
	for i, matcher := range input {
		itemPath := fmt.Sprintf("%s[%d]", path, i)
		if (matcher.Claim == nil) == (matcher.Path == nil) {
			return nil, fmt.Errorf("%s must set exactly one of claim or path", itemPath)
		}
		if matcher.Claim != nil {
			name := strings.TrimSpace(matcher.Claim.Name)
			if name == "" || name != matcher.Claim.Name {
				return nil, fmt.Errorf("%s.claim.name must be non-empty without surrounding whitespace", itemPath)
			}
			if _, duplicate := seenClaims[name]; duplicate {
				return nil, fmt.Errorf("%s repeats claim %q", path, name)
			}
			seenClaims[name] = struct{}{}
			value, err := normalizeEntrypointClaimValue(matcher.Claim.Exact)
			if err != nil {
				return nil, fmt.Errorf("%s.claim.exact: %w", itemPath, err)
			}
			result = append(result, EntrypointMatch{Claim: &EntrypointClaimMatch{Name: name, Value: value}})
			continue
		}
		pathMatchers++
		if pathMatchers > 1 {
			return nil, fmt.Errorf("%s may contain at most one path matcher", path)
		}
		exact := strings.TrimSpace(matcher.Path.Exact)
		prefix := strings.TrimSpace(matcher.Path.Prefix)
		if (exact == "") == (prefix == "") {
			return nil, fmt.Errorf("%s.path must set exactly one of exact or prefix", itemPath)
		}
		value := exact
		if value == "" {
			value = prefix
		}
		if !strings.HasPrefix(value, "/") || strings.Contains(value, "?") {
			return nil, fmt.Errorf("%s.path must be an absolute path without a query", itemPath)
		}
		result = append(result, EntrypointMatch{Path: &EntrypointPathMatch{Exact: exact, Prefix: prefix}})
	}
	sort.Slice(result, func(i, j int) bool { return entrypointMatchKey(result[i]) < entrypointMatchKey(result[j]) })
	return result, nil
}

func normalizeEntrypointClaimValue(value interface{}) (EntrypointClaimValue, error) {
	switch typed := value.(type) {
	case string:
		return EntrypointClaimValue{Kind: "string", String: typed}, nil
	case bool:
		return EntrypointClaimValue{Kind: "boolean", Boolean: typed}, nil
	case int:
		return EntrypointClaimValue{Kind: "integer", Integer: int64(typed)}, nil
	case int64:
		return EntrypointClaimValue{Kind: "integer", Integer: typed}, nil
	case uint64:
		if typed > math.MaxInt64 {
			return EntrypointClaimValue{}, fmt.Errorf("integer exceeds int64")
		}
		return EntrypointClaimValue{Kind: "integer", Integer: int64(typed)}, nil
	default:
		return EntrypointClaimValue{}, fmt.Errorf("must be a string, boolean, or integer")
	}
}

func deriveEntrypointRecipe(cfg *RouterConfig, base *RoutingRecipe, entrypointID, ruleID string, assignments map[string]RoutingAssignmentSet) (*RoutingRecipe, error) {
	if cfg == nil || base == nil {
		return nil, fmt.Errorf("cannot derive an entrypoint recipe from a nil config or recipe")
	}
	derived := &RoutingRecipe{
		ID:          base.ID,
		Revision:    base.Revision,
		Name:        base.Name,
		Description: base.Description,
		Profile:     base.Profile,
	}
	derived.Profile.Decisions = cloneEntrypointDecisions(base.Profile.Decisions)
	decisionIndexes := make(map[string]int, len(derived.Profile.Decisions))
	for i := range derived.Profile.Decisions {
		decisionIndexes[derived.Profile.Decisions[i].ID] = i
	}
	if len(assignments) != len(decisionIndexes) {
		return nil, fmt.Errorf("action must assign every decision in recipe %s@%d", base.ID, base.Revision)
	}
	for _, decisionID := range sortedAssignmentDecisionIDs(assignments) {
		decisionIndex, ok := decisionIndexes[decisionID]
		if !ok {
			return nil, fmt.Errorf("assignments[%s] references unknown decision in recipe %s@%d", decisionID, base.ID, base.Revision)
		}
		assignmentSet := assignments[decisionID]
		primary := make([]RoutingModelAssignment, 0, len(assignmentSet.Models))
		for _, assignment := range assignmentSet.Models {
			if assignment.Priority == 0 {
				primary = append(primary, assignment)
			}
		}
		refs, err := assignmentModelRefs(cfg, decisionID, primary)
		if err != nil {
			return nil, err
		}
		if err := rebindEntrypointDecision(&derived.Profile.Decisions[decisionIndex], refs); err != nil {
			return nil, fmt.Errorf("assignments[%s]: %w", decisionID, err)
		}
	}
	derived.runtimeScope = entrypointRuntimeScope(base, entrypointID, ruleID, assignments)
	return derived, nil
}

func assignmentModelRefs(cfg *RouterConfig, decisionID string, assignments []RoutingModelAssignment) ([]ModelRef, error) {
	if len(assignments) == 0 {
		return nil, fmt.Errorf("assignments[%s] must contain at least one model", decisionID)
	}
	refs := make([]ModelRef, 0, len(assignments))
	for index, assignment := range assignments {
		params, exists := cfg.ModelConfig[assignment.ModelName]
		if !exists || params.ResourceID != assignment.ModelID || params.ResourceRevision != assignment.ModelRevision {
			return nil, fmt.Errorf("assignments[%s][%d] references unavailable model revision %s@%d", decisionID, index, assignment.ModelID, assignment.ModelRevision)
		}
		weight, err := strconv.ParseFloat(assignment.Weight, 64)
		if err != nil || weight <= 0 || math.IsInf(weight, 0) || math.IsNaN(weight) {
			return nil, fmt.Errorf("assignments[%s][%d].weight is invalid", decisionID, index)
		}
		ref := ModelRef{Model: assignment.ModelName, LoRAName: assignment.LoRAName, Weight: weight}
		if assignment.Reasoning != nil {
			enabled := assignment.Reasoning.Enabled
			ref.UseReasoning = &enabled
			ref.ReasoningEffort = assignment.Reasoning.Effort
			ref.ReasoningDescription = assignment.Reasoning.Description
		} else {
			enabled := false
			ref.UseReasoning = &enabled
		}
		refs = append(refs, ref)
	}
	return refs, nil
}

func entrypointRuntimeScope(base *RoutingRecipe, entrypointID, ruleID string, assignments map[string]RoutingAssignmentSet) RecipeName {
	payload := struct {
		RecipeID       string
		RecipeRevision int64
		EntrypointID   string
		RuleID         string
		Assignments    map[string]RoutingAssignmentSet
	}{base.ID, base.Revision, entrypointID, ruleID, assignments}
	encoded, _ := json.Marshal(payload)
	digest := sha256.Sum256(encoded)
	return RecipeName(fmt.Sprintf("entrypoint/%x", digest[:16]))
}

func rebindEntrypointDecision(decision *Decision, refs []ModelRef) error {
	decision.ModelRefs = cloneModelRefs(refs)
	models := modelNamesFromRefs(refs)
	primary := models[0]
	if algorithm := decision.Algorithm; algorithm != nil {
		if algorithm.Fusion != nil {
			algorithm.Fusion.Model = primary
			algorithm.Fusion.AnalysisModels = append([]string(nil), models...)
			algorithm.Fusion.AnalysisOverrides = filterFusionOverrides(algorithm.Fusion.AnalysisOverrides, models)
			if algorithm.Fusion.MinSuccessfulResponses > len(models) {
				return fmt.Errorf("algorithm.fusion.min_successful_responses=%d exceeds the assigned candidate count %d", algorithm.Fusion.MinSuccessfulResponses, len(models))
			}
			if algorithm.Fusion.Grounding != nil && algorithm.Fusion.Grounding.MinKeep > len(models) {
				return fmt.Errorf("algorithm.fusion.grounding.min_keep=%d exceeds the assigned candidate count %d", algorithm.Fusion.Grounding.MinKeep, len(models))
			}
		}
		if algorithm.Workflows != nil {
			algorithm.Workflows.Planner.Model = primary
			algorithm.Workflows.Final.Model = primary
			for i := range algorithm.Workflows.Roles {
				algorithm.Workflows.Roles[i].Models = append([]string(nil), models...)
			}
			if algorithm.Workflows.MinSuccessfulResponses > len(models) {
				return fmt.Errorf("algorithm.workflows.min_successful_responses=%d exceeds the assigned candidate count %d", algorithm.Workflows.MinSuccessfulResponses, len(models))
			}
		}
		if algorithm.ReMoM != nil {
			algorithm.ReMoM.SynthesisModel = primary
			if limit := maxEntrypointBindingInt(algorithm.ReMoM.BreadthSchedule); limit > 0 && algorithm.ReMoM.MinSuccessfulResponses > limit {
				return fmt.Errorf("algorithm.remom.min_successful_responses=%d exceeds every configured round breadth (maximum %d)", algorithm.ReMoM.MinSuccessfulResponses, limit)
			}
		}
		if algorithm.Prompt != nil {
			algorithm.Prompt.Model = primary
		}
	}
	for i := range decision.CandidateIterations {
		iteration := &decision.CandidateIterations[i]
		switch strings.TrimSpace(iteration.Source) {
		case "models":
			iteration.Models = cloneModelRefs(refs)
		case "decision.candidates":
			iteration.Models = nil
		default:
			iteration.Models = nil
		}
	}
	return nil
}

func authoringEntrypointFromRuntime(entrypoint EntrypointMapping, recipesByID map[string]*RoutingRecipe) AuthoringEntrypoint {
	rules := make([]AuthoringEntrypointRule, 0, len(entrypoint.Rules))
	for _, rule := range entrypoint.Rules {
		rules = append(rules, authoringEntrypointRuleFromRuntime(rule, recipesByID[rule.Action.RecipeID]))
	}
	result := AuthoringEntrypoint{
		Name:    entrypoint.Name,
		Aliases: entrypointAliases(entrypoint.Name, entrypoint.ModelNames), Rules: rules,
	}
	if len(rules) == 1 && len(rules[0].Matches) == 0 {
		result.Recipe = rules[0].Recipe
		result.Assignments = rules[0].Assignments
		result.Rules = nil
	}
	return result
}

func authoringEntrypointRuleFromRuntime(rule EntrypointRule, recipe *RoutingRecipe) AuthoringEntrypointRule {
	matches := make([]AuthoringEntrypointMatch, 0, len(rule.Matches))
	for _, match := range rule.Matches {
		matches = append(matches, authoringEntrypointMatchFromRuntime(match))
	}
	decisionNames := make(map[string]string)
	if recipe != nil {
		decisionNames = make(map[string]string, len(recipe.Profile.Decisions))
		for _, decision := range recipe.Profile.Decisions {
			decisionNames[decision.ID] = decision.Name
		}
	}
	assignments := make(map[string]AuthoringAssignmentSet, len(rule.Action.Assignments))
	for decisionID, runtimeSet := range rule.Action.Assignments {
		decisionName := decisionNames[decisionID]
		if decisionName == "" {
			decisionName = decisionID
		}
		canonicalSet := AuthoringAssignmentSet{Models: make([]AuthoringModelAssignment, 0, len(runtimeSet.Models))}
		for _, ref := range runtimeSet.Models {
			canonicalSet.Models = append(canonicalSet.Models, AuthoringModelAssignment{Model: ref.ModelName, Priority: ref.Priority, Weight: ref.Weight, LoRAName: ref.LoRAName, Reasoning: authoringAssignmentReasoningFromRuntime(ref.Reasoning)})
		}
		if runtimeSet.Fallback != nil {
			canonicalSet.Fallback = &AuthoringFallbackPolicy{Strategy: runtimeSet.Fallback.Strategy, On: append([]string(nil), runtimeSet.Fallback.On...)}
		}
		assignments[decisionName] = canonicalSet
	}
	return AuthoringEntrypointRule{
		Name: rule.Name, Matches: matches,
		Recipe: string(rule.Action.Recipe), Assignments: assignments,
	}
}

func authoringEntrypointMatchFromRuntime(match EntrypointMatch) AuthoringEntrypointMatch {
	if match.Claim != nil {
		var value interface{}
		switch match.Claim.Value.Kind {
		case "string":
			value = match.Claim.Value.String
		case "boolean":
			value = match.Claim.Value.Boolean
		case "integer":
			value = match.Claim.Value.Integer
		}
		return AuthoringEntrypointMatch{Claim: &AuthoringClaimMatch{Name: match.Claim.Name, Exact: value}}
	}
	return AuthoringEntrypointMatch{Path: &AuthoringPathMatch{Exact: match.Path.Exact, Prefix: match.Path.Prefix}}
}

func authoringAssignmentReasoningFromRuntime(input *RoutingAssignmentReasoning) *AuthoringAssignmentReasoning {
	if input == nil {
		return nil
	}
	return &AuthoringAssignmentReasoning{
		Enabled: input.Enabled, Effort: input.Effort, Description: input.Description,
	}
}

func entrypointMatchKey(match EntrypointMatch) string {
	encoded, _ := json.Marshal(match)
	return string(encoded)
}

func cloneEntrypointDecisions(input []Decision) []Decision {
	if len(input) == 0 {
		return nil
	}
	output := make([]Decision, len(input))
	for i := range input {
		output[i] = input[i]
		output[i].ModelRefs = cloneModelRefs(input[i].ModelRefs)
		output[i].Algorithm = cloneEntrypointAlgorithm(input[i].Algorithm)
		output[i].CandidateIterations = cloneCandidateIterations(input[i].CandidateIterations)
	}
	return output
}

func cloneEntrypointAlgorithm(input *AlgorithmConfig) *AlgorithmConfig {
	if input == nil {
		return nil
	}
	output := *input
	if input.Confidence != nil {
		value := *input.Confidence
		output.Confidence = &value
	}
	if input.Ratings != nil {
		value := *input.Ratings
		output.Ratings = &value
	}
	if input.ReMoM != nil {
		value := *input.ReMoM
		value.BreadthSchedule = append([]int(nil), input.ReMoM.BreadthSchedule...)
		output.ReMoM = &value
	}
	if input.Fusion != nil {
		value := *input.Fusion
		value.AnalysisModels = append([]string(nil), input.Fusion.AnalysisModels...)
		value.AnalysisOverrides = append([]FusionModelOverride(nil), input.Fusion.AnalysisOverrides...)
		if input.Fusion.Grounding != nil {
			grounding := *input.Fusion.Grounding
			value.Grounding = &grounding
		}
		output.Fusion = &value
	}
	if input.Workflows != nil {
		value := *input.Workflows
		value.Roles = make([]WorkflowRoleConfig, len(input.Workflows.Roles))
		for i := range input.Workflows.Roles {
			value.Roles[i] = input.Workflows.Roles[i]
			value.Roles[i].Models = append([]string(nil), input.Workflows.Roles[i].Models...)
			value.Roles[i].AccessList = append([]string(nil), input.Workflows.Roles[i].AccessList...)
		}
		output.Workflows = &value
	}
	if input.Prompt != nil {
		value := *input.Prompt
		output.Prompt = &value
	}
	return &output
}

func cloneCandidateIterations(input []CandidateIterationConfig) []CandidateIterationConfig {
	if len(input) == 0 {
		return nil
	}
	output := make([]CandidateIterationConfig, len(input))
	for i := range input {
		output[i] = input[i]
		output[i].Models = cloneModelRefs(input[i].Models)
		output[i].Outputs = append([]CandidateIterationOutputConfig(nil), input[i].Outputs...)
	}
	return output
}

func cloneModelRefs(input []ModelRef) []ModelRef {
	if len(input) == 0 {
		return nil
	}
	output := make([]ModelRef, len(input))
	for i := range input {
		output[i] = input[i]
		if input[i].UseReasoning != nil {
			value := *input[i].UseReasoning
			output[i].UseReasoning = &value
		}
	}
	return output
}

func modelNamesFromRefs(refs []ModelRef) []string {
	models := make([]string, 0, len(refs))
	for _, ref := range refs {
		models = append(models, ref.Model)
	}
	return models
}

func filterFusionOverrides(input []FusionModelOverride, allowed []string) []FusionModelOverride {
	if len(input) == 0 {
		return nil
	}
	set := make(map[string]struct{}, len(allowed))
	for _, model := range allowed {
		set[model] = struct{}{}
	}
	output := make([]FusionModelOverride, 0, len(input))
	for _, override := range input {
		if _, ok := set[override.Model]; ok {
			output = append(output, override)
		}
	}
	return output
}

func maxEntrypointBindingInt(values []int) int {
	maximum := 0
	for _, value := range values {
		if value > maximum {
			maximum = value
		}
	}
	return maximum
}
