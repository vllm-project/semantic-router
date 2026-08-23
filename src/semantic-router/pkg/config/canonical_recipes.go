package config

import (
	"fmt"
	"sort"
	"strings"
)

// AuthoringEntrypoint is one callable v0.4 virtual Model. The common form
// selects one Recipe directly; advanced authoring may instead provide bounded
// conditional rules. Generated identity exists only in routingsnapshot.
type AuthoringEntrypoint struct {
	Name        string                            `yaml:"name" json:"name"`
	Aliases     []string                          `yaml:"aliases,omitempty" json:"aliases,omitempty"`
	Recipe      string                            `yaml:"recipe,omitempty" json:"recipe,omitempty"`
	Assignments map[string]AuthoringAssignmentSet `yaml:"assignments,omitempty" json:"assignments,omitempty"`
	Rules       []AuthoringEntrypointRule         `yaml:"rules,omitempty" json:"rules,omitempty"`
}

type AuthoringEntrypointRule struct {
	Name        string                            `yaml:"name" json:"name"`
	Matches     []AuthoringEntrypointMatch        `yaml:"matches,omitempty" json:"matches,omitempty"`
	Recipe      string                            `yaml:"recipe" json:"recipe"`
	Assignments map[string]AuthoringAssignmentSet `yaml:"assignments" json:"assignments"`
}

// AuthoringEntrypointMatch is an intentionally bounded matcher union.
// Exactly one of Claim or Path must be present.
type AuthoringEntrypointMatch struct {
	Claim *AuthoringClaimMatch `yaml:"claim,omitempty" json:"claim,omitempty"`
	Path  *AuthoringPathMatch  `yaml:"path,omitempty" json:"path,omitempty"`
}

type AuthoringClaimMatch struct {
	Name  string      `yaml:"name" json:"name"`
	Exact interface{} `yaml:"exact" json:"exact"`
}

type AuthoringPathMatch struct {
	Exact  string `yaml:"exact,omitempty" json:"exact,omitempty"`
	Prefix string `yaml:"prefix,omitempty" json:"prefix,omitempty"`
}

// AuthoringAssignmentSet binds one readable Decision name to Models. Models
// at priority zero form the initial selection tier; fallback authorizes later
// contiguous tiers.
type AuthoringAssignmentSet struct {
	Models   []AuthoringModelAssignment `yaml:"models" json:"models"`
	Fallback *AuthoringFallbackPolicy   `yaml:"fallback,omitempty" json:"fallback,omitempty"`
}

type AuthoringFallbackPolicy struct {
	Strategy string   `yaml:"strategy" json:"strategy"`
	On       []string `yaml:"on" json:"on"`
}

type AuthoringModelAssignment struct {
	Model     string                        `yaml:"model" json:"model"`
	Priority  int                           `yaml:"priority,omitempty" json:"priority,omitempty"`
	Weight    string                        `yaml:"weight,omitempty" json:"weight,omitempty"`
	LoRAName  string                        `yaml:"lora,omitempty" json:"lora,omitempty"`
	Reasoning *AuthoringAssignmentReasoning `yaml:"reasoning,omitempty" json:"reasoning,omitempty"`
}

type AuthoringAssignmentReasoning struct {
	Enabled     bool   `yaml:"enabled" json:"enabled"`
	Effort      string `yaml:"effort,omitempty" json:"effort,omitempty"`
	Description string `yaml:"description,omitempty" json:"description,omitempty"`
}

// AuthoringRecipe is a reusable, Model-free routing source document.
// Publication identity and Decision IDs are generated only while compiling
// the immutable routingsnapshot.
type AuthoringRecipe struct {
	Name        string           `yaml:"name" json:"name"`
	Description string           `yaml:"description,omitempty" json:"description,omitempty"`
	Document    CanonicalRouting `yaml:"document" json:"document"`
}

func validateAuthoringRecipes(recipes []AuthoringRecipe) error {
	seenNames := make(map[RecipeName]struct{}, len(recipes))
	for index, recipe := range recipes {
		name := RecipeName(strings.TrimSpace(recipe.Name))
		if name == "" {
			return fmt.Errorf("recipes[%d].name cannot be empty", index)
		}
		if string(name) != recipe.Name {
			return fmt.Errorf("recipes[%s].name must not contain surrounding whitespace", name)
		}
		if _, exists := seenNames[name]; exists {
			return fmt.Errorf("recipes[%s]: duplicate recipe name", name)
		}
		seenNames[name] = struct{}{}
		if len(recipe.Document.Decisions) == 0 {
			return fmt.Errorf("recipes[%s].document.decisions must not be empty", name)
		}
		if err := validateDecisionNames("recipes["+recipe.Name+"].document", recipe.Document.Decisions); err != nil {
			return err
		}
		if err := validateRecipeDocumentModelFree(recipe.Document); err != nil {
			return fmt.Errorf("recipes[%s].document: %w", name, err)
		}
	}
	return nil
}

func validateDecisionNames(path string, decisions []Decision) error {
	seen := make(map[string]struct{}, len(decisions))
	for index, decision := range decisions {
		name := strings.TrimSpace(decision.Name)
		if name == "" || name != decision.Name {
			return fmt.Errorf("%s.decisions[%d].name must be non-empty without surrounding whitespace", path, index)
		}
		if _, duplicate := seen[name]; duplicate {
			return fmt.Errorf("%s: duplicate decision name %q", path, name)
		}
		seen[name] = struct{}{}
	}
	return nil
}

func validateAuthoringEntrypoints(
	entrypoints []AuthoringEntrypoint,
	recipes []AuthoringRecipe,
	modelsByName map[string]AuthoringModel,
) error {
	recipesByName := make(map[string]AuthoringRecipe, len(recipes))
	for _, recipe := range recipes {
		recipesByName[recipe.Name] = recipe
	}
	claimedAliases := make(map[string]struct{})
	for index, entrypoint := range entrypoints {
		path := fmt.Sprintf("entrypoints[%d]", index)
		name := strings.TrimSpace(entrypoint.Name)
		if name == "" || name != entrypoint.Name {
			return fmt.Errorf("%s.name must be non-empty without surrounding whitespace", path)
		}
		aliases := stableUniqueStrings(append([]string{name}, entrypoint.Aliases...))
		for _, alias := range aliases {
			if _, duplicate := claimedAliases[alias]; duplicate {
				return fmt.Errorf("%s: model name %q is already mapped by another entrypoint", path, alias)
			}
			claimedAliases[alias] = struct{}{}
			if model, exists := modelsByName[alias]; exists {
				return fmt.Errorf("%s: model name %q is already a configured model", path, model.Name)
			}
			for _, model := range modelsByName {
				if routingModelHasLoRA(model, alias) {
					return fmt.Errorf("%s: model name %q is already a configured LoRA adapter", path, alias)
				}
			}
		}
		rules, err := authoringEntrypointRules(entrypoint)
		if err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
		seenRules := make(map[string]struct{}, len(rules))
		for ruleIndex, rule := range rules {
			rulePath := fmt.Sprintf("%s.rules[%d]", path, ruleIndex)
			if strings.TrimSpace(rule.Name) == "" || strings.TrimSpace(rule.Name) != rule.Name {
				return fmt.Errorf("%s.name must be non-empty without surrounding whitespace", rulePath)
			}
			if _, duplicate := seenRules[rule.Name]; duplicate {
				return fmt.Errorf("%s repeats rule name %q", path, rule.Name)
			}
			seenRules[rule.Name] = struct{}{}
			recipe, found := recipesByName[strings.TrimSpace(rule.Recipe)]
			if !found {
				return fmt.Errorf("%s.recipe references unknown Recipe %q", rulePath, rule.Recipe)
			}
			if _, err := normalizeEntrypointMatches(rulePath+".matches", rule.Matches); err != nil {
				return err
			}
			if err := validateAuthoringAssignmentNames(rulePath+".assignments", rule.Assignments, recipe, modelsByName); err != nil {
				return err
			}
		}
	}
	return nil
}

func authoringEntrypointRules(entrypoint AuthoringEntrypoint) ([]AuthoringEntrypointRule, error) {
	if len(entrypoint.Rules) == 0 {
		if strings.TrimSpace(entrypoint.Recipe) == "" {
			return nil, fmt.Errorf("recipe cannot be empty")
		}
		return []AuthoringEntrypointRule{{
			Name: "default", Recipe: entrypoint.Recipe, Assignments: entrypoint.Assignments,
		}}, nil
	}
	if strings.TrimSpace(entrypoint.Recipe) != "" || len(entrypoint.Assignments) != 0 {
		return nil, fmt.Errorf("cannot mix recipe/assignments with conditional rules")
	}
	return append([]AuthoringEntrypointRule(nil), entrypoint.Rules...), nil
}

func validateAuthoringAssignmentNames(
	path string,
	assignments map[string]AuthoringAssignmentSet,
	recipe AuthoringRecipe,
	modelsByName map[string]AuthoringModel,
) error {
	decisions := make(map[string]Decision, len(recipe.Document.Decisions))
	for _, decision := range recipe.Document.Decisions {
		decisions[decision.Name] = decision
	}
	if len(assignments) != len(decisions) {
		return fmt.Errorf("%s must assign every decision in Recipe %q", path, recipe.Name)
	}
	for decisionName, set := range assignments {
		if _, found := decisions[decisionName]; !found {
			return fmt.Errorf("%s.%s references unknown Decision name", path, decisionName)
		}
		if len(set.Models) == 0 {
			return fmt.Errorf("%s.%s.models must contain at least one Model", path, decisionName)
		}
		for modelIndex, assignment := range set.Models {
			model, found := modelsByName[strings.TrimSpace(assignment.Model)]
			if !found {
				return fmt.Errorf("%s.%s.models[%d].model references unknown Model %q", path, decisionName, modelIndex, assignment.Model)
			}
			if assignment.LoRAName != "" && !routingModelHasLoRA(model, assignment.LoRAName) {
				return fmt.Errorf("%s.%s.models[%d]: Model %q does not declare LoRA %q", path, decisionName, modelIndex, model.Name, assignment.LoRAName)
			}
		}
	}
	return nil
}

func canonicalRecipesFromRouterConfig(cfg *RouterConfig) []AuthoringRecipe {
	if cfg == nil || len(cfg.Recipes) == 0 {
		return nil
	}
	recipes := make([]AuthoringRecipe, 0, len(cfg.Recipes))
	for _, recipe := range cfg.Recipes {
		decisions := copyDecisions(recipe.Profile.Decisions)
		for index := range decisions {
			decisions[index].ID = ""
			stripManagedRecipeModelSelection(&decisions[index])
		}
		recipes = append(recipes, AuthoringRecipe{
			Name: string(recipe.Name), Description: recipe.Description,
			Document: CanonicalRouting{
				Signals:     canonicalSignalsFromSignals(recipe.Profile.Signals),
				Projections: canonicalProjectionsFromProjections(recipe.Profile.Projections),
				Decisions:   decisions, Strategy: recipe.Profile.Strategy,
			},
		})
	}
	return recipes
}

func canonicalEntrypointsFromRouterConfig(cfg *RouterConfig) []AuthoringEntrypoint {
	if cfg == nil || len(cfg.Entrypoints) == 0 {
		return nil
	}
	recipesByID := make(map[string]*RoutingRecipe, len(cfg.Recipes))
	for index := range cfg.Recipes {
		recipesByID[cfg.Recipes[index].ID] = &cfg.Recipes[index]
	}
	entrypoints := make([]AuthoringEntrypoint, 0, len(cfg.Entrypoints))
	for _, entrypoint := range cfg.Entrypoints {
		entrypoints = append(entrypoints, authoringEntrypointFromRuntime(entrypoint, recipesByID))
	}
	return entrypoints
}

func sortedAssignmentDecisionIDs(assignments map[string]RoutingAssignmentSet) []string {
	ids := make([]string, 0, len(assignments))
	for id := range assignments {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}
