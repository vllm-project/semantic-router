package config

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"
)

// The vSR apps that take a listener's rules, and how a rule fails.
const (
	VSRAppGuard        = "guard"
	VSRAppRedact       = "redact"
	VSREnforceRequired = "required"
	VSREnforceAdvisory = "advisory"
	// MaxVSRRulesPerApp bounds the rules one plugin call asks the Router.
	MaxVSRRulesPerApp = 16
)

// VSRApps are the apps a listener binds rules to, in listing order.
var VSRApps = []string{VSRAppGuard, VSRAppRedact}

// VSRRules are rules of one recipe the vSR agent plugins pointed at the
// listener enforce: guard holds a shell command one of its rules matches,
// redact hides what one of its rules finds in a tool's output. The plugins
// learn each rule's name and description from the listing and ask the Router
// to evaluate it (POST /v1/systemone/rules); a rule's pattern or question
// never leaves the Router.
type VSRRules struct {
	Recipe string    `yaml:"recipe" json:"recipe" jsonschema:"required,minLength=1"`
	Guard  []VSRRule `yaml:"guard,omitempty" json:"guard,omitempty"`
	Redact []VSRRule `yaml:"redact,omitempty" json:"redact,omitempty"`
}

// VSRRule binds one rule of the recipe to an app. A required rule never steps
// aside: when it cannot be checked the command waits for the person and the
// output is withheld. An advisory rule, the default, then goes unchecked, as
// the plugins' own questions do.
type VSRRule struct {
	SignalRef `yaml:",inline"`
	Enforce   string `yaml:"enforce,omitempty" json:"enforce,omitempty" jsonschema:"enum=required,enum=advisory"`
}

// Required reports whether the rule fails closed.
func (r VSRRule) Required() bool { return r.Enforce == VSREnforceRequired }

// App returns the rules bound to an app, nil for one that takes none.
func (r *VSRRules) App(app string) []VSRRule {
	if r == nil {
		return nil
	}
	switch app {
	case VSRAppGuard:
		return r.Guard
	case VSRAppRedact:
		return r.Redact
	}
	return nil
}

// RulesScope is the configuration of the recipe the rules bind, nil when it
// names none.
func (c *RouterConfig) RulesScope(rules *VSRRules) *RouterConfig {
	if rules == nil {
		return nil
	}
	if len(c.Recipes) == 0 && RecipeName(rules.Recipe) == DefaultRecipeName {
		return c
	}
	recipe, ok := c.RecipeByName(RecipeName(rules.Recipe))
	if !ok {
		return nil
	}
	return c.ConfigForRecipe(recipe)
}

// validateVSRRules checks a listener's rules against the recipe they bind:
// the Router prepares the classifiers of a recipe an entrypoint reaches, and
// evaluates rules on those.
func validateVSRRules(cfg *RouterConfig, path string, rules *VSRRules) error {
	scope := cfg.RulesScope(rules)
	switch {
	case scope == nil:
		return CheckError(path+".recipe", CodeInvalidValue, fmt.Sprintf("%q names no recipe", rules.Recipe))
	case len(cfg.Recipes) > 0 && !cfg.IsRecipeReachableForRouting(RecipeName(rules.Recipe)):
		return CheckError(path+".recipe", CodeInvalidValue,
			fmt.Sprintf("recipe %q is not reached from an entrypoint, so the Router does not prepare its rules; map a model name to it", rules.Recipe))
	}
	for _, app := range VSRApps {
		bound := rules.App(app)
		if len(bound) > MaxVSRRulesPerApp {
			return CheckError(path+"."+app, CodeInvalidValue, fmt.Sprintf("binds %d rules, more than the %d one call evaluates", len(bound), MaxVSRRulesPerApp))
		}
		for i, rule := range bound {
			at := fmt.Sprintf("%s.%s[%d]", path, app, i)
			if slices.IndexFunc(bound, func(other VSRRule) bool { return other.Name == rule.Name }) != i {
				return CheckError(at, CodeInvalidValue, fmt.Sprintf("%q is bound twice; the plugins name a rule by its name alone", rule.Name))
			}
			if rule.Enforce != "" && rule.Enforce != VSREnforceRequired && rule.Enforce != VSREnforceAdvisory {
				return CheckError(at+".enforce", CodeInvalidValue, fmt.Sprintf("%q is not required or advisory", rule.Enforce))
			}
			check := scope.RuleAnswersYesNo
			if app == VSRAppRedact {
				check = scope.RuleFindsText
			}
			if err := check(rule.SignalRef); err != nil {
				return CheckError(at, CodeInvalidValue, err.Error())
			}
		}
	}
	return nil
}

// RuleAnswersYesNo checks that ref names a rule of this recipe that says
// whether a text matches it: any keyword rule, or a request-stage systemone
// noul question, or choice question with labels (one without matches only
// through a condition's label).
func (c *RouterConfig) RuleAnswersYesNo(ref SignalRef) error {
	switch ref.Type {
	case SignalTypeKeyword:
		if !slices.ContainsFunc(c.KeywordRules, func(rule KeywordRule) bool { return rule.Name == ref.Name }) {
			return fmt.Errorf("%q names no keyword rule of this recipe", ref.Name)
		}
	case SignalTypeSystemOne:
		rule := systemOneRuleByName(c.SystemOneRules, ref.Name)
		switch {
		case rule == nil:
			return fmt.Errorf("%q names no systemone rule of this recipe", ref.Name)
		case rule.Question.Type == SystemOneQuestionChoice && len(rule.Labels) == 0:
			return fmt.Errorf("systemone rule %q has no labels, so it never matches on its own; list the options that match", ref.Name)
		case rule.Question.Type != SystemOneQuestionNoul && rule.Question.Type != SystemOneQuestionChoice:
			return fmt.Errorf("systemone rule %q asks a %s question; guard takes a noul or choice question", ref.Name, rule.Question.Type)
		case rule.EffectiveStage() != SystemOneStageRequest:
			return fmt.Errorf("systemone rule %q reads the response; guard reads a command", ref.Name)
		}
	default:
		return fmt.Errorf("type %q is not a rule the plugins can ask; use keyword or systemone", ref.Type)
	}
	return nil
}

// VSRRuleView is how the listing names one rule to the plugins.
type VSRRuleView struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
}

// VSRRulesView is the listing's rules: per app, each rule's name, description
// and whether it is required, and a revision that changes with any of them,
// with a rule's definition, or with the recipe the rules bind.
type VSRRulesView struct {
	Revision string        `json:"revision"`
	Guard    []VSRRuleView `json:"guard"`
	Redact   []VSRRuleView `json:"redact"`
}

// VSRListing is the vsr profile as a listener's System One listing hands it to
// the plugins: the profile's settings and the view of its rules.
type VSRListing struct {
	VSRProfile
	Rules *VSRRulesView `json:"rules,omitempty"`
}

// VSRListingOf is profile as the listing publishes it.
func (c *RouterConfig) VSRListingOf(profile *VSRProfile) VSRListing {
	listed := *profile
	listed.Rules = nil
	scope := c.RulesScope(profile.Rules)
	if scope == nil {
		return VSRListing{VSRProfile: listed}
	}
	view := &VSRRulesView{Guard: []VSRRuleView{}, Redact: []VSRRuleView{}}
	hashed := map[string]any{"recipe": profile.Rules.Recipe}
	for _, app := range VSRApps {
		var bound []any
		for _, rule := range profile.Rules.App(app) {
			definition, description := scope.ruleDefinition(rule.SignalRef)
			bound = append(bound, map[string]any{"rule": rule, "definition": definition})
			entry := VSRRuleView{Name: rule.Name, Description: description, Required: rule.Required()}
			if app == VSRAppGuard {
				view.Guard = append(view.Guard, entry)
			} else {
				view.Redact = append(view.Redact, entry)
			}
		}
		hashed[app] = bound
	}
	encoded, _ := json.Marshal(hashed)
	sum := sha256.Sum256(encoded)
	view.Revision = hex.EncodeToString(sum[:6])
	return VSRListing{VSRProfile: listed, Rules: view}
}

// ruleDefinition is the rule ref names, for the revision, and its
// description.
func (c *RouterConfig) ruleDefinition(ref SignalRef) (any, string) {
	switch ref.Type {
	case SignalTypeKeyword:
		if index := slices.IndexFunc(c.KeywordRules, func(rule KeywordRule) bool { return rule.Name == ref.Name }); index >= 0 {
			return c.KeywordRules[index], c.KeywordRules[index].Description
		}
	case SignalTypeSystemOne:
		if rule := systemOneRuleByName(c.SystemOneRules, ref.Name); rule != nil {
			return *rule, rule.Description
		}
	}
	return nil, ""
}
