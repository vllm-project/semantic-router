package config

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

const (
	DecisionPluginMasking = "masking"

	// MaskingPlaceholderIndexToken is substituted with the per-request,
	// per-entity-type occurrence index. A template without it would collapse
	// distinct values onto one placeholder, so the validator requires it.
	MaskingPlaceholderIndexToken = "{index}"
)

// MaskingPluginConfig replaces personal data in the provider-bound request
// with placeholders. See issue #3566. Failure is always closed (D4): there is
// deliberately no fail-open option for a privacy control.
type MaskingPluginConfig struct {
	Enabled bool `json:"enabled" yaml:"enabled"`
	// Signal names the pii rule of the decision's recipe that finds
	// the personal data: its detector, its threshold, and its
	// pii_types_allowed, which are never masked. It may be left out when
	// Rules names what to mask.
	Signal string `json:"signal,omitempty" yaml:"signal,omitempty"`
	// Rules are rules of the recipe that find text, a keyword rule that
	// locates its matches or a systemone span rule, whose matches are masked
	// as well, each under its own name.
	Rules []SignalRef `json:"rules,omitempty" yaml:"rules,omitempty"`
	// EntityTypes is an INCLUDE list of the pii rule's labels: only these
	// types are masked. Empty means mask every type the rule counts. It does
	// not apply to Rules, which are always masked.
	EntityTypes []string `json:"entity_types,omitempty" yaml:"entity_types,omitempty"`
	// Placeholders maps an entity type, a pii label or the name of one of
	// Rules, to a template containing {index}. Types absent from the map use
	// the default [TYPE_n] shape.
	Placeholders map[string]string `json:"placeholders,omitempty" yaml:"placeholders,omitempty"`
}

// SignalRef names one signal rule of a recipe, as a decision condition does:
// names are unique only within a signal type.
type SignalRef struct {
	Type string `json:"type" yaml:"type" jsonschema:"required,enum=keyword,enum=systemone"`
	Name string `json:"name" yaml:"name" jsonschema:"required,minLength=1"`
}

// GetMaskingConfig returns the masking plugin configuration, or nil when the
// decision does not configure it.
func (d *Decision) GetMaskingConfig() *MaskingPluginConfig {
	return MaskingPluginOf(d.Plugins)
}

// MaskingPluginOf is the masking plugin among a request's plugins, nil
// without one.
func MaskingPluginOf(plugins []DecisionPlugin) *MaskingPluginConfig {
	return decodePlugin(plugins, DecisionPluginMasking, &MaskingPluginConfig{})
}

// EffectivePlaceholder returns the template for an entity type, defaulting to
// the same "[TYPE_n]" shape the classification API already emits, with a
// rule's name in capitals and anything but letters, digits and _ as _.
func (c *MaskingPluginConfig) EffectivePlaceholder(entityType string) string {
	if c != nil {
		if template, ok := c.Placeholders[entityType]; ok && template != "" {
			return template
		}
	}
	name := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' {
			return r - 'a' + 'A'
		}
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			return r
		}
		return '_'
	}, entityType)
	return "[" + name + "_" + MaskingPlaceholderIndexToken + "]"
}

// RuleNames are the names of the rules masked besides the pii rule.
func (c *MaskingPluginConfig) RuleNames() []string {
	names := make([]string, 0, len(c.Rules))
	for _, rule := range c.Rules {
		names = append(names, rule.Name)
	}
	return names
}

// MaskingEntityTypes are the labels the pii preset produces, which are the
// entity types masking can name.
func MaskingEntityTypes() []string {
	preset, _ := presets.Lookup(presets.PII, presets.StageRequest)
	return preset.OptionKeys()
}

func validateMaskingPlugin(at PluginAt, p *MaskingPluginConfig) error {
	if strings.TrimSpace(p.Signal) == "" && len(p.Rules) == 0 {
		return at.Errorf("signal or rules is required: name the pii rule whose detections are masked, or the rules whose matches are")
	}
	labels := MaskingEntityTypes()
	for _, entityType := range p.EntityTypes {
		if !slices.Contains(labels, entityType) {
			return at.Errorf("entity_types: %q is not a label the pii preset produces (%s)", entityType, strings.Join(labels, ", "))
		}
	}
	rules := p.RuleNames()
	for i, rule := range p.Rules {
		switch {
		case rule.Type != SignalTypeKeyword && rule.Type != SignalTypeSystemOne:
			return at.Errorf("rules[%d]: type %q does not find text; use keyword or systemone", i, rule.Type)
		case strings.TrimSpace(rule.Name) == "":
			return at.Errorf("rules[%d]: name is required", i)
		case slices.Contains(labels, rule.Name):
			return at.Errorf("rules[%d]: %q is a pii label; a rule is masked under its own name", i, rule.Name)
		case slices.Index(rules, rule.Name) != i:
			return at.Errorf("rules[%d]: %q is listed twice; each rule is masked under its own name", i, rule.Name)
		}
	}
	for entityType, template := range p.Placeholders {
		if !slices.Contains(labels, entityType) && !slices.Contains(rules, entityType) {
			return at.Errorf("placeholders: %q is neither a label the pii preset produces (%s) nor one of rules", entityType, strings.Join(labels, ", "))
		}
		// Without {index} two distinct values collapse onto one placeholder,
		// which the issue's determinism criterion forbids (#3566); with it
		// twice, [X_{index}{index}] at 1 reads as [X_{index}] at 11.
		if strings.Count(template, MaskingPlaceholderIndexToken) != 1 {
			return at.Errorf("placeholder for %q must contain %s exactly once", entityType, MaskingPlaceholderIndexToken)
		}
	}
	return validateMaskingPlaceholdersDistinct(at, p, labels)
}

// validateMaskingPlaceholdersDistinct refuses two entity types masking can
// emit whose templates can produce the same placeholder: each type counts its
// own indexes, so two values would read as one. Digits next to {index} join
// the index, so [X_{index}] and [X_1{index}] both produce [X_10].
func validateMaskingPlaceholdersDistinct(at PluginAt, p *MaskingPluginConfig, labels []string) error {
	var types []string
	if strings.TrimSpace(p.Signal) != "" {
		types = labels
		if len(p.EntityTypes) > 0 {
			types = p.EntityTypes
		}
	}
	owner := map[string]string{}
	for _, entityType := range append(slices.Clone(types), p.RuleNames()...) {
		template := p.EffectivePlaceholder(entityType)
		before, after, _ := strings.Cut(template, MaskingPlaceholderIndexToken)
		shape := strings.TrimRight(before, "0123456789") + MaskingPlaceholderIndexToken + strings.TrimLeft(after, "0123456789")
		if other, taken := owner[shape]; taken && other != entityType {
			pair := []string{other, entityType}
			sort.Strings(pair)
			return at.Errorf("placeholders: %q and %q can both produce the same placeholder (%s)", pair[0], pair[1], template)
		}
		owner[shape] = entityType
	}
	return nil
}

// validateMaskingSignals resolves every masking plugin of a recipe, its
// default plugins' and its decisions', to the pii rule and the rules it
// names.
func validateMaskingSignals(cfg *RouterConfig) error {
	if masking := MaskingPluginOf(cfg.DefaultPlugins); masking != nil {
		if err := cfg.resolveMasking(masking); err != nil {
			return fmt.Errorf("%s masking: %w", cfg.ScopePath("plugins"), err)
		}
	}
	for i := range cfg.Decisions {
		if masking := cfg.Decisions[i].GetMaskingConfig(); masking != nil {
			if err := cfg.resolveMasking(masking); err != nil {
				return fmt.Errorf("decision %q masking: %w", cfg.Decisions[i].Name, err)
			}
		}
	}
	return nil
}

// resolveMasking checks that what a masking plugin names is in this recipe
// and finds text.
func (c *RouterConfig) resolveMasking(masking *MaskingPluginConfig) error {
	if masking.Signal != "" && c.PIIRule(masking.Signal) == nil {
		return fmt.Errorf("signal %q names no pii rule of this recipe", masking.Signal)
	}
	for _, ref := range masking.Rules {
		if err := c.RuleFindsText(ref); err != nil {
			return fmt.Errorf("rules: %w", err)
		}
	}
	return nil
}

// RuleFindsText checks that ref names a rule of this recipe that can say
// where in a text it matched: a keyword rule that locates its matches
// (KeywordRuleFindsText), or a request-stage systemone span rule.
func (c *RouterConfig) RuleFindsText(ref SignalRef) error {
	switch ref.Type {
	case SignalTypeKeyword:
		index := slices.IndexFunc(c.KeywordRules, func(rule KeywordRule) bool { return rule.Name == ref.Name })
		if index < 0 {
			return fmt.Errorf("%q names no keyword rule of this recipe", ref.Name)
		}
		if err := KeywordRuleFindsText(c.KeywordRules[index]); err != nil {
			return fmt.Errorf("keyword rule %q does not find text: %w", ref.Name, err)
		}
	case SignalTypeSystemOne:
		rule := systemOneRuleByName(c.SystemOneRules, ref.Name)
		switch {
		case rule == nil:
			return fmt.Errorf("%q names no systemone rule of this recipe", ref.Name)
		case rule.Question.Type != SystemOneQuestionSpan:
			return fmt.Errorf("systemone rule %q asks a %s question; only a span question finds text", ref.Name, rule.Question.Type)
		case rule.EffectiveStage() != SystemOneStageRequest:
			return fmt.Errorf("systemone rule %q reads the response; masking reads the request", ref.Name)
		}
	default:
		return fmt.Errorf("type %q does not find text; use keyword or systemone", ref.Type)
	}
	return nil
}

// PIIRule returns the pii rule of this recipe named name.
func (c *RouterConfig) PIIRule(name string) *PIIRule {
	for i := range c.PIIRules {
		if c.PIIRules[i].Name == name {
			return &c.PIIRules[i]
		}
	}
	return nil
}

// MaskingSignals are the pii rules the enabled masking plugins of this
// recipe name, in decision order, then its default plugins'.
func (c *RouterConfig) MaskingSignals() []string {
	var names []string
	add := func(masking *MaskingPluginConfig) {
		if masking != nil && masking.Enabled && masking.Signal != "" && !slices.Contains(names, masking.Signal) {
			names = append(names, masking.Signal)
		}
	}
	for i := range c.Decisions {
		add(c.Decisions[i].GetMaskingConfig())
	}
	add(MaskingPluginOf(c.DefaultPlugins))
	return names
}
