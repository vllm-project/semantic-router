package config

import (
	"fmt"
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

const (
	DecisionPluginMasking = "masking"

	// MaskingPlaceholderIndexToken is substituted with the per-request,
	// per-entity-type occurrence index. A template without it would collapse
	// distinct values onto one placeholder, so the validator requires it.
	MaskingPlaceholderIndexToken = "{index}"

	MaskingIndexRegistry = "registry"
	MaskingIndexSequence = "sequence"
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
	// Index is how placeholders are numbered. registry keeps one index per
	// value for good in IdentityNamespace, so a value has the same
	// placeholder in every request and session. sequence numbers values in
	// the order one request meets them. Left out, it is registry when
	// IdentityNamespace is set and sequence otherwise.
	Index string `json:"index,omitempty" yaml:"index,omitempty" jsonschema:"enum=registry,enum=sequence"`
	// IdentityNamespace names a namespace of global.services.masking_identity.
	IdentityNamespace string `json:"identity_namespace,omitempty" yaml:"identity_namespace,omitempty"`
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

// RegistryMode reports whether placeholders come from an identity namespace.
func (c *MaskingPluginConfig) RegistryMode() bool {
	return c != nil && (c.Index == MaskingIndexRegistry || c.Index == "" && c.IdentityNamespace != "")
}

// CanonicalType is the type a registry keeps indexes under, which tells a pii
// label from a rule of the same name.
func (c *MaskingPluginConfig) CanonicalType(entityType string) string {
	for _, rule := range c.Rules {
		if rule.Name == entityType {
			return "rule:" + rule.Type + ":" + rule.Name
		}
	}
	return "pii:" + entityType
}

func validateMaskingPlugin(at PluginAt, p *MaskingPluginConfig) error {
	if p.Index != "" && p.Index != MaskingIndexRegistry && p.Index != MaskingIndexSequence {
		return at.Errorf("index must be %s or %s", MaskingIndexRegistry, MaskingIndexSequence)
	}
	if p.RegistryMode() && p.IdentityNamespace == "" {
		return at.Errorf("index: %s needs identity_namespace", MaskingIndexRegistry)
	}
	if p.Index == MaskingIndexSequence && p.IdentityNamespace != "" {
		return at.Errorf("identity_namespace applies only to index: %s", MaskingIndexRegistry)
	}
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
		// The index must end where the template says it does. A digit next to
		// {index} joins it, [X_{index}0] at 1 writing [X_10], which reads as
		// [X_{index}] at 10, and an index at the template's edge runs into a
		// digit the text has beside the value: #{index} for Bob in "Bob5"
		// writes #15.
		before, after, _ := strings.Cut(template, MaskingPlaceholderIndexToken)
		last, _ := utf8.DecodeLastRuneInString(before)
		first, _ := utf8.DecodeRuneInString(after)
		if before == "" || after == "" || unicode.IsDigit(last) || unicode.IsDigit(first) {
			return at.Errorf("placeholder for %q must put a character other than a digit right before and right after %s, as in #%s#", entityType, MaskingPlaceholderIndexToken, MaskingPlaceholderIndexToken)
		}
	}
	return validateMaskingPlaceholdersDistinct(at, p, labels)
}

// validateMaskingPlaceholdersDistinct refuses identical templates for different
// entity types. In sequence mode different templates can still intersect, and
// the allocator skips already-issued strings when their outputs collide. In
// registry mode a value keeps its index for good, so nothing can be skipped:
// intersecting templates are refused too.
func validateMaskingPlaceholdersDistinct(at PluginAt, p *MaskingPluginConfig, labels []string) error {
	var types []string
	if strings.TrimSpace(p.Signal) != "" {
		types = labels
		if len(p.EntityTypes) > 0 {
			types = p.EntityTypes
		}
	}
	owner := map[string]string{}
	var seen []string
	for _, entityType := range append(slices.Clone(types), p.RuleNames()...) {
		template := p.EffectivePlaceholder(entityType)
		if other, taken := owner[template]; taken && other != entityType {
			pair := []string{other, entityType}
			sort.Strings(pair)
			return at.Errorf("placeholders: %q and %q can both produce the same placeholder (%s)", pair[0], pair[1], template)
		}
		if _, taken := owner[template]; taken {
			continue
		}
		if p.RegistryMode() {
			for _, other := range seen {
				if written, ok := maskingTemplatesIntersect(other, template); ok {
					pair := []string{owner[other], entityType}
					sort.Strings(pair)
					return at.Errorf("placeholders: %q and %q can both produce %s, and index: %s keeps each value's index for good, so their templates must not intersect", pair[0], pair[1], written, MaskingIndexRegistry)
				}
			}
		}
		owner[template] = entityType
		seen = append(seen, template)
	}
	return nil
}

// maskingTemplatesIntersect reports a placeholder that two different
// templates both write at canonical decimal indexes. Validation keeps a
// character other than a digit on each side of {index}, so the template with
// the shorter prefix decides the only index that can collide: the digits that
// open the rest of the longer prefix.
func maskingTemplatesIntersect(a, b string) (string, bool) {
	aPrefix, aSuffix, _ := strings.Cut(a, MaskingPlaceholderIndexToken)
	bPrefix, bSuffix, _ := strings.Cut(b, MaskingPlaceholderIndexToken)
	if len(aPrefix) > len(bPrefix) {
		aPrefix, aSuffix, bPrefix, bSuffix = bPrefix, bSuffix, aPrefix, aSuffix
	}
	rest, ok := strings.CutPrefix(bPrefix, aPrefix)
	if !ok || rest == "" {
		return "", false
	}
	index := rest[:len(rest)-len(strings.TrimLeft(rest, "0123456789"))]
	if !canonicalMaskingIndex(index) {
		return "", false
	}
	written := aPrefix + index + aSuffix
	middle, ok := strings.CutPrefix(written, bPrefix)
	if !ok {
		return "", false
	}
	other, ok := strings.CutSuffix(middle, bSuffix)
	return written, ok && canonicalMaskingIndex(other)
}

// canonicalMaskingIndex reports whether s is an index as the allocator writes
// it: decimal digits with no leading zero.
func canonicalMaskingIndex(s string) bool {
	if s == "" || len(s) > 1 && s[0] == '0' {
		return false
	}
	return strings.Trim(s, "0123456789") == ""
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
	if err := c.resolveMaskingIdentity(masking); err != nil {
		return err
	}
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
