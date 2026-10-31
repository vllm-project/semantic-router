package config

import (
	"fmt"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// acceptedAutoMixFields is the decision-level AutoMix public contract accepted
// in #2946; selection.AutoMixConfig carries more research settings that stay
// internal.
var acceptedAutoMixFields = []string{
	"cost_aware_routing",
	"cost_quality_tradeoff",
	"discount_factor",
	"max_escalations",
	"use_logprob_verification",
	"verification_threshold",
}

// v03AutoMixFields is the field inventory published for the v0.3 release.
// Versioned pages keep describing their release, so they are pinned to this
// literal set instead of the live struct.
var v03AutoMixFields = []string{
	"cost_aware_routing",
	"cost_quality_tradeoff",
	"discount_factor",
	"max_escalations",
	"use_logprob_verification",
	"verification_threshold",
}

// autoMixInternalOnlyFields are selection.AutoMixConfig research settings that
// are not part of the public algorithm.automix contract. Request-time
// verification and escalation belong to the confidence Looper.
var autoMixInternalOnlyFields = []string{
	"belief_particles",
	"cost_lambda",
	"enable_cascade",
	"enable_self_verification",
	"use_pomdp_router",
	"verification_samples",
	"verification_temperature",
	"verifier_server_url",
}

type autoMixDocInventory struct {
	path   string
	fields []string
	// parametersHeading names the heading that owns the parameter table;
	// empty when the page documents its inventory only through YAML.
	parametersHeading string
	banInternalFields bool
}

func autoMixDocInventories(t *testing.T) []autoMixDocInventory {
	return []autoMixDocInventory{
		{
			path:              repoRel("website", "versioned_docs", "version-v0.3", "tutorials", "algorithm", "selection", "automix.md"),
			fields:            v03AutoMixFields,
			parametersHeading: "### Parameters",
			banInternalFields: true,
		},
		{
			path:              repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "version-v0.3", "tutorials", "algorithm", "selection", "automix.md"),
			fields:            v03AutoMixFields,
			parametersHeading: "### 参数",
			banInternalFields: true,
		},
		{
			path:   repoRel("website", "docs", "tutorials", "algorithm", "selection", "automix.md"),
			fields: autoMixYAMLTagNames(t),
		},
	}
}

func TestAutoMixPublicFieldsMatchAcceptedContract(t *testing.T) {
	if diff := diffFieldSets(acceptedAutoMixFields, autoMixYAMLTagNames(t)); diff != "" {
		t.Fatalf("AutoMixSelectionConfig drifted from the accepted six-field public contract (#2946): %s. "+
			"Changing it is a config-contract change: get a maintainer decision and update docs, fragments, and schema.", diff)
	}
}

func TestAutoMixDocsMatchPublishedContract(t *testing.T) {
	root := repoRootFromTestFile(t)
	for _, inventory := range autoMixDocInventories(t) {
		t.Run(inventory.path, func(t *testing.T) {
			content := readRepoFile(t, root, inventory.path)
			problems := autoMixYAMLExampleProblems(content, inventory.fields)
			if inventory.parametersHeading != "" {
				problems = append(problems, autoMixParameterTableProblems(content, inventory.parametersHeading, inventory.fields)...)
			}
			for _, problem := range problems {
				t.Error(problem)
			}
			if inventory.banInternalFields {
				for _, name := range autoMixInternalOnlyFields {
					if strings.Contains(content, name) {
						t.Errorf("page mentions internal-only AutoMix field %q", name)
					}
				}
			}
		})
	}
}

func TestAutoMixDocsCheckerReportsMalformedExample(t *testing.T) {
	page := autoMixTestPage(acceptedAutoMixFields) + "\n```yaml\nalgorithm:\n  type: automix\n  \"automix\":\n    verification_threshold: [unterminated\n```\n"
	problems := autoMixYAMLExampleProblems(page, acceptedAutoMixFields)
	if !problemsContain(problems, "YAML example does not parse") {
		t.Fatalf("expected a parse error for the malformed second example, got %v", problems)
	}
}

func TestAutoMixDocsCheckerReportsUnknownTableSpellings(t *testing.T) {
	page := autoMixTestPage(append([]string{"verificationTemperature", "cost-lambda"}, acceptedAutoMixFields...))
	problems := autoMixParameterTableProblems(page, "### Parameters", acceptedAutoMixFields)
	if !problemsContain(problems, "extra [cost-lambda verificationTemperature]") {
		t.Fatalf("expected both unsupported spellings as extra fields, got %v", problems)
	}
}

// autoMixTestPage renders a minimal tutorial page with one complete
// six-field example and a parameter table listing tableFields.
func autoMixTestPage(tableFields []string) string {
	var page strings.Builder
	page.WriteString("## Configuration\n\n```yaml\nalgorithm:\n  type: automix\n  automix:\n")
	for _, field := range acceptedAutoMixFields {
		page.WriteString("    " + field + ": 1\n")
	}
	page.WriteString("```\n\n### Parameters\n\n| Parameter | Type | Description |\n|-----------|------|-------------|\n")
	for _, field := range tableFields {
		page.WriteString("| `" + field + "` | float | x |\n")
	}
	return page.String()
}

func problemsContain(problems []string, substring string) bool {
	for _, problem := range problems {
		if strings.Contains(problem, substring) {
			return true
		}
	}
	return false
}

func TestAutoMixLoaderRejectsInternalOnlyFields(t *testing.T) {
	for _, name := range autoMixInternalOnlyFields {
		t.Run(name, func(t *testing.T) {
			_, err := ParseYAMLBytes([]byte(autoMixDecisionConfig("verification_threshold: 0.7\n" + name + ": true")))
			if err == nil {
				t.Fatalf("expected %q under algorithm.automix to be rejected", name)
			}
			want := fmt.Sprintf("unknown field %q in routing.decisions[0].algorithm.automix", name)
			if !strings.Contains(err.Error(), want) {
				t.Fatalf("error = %q, want it to contain %q", err.Error(), want)
			}
		})
	}
}

// The loader must keep explicit false and zero values; #2950 was rejected for
// zero-value defaulting that rewrote them. This covers parsed configuration
// only, not how the extproc builder wires the selector.
func TestAutoMixLoaderPreservesExplicitZeroValues(t *testing.T) {
	cfg, err := ParseYAMLBytes([]byte(autoMixDecisionConfig(strings.Join([]string{
		"verification_threshold: 0.8",
		"max_escalations: 3",
		"cost_aware_routing: false",
		"cost_quality_tradeoff: 0",
		"discount_factor: 0.9",
		"use_logprob_verification: false",
	}, "\n"))))
	if err != nil {
		t.Fatalf("parse six-field AutoMix config: %v", err)
	}
	decisions := cfg.AllRoutingDecisions()
	if len(decisions) != 1 || decisions[0].Algorithm == nil || decisions[0].Algorithm.AutoMix == nil {
		t.Fatalf("expected one decision with algorithm.automix, got %+v", decisions)
	}
	want := AutoMixSelectionConfig{
		VerificationThreshold:  0.8,
		MaxEscalations:         3,
		CostAwareRouting:       false,
		CostQualityTradeoff:    0,
		DiscountFactor:         0.9,
		UseLogprobVerification: false,
	}
	if got := *decisions[0].Algorithm.AutoMix; got != want {
		t.Fatalf("parsed AutoMix = %+v, want %+v", got, want)
	}
}

func TestAutoMixFragmentUsesPublicFields(t *testing.T) {
	root := repoRootFromTestFile(t)
	fragmentPath := repoRel("config", "fragments", "algorithm", "selection", "automix.yaml")
	keys, found, err := autoMixKeysFromYAML(readRepoFile(t, root, fragmentPath))
	if err != nil || !found {
		t.Fatalf("%s must contain algorithm.automix (found=%v, err=%v)", fragmentPath, found, err)
	}
	accepted := stringSet(acceptedAutoMixFields)
	for _, key := range keys {
		if !accepted[key] {
			t.Errorf("%s uses unsupported AutoMix field %q", fragmentPath, key)
		}
	}
}

func autoMixYAMLExampleProblems(content string, expected []string) []string {
	expectedSet := stringSet(expected)
	var problems []string
	examples, complete := 0, false
	for _, block := range extractFencedYAMLBlocks(content) {
		keys, found, err := autoMixKeysFromYAML(block)
		if err != nil {
			problems = append(problems, fmt.Sprintf("YAML example does not parse: %v", err))
			continue
		}
		if !found {
			continue
		}
		examples++
		for _, key := range keys {
			if !expectedSet[key] {
				problems = append(problems, fmt.Sprintf("YAML example uses unsupported AutoMix field %q", key))
			}
		}
		if diffFieldSets(expected, keys) == "" {
			complete = true
		}
	}
	if examples == 0 {
		problems = append(problems, "page has no algorithm.automix YAML example")
	} else if !complete {
		problems = append(problems, fmt.Sprintf("no YAML example lists the complete AutoMix field set %v", expected))
	}
	return problems
}

func autoMixParameterTableProblems(content string, heading string, expected []string) []string {
	fields, found := parameterTableFields(content, heading)
	if !found {
		return []string{fmt.Sprintf("page has no %q section", heading)}
	}
	if len(fields) == 0 {
		return []string{fmt.Sprintf("%q section has no parameter table", heading)}
	}
	if diff := diffFieldSets(expected, fields); diff != "" {
		return []string{fmt.Sprintf("parameter table drifted: %s", diff)}
	}
	return nil
}

func autoMixYAMLTagNames(t *testing.T) []string {
	return requiredYAMLFields(t, reflect.TypeOf(AutoMixSelectionConfig{}))
}

var fencedYAMLBlockPattern = regexp.MustCompile("(?ms)^```ya?ml[^\\n]*\\n(.*?)^```")

func extractFencedYAMLBlocks(markdown string) []string {
	matches := fencedYAMLBlockPattern.FindAllStringSubmatch(markdown, -1)
	blocks := make([]string, 0, len(matches))
	for _, match := range matches {
		blocks = append(blocks, match[1])
	}
	return blocks
}

// autoMixKeysFromYAML returns the keys under algorithm.automix, reporting
// found=false when the document has no such block.
func autoMixKeysFromYAML(document string) ([]string, bool, error) {
	var doc struct {
		Algorithm struct {
			AutoMix map[string]interface{} `yaml:"automix"`
		} `yaml:"algorithm"`
	}
	if err := yaml.Unmarshal([]byte(document), &doc); err != nil {
		return nil, false, err
	}
	if doc.Algorithm.AutoMix == nil {
		return nil, false, nil
	}
	keys := make([]string, 0, len(doc.Algorithm.AutoMix))
	for key := range doc.Algorithm.AutoMix {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys, true, nil
}

var tableDelimiterRowPattern = regexp.MustCompile(`^\|?(\s*:?-+:?\s*\|)*\s*:?-+:?\s*\|?$`)

// parameterTableFields collects the first-column value of every data row in
// the table inside the section opened by heading, stopping at the next heading
// of the same or a higher level so unrelated tables stay out of the inventory.
// Names are kept verbatim (minus backticks) so unknown spellings show up as
// drift instead of being filtered out.
func parameterTableFields(markdown string, heading string) ([]string, bool) {
	level := len(heading) - len(strings.TrimLeft(heading, "#"))
	var fields []string
	inSection, headerSeen := false, false
	for _, line := range strings.Split(markdown, "\n") {
		trimmed := strings.TrimSpace(line)
		if !inSection {
			inSection = trimmed == heading
			continue
		}
		if strings.HasPrefix(trimmed, "#") {
			if lineLevel := len(trimmed) - len(strings.TrimLeft(trimmed, "#")); lineLevel <= level {
				break
			}
		}
		if !strings.HasPrefix(trimmed, "|") {
			continue
		}
		if !headerSeen {
			headerSeen = true
			continue
		}
		if tableDelimiterRowPattern.MatchString(trimmed) {
			continue
		}
		cells := strings.Split(strings.Trim(trimmed, "|"), "|")
		fields = append(fields, strings.Trim(strings.TrimSpace(cells[0]), "`"))
	}
	return fields, inSection
}

func autoMixDecisionConfig(automixFields string) string {
	indented := "          " + strings.ReplaceAll(automixFields, "\n", "\n          ")
	return `version: v0.3
listeners: []
providers:
  models:
    - name: m1
      backend_refs: [{endpoint: "127.0.0.1:8000"}]
routing:
  decisions:
    - name: d1
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: m1}]
      algorithm:
        type: automix
        automix:
` + indented + "\n"
}

func diffFieldSets(expected []string, actual []string) string {
	expectedSet, actualSet := stringSet(expected), stringSet(actual)
	var missing, extra []string
	for name := range expectedSet {
		if !actualSet[name] {
			missing = append(missing, name)
		}
	}
	for name := range actualSet {
		if !expectedSet[name] {
			extra = append(extra, name)
		}
	}
	if len(missing) == 0 && len(extra) == 0 {
		return ""
	}
	sort.Strings(missing)
	sort.Strings(extra)
	return fmt.Sprintf("missing %v, extra %v", missing, extra)
}

func stringSet(values []string) map[string]bool {
	set := make(map[string]bool, len(values))
	for _, value := range values {
		set[value] = true
	}
	return set
}
