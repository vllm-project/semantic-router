package config

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// v03AutoMixFields is the decision-level algorithm.automix contract published
// for the v0.3 release (#2946). AutoMix has since been removed from the router
// (#5886), but the versioned v0.3 pages still document it, so they are pinned
// to this literal set.
var v03AutoMixFields = []string{
	"cost_aware_routing",
	"cost_quality_tradeoff",
	"discount_factor",
	"max_escalations",
	"use_logprob_verification",
	"verification_threshold",
}

// autoMixInternalOnlyFields are research settings of the v0.3 AutoMix selector
// that were never part of the public algorithm.automix contract. In v0.3,
// request-time verification and escalation belong to the confidence Looper.
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
	path string
	// parametersHeading names the heading that owns the parameter table.
	parametersHeading string
}

func autoMixDocInventories() []autoMixDocInventory {
	return []autoMixDocInventory{
		{
			path:              repoRel("website", "versioned_docs", "version-v0.3", "tutorials", "algorithm", "selection", "automix.md"),
			parametersHeading: "### Parameters",
		},
		{
			path:              repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "version-v0.3", "tutorials", "algorithm", "selection", "automix.md"),
			parametersHeading: "### 参数",
		},
	}
}

func TestAutoMixDocsMatchPublishedContract(t *testing.T) {
	root := repoRootFromTestFile(t)
	for _, inventory := range autoMixDocInventories() {
		t.Run(inventory.path, func(t *testing.T) {
			content := readRepoFile(t, root, inventory.path)
			problems := autoMixYAMLExampleProblems(content, v03AutoMixFields)
			problems = append(problems, autoMixParameterTableProblems(content, inventory.parametersHeading, v03AutoMixFields)...)
			for _, problem := range problems {
				t.Error(problem)
			}
			for _, name := range autoMixInternalOnlyFields {
				if strings.Contains(content, name) {
					t.Errorf("page mentions internal-only AutoMix field %q", name)
				}
			}
		})
	}
}

func TestAutoMixDocsCheckerReportsMalformedExample(t *testing.T) {
	page := autoMixTestPage(v03AutoMixFields) + "\n```yaml\nalgorithm:\n  type: automix\n  \"automix\":\n    verification_threshold: [unterminated\n```\n"
	problems := autoMixYAMLExampleProblems(page, v03AutoMixFields)
	if !problemsContain(problems, "YAML example does not parse") {
		t.Fatalf("expected a parse error for the malformed second example, got %v", problems)
	}
}

func TestAutoMixDocsCheckerReportsUnknownTableSpellings(t *testing.T) {
	page := autoMixTestPage(append([]string{"verificationTemperature", "cost-lambda"}, v03AutoMixFields...))
	problems := autoMixParameterTableProblems(page, "### Parameters", v03AutoMixFields)
	if !problemsContain(problems, "extra [cost-lambda verificationTemperature]") {
		t.Fatalf("expected both unsupported spellings as extra fields, got %v", problems)
	}
}

// autoMixTestPage renders a minimal tutorial page with one complete
// six-field example and a parameter table listing tableFields.
func autoMixTestPage(tableFields []string) string {
	var page strings.Builder
	page.WriteString("## Configuration\n\n```yaml\nalgorithm:\n  type: automix\n  automix:\n")
	for _, field := range v03AutoMixFields {
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
