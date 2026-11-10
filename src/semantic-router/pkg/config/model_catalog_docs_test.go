package config

import (
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The examples on the model catalog pages are the configurations these tests
// load. Each titled YAML block, merged onto testdata/model_catalog_docs/base.yaml
// when it is a fragment, must load and resolve as its expectation below
// says; the earlier form of each migration pair must be refused, and migrate
// must rewrite it to resolve as the new one; the zh-Hans pages carry the same
// blocks.

const modelCatalogDocsBase = "testdata/model_catalog_docs/base.yaml"

var modelCatalogDocPages = []string{
	"installation/choose-a-model-for-a-signal.md",
	"installation/model-catalog.md",
	"installation/model-resource-migration.md",
}

var titledYAMLBlock = regexp.MustCompile("(?ms)^```ya?ml title=\"([^\"]+)\"\\n(.*?)^```")

// allSignals are the signal types every expectation that names "*" means.
var allSignals = []string{
	"domain", "fact_check", "user_feedback", "jailbreak", "safety", "pii", "hallucination", "refusal", "turn_relation", "complexity", "modality",
}

// docExampleResolutions is what each documented example resolves to: an
// application signal type, a rule (<type>.<rule>, systemone.<rule>),
// selector.<decision>, embedding or multimodal, optionally prefixed with
// recipe:<name>/, to the deployment that answers it. "*" sets every
// signal in allSignals.
var docExampleResolutions = map[string]map[string]string{
	"s1-defaults":         {"*": "primary", "embedding": "vela-embedding"},
	"s2-remote-systemone": {"*": "d3-remote", "embedding": "vela-embedding"},
	"s3-mixed":            {"*": "d3-remote", "pii": "local-vela"},
	"s4-type-and-rule-overrides": {
		"*": "kai", "domain": "d3-remote", "systemone.needs_reasoning": "kai", "systemone.support_team": "d3-remote",
	},
	"s4-recipe-scope": {"*": "kai", "recipe:support/domain": "d3-remote", "recipe:support/pii": "kai"},
	"s5-rule-level": {
		"domain.math": "primary", "domain.law": "kai", "safety.harmful": "d3-remote", "selector.pick-model": "kai", "domain": "primary",
	},
	"field-reference":      {"*": "primary", "pii": "primary", "embedding": "vela-embedding"},
	"r1-remote-embedding":  {"embedding": "remote-embed", "*": "primary"},
	"r3-remote-judge":      {"jailbreak": "d3-judge", "hallucination": "d3-judge", "domain": "primary"},
	"r4-systemone-gateway": {"*": "d3-gateway"},
	"r5-attached-runtime":  {"*": "decision-kai"},
	"r6-request-backends":  {"*": "primary"},
}

// docExampleWarnings are the warning codes an example must report.
var docExampleWarnings = map[string][]string{}

type docExample struct {
	page, title, text string
}

func docsRoot() string { return filepath.Join("..", "..", "..", "..", "website") }

func readDocExamples(t *testing.T, path string) []docExample {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var examples []docExample
	for _, match := range titledYAMLBlock.FindAllStringSubmatch(string(data), -1) {
		examples = append(examples, docExample{page: path, title: match[1], text: match[2]})
	}
	return examples
}

func englishDocExamples(t *testing.T) []docExample {
	t.Helper()
	var examples []docExample
	for _, page := range modelCatalogDocPages {
		examples = append(examples, readDocExamples(t, filepath.Join(docsRoot(), "docs", page))...)
	}
	return examples
}

// mergeYAML merges overlay onto base: maps merge key by key, anything else
// the overlay writes replaces the base's.
func mergeYAML(base, overlay map[string]interface{}) map[string]interface{} {
	merged := make(map[string]interface{}, len(base)+len(overlay))
	for key, value := range base {
		merged[key] = value
	}
	for key, value := range overlay {
		if inner, ok := value.(map[string]interface{}); ok {
			if existing, ok := merged[key].(map[string]interface{}); ok {
				merged[key] = mergeYAML(existing, inner)
				continue
			}
		}
		merged[key] = value
	}
	return merged
}

// catalogExampleDocument is an example merged onto the base.
func catalogExampleDocument(t *testing.T, example docExample) []byte {
	t.Helper()
	var base, overlay map[string]interface{}
	data, err := os.ReadFile(modelCatalogDocsBase)
	if err != nil {
		t.Fatal(err)
	}
	if err = yaml.Unmarshal(data, &base); err != nil {
		t.Fatal(err)
	}
	if err = yaml.Unmarshal([]byte(example.text), &overlay); err != nil {
		t.Fatalf("%s: not YAML: %v", example.title, err)
	}
	encoded, err := yaml.Marshal(mergeYAML(base, overlay))
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

// loadDocExample loads an example, merged onto the base.
func loadDocExample(t *testing.T, example docExample) (*RouterConfig, error) {
	t.Helper()
	return ParseYAMLBytesWithoutEnvExpansion(catalogExampleDocument(t, example))
}

// resolveDocKey is the deployment a documented key resolves to.
func resolveDocKey(cfg *RouterConfig, key string) string {
	scoped := cfg
	if rest, ok := strings.CutPrefix(key, "recipe:"); ok {
		name, signal, _ := strings.Cut(rest, "/")
		recipe := findRecipe(cfg.Recipes, RecipeName(name))
		if recipe == nil {
			return "no recipe " + name
		}
		scoped, key = cfg.ConfigForRecipe(recipe), signal
	}
	switch {
	case key == "embedding":
		return scoped.EmbeddingDeployment
	case strings.HasPrefix(key, "systemone."):
		for _, rule := range scoped.SystemOneRules {
			if "systemone."+rule.Name == key {
				return scoped.SystemOneRuleDeployment(rule)
			}
		}
		return "no rule " + key
	case knownBindingKey(key):
		name, _ := scoped.ApplicationTypeDeployment(key)
		return name
	case strings.Contains(key, ".") && knownBindingKey(strings.SplitN(key, ".", 2)[0]):
		for _, rule := range scoped.ApplicationRules() {
			if rule.Type+"."+rule.Rule.Name == key {
				return scoped.deploymentOrDecisionModel(rule.Rule.Deployment)
			}
		}
		return "no rule " + key
	case strings.HasPrefix(key, "selector."):
		for _, decision := range scoped.Decisions {
			if "selector."+decision.Name == key && decision.Algorithm != nil && decision.Algorithm.SystemOne != nil {
				return scoped.SystemOneSelectorDeployment(*decision.Algorithm.SystemOne)
			}
		}
		return "no selector " + key
	}
	resolved, ok, err := scoped.ResolveTask(key)
	switch {
	case err != nil:
		return "error: " + err.Error()
	case !ok:
		return "none"
	}
	return resolved.Name
}

func warningCodes(cfg *RouterConfig) map[string]int {
	codes := map[string]int{}
	for _, warning := range Warnings(cfg) {
		codes[warning.Code]++
	}
	return codes
}

func TestModelCatalogDocExamplesResolveAsDocumented(t *testing.T) {
	examples := englishDocExamples(t)
	seen := map[string]bool{}
	for _, example := range examples {
		if seen[example.title] {
			t.Errorf("%s: two examples are titled %q", example.page, example.title)
		}
		seen[example.title] = true
		if strings.HasPrefix(example.title, "migrate-") {
			continue
		}
		t.Run(example.title, func(t *testing.T) {
			want, ok := docExampleResolutions[example.title]
			if !ok {
				t.Fatalf("%s: example %q has no expected resolution in docExampleResolutions", example.page, example.title)
			}
			cfg, err := loadDocExample(t, example)
			if err != nil {
				t.Fatalf("%s: example %q does not load: %v", example.page, example.title, err)
			}
			codes := warningCodes(cfg)
			for _, code := range docExampleWarnings[example.title] {
				if codes[code] == 0 {
					t.Errorf("example %q reports no %s warning: %+v", example.title, code, Warnings(cfg))
				}
			}
			expected := map[string]string{}
			if every, ok := want["*"]; ok {
				for _, signal := range allSignals {
					expected[signal] = every
				}
			}
			for key, deployment := range want {
				if key != "*" {
					expected[key] = deployment
				}
			}
			for _, key := range sortedStringKeys(expected) {
				if got := resolveDocKey(cfg, key); got != expected[key] {
					t.Errorf("example %q: %s resolves to %q, documented as %q", example.title, key, got, expected[key])
				}
			}
		})
	}
	for title := range docExampleResolutions {
		if !seen[title] {
			t.Errorf("expected example %q is on no model catalog page", title)
		}
	}
}

// effectiveDeployments is what a configuration's signals resolve to, by
// deployment identity rather than name: the keys named in either of two
// configurations, the request backends and the admission gates.
func effectiveDeployments(cfg *RouterConfig, keys []string) map[string]string {
	identity := func(name string) string {
		deployment, ok := cfg.ModelDeployments[name]
		if !ok {
			return name
		}
		return resolvedIdentity(name, deployment)
	}
	effective := map[string]string{}
	for _, key := range keys {
		name := resolveDocKey(cfg, key)
		if name == "none" {
			effective[key] = "no model"
			continue
		}
		effective[key] = identity(name)
	}
	canonical := CanonicalConfigFromRouterConfig(cfg)
	providers, _ := yaml.Marshal(canonical.Providers)
	effective["providers"] = string(providers)
	admission, _ := yaml.Marshal(canonical.Global.ModelCatalog.Admission)
	effective["admission"] = string(admission)
	return effective
}

// The earlier form of each pair fails to load with a removed form's line, and
// what migrate writes for it resolves to the same deployments as the new
// form.
func TestModelResourceMigrationPairsMigrateToTheNewForm(t *testing.T) {
	examples := map[string]docExample{}
	for _, example := range readDocExamples(t, filepath.Join(docsRoot(), "docs", "installation", "model-resource-migration.md")) {
		examples[example.title] = example
	}
	pairs := 0
	for title, before := range examples {
		name, isBefore := strings.CutSuffix(title, "-before")
		if !isBefore {
			if _, isAfter := strings.CutSuffix(title, "-after"); !isAfter {
				t.Errorf("migration example %q is neither a -before nor an -after", title)
			}
			continue
		}
		after, ok := examples[name+"-after"]
		if !ok {
			t.Errorf("migration example %q has no %s-after", title, name)
			continue
		}
		pairs++
		t.Run(name, func(t *testing.T) {
			document := catalogExampleDocument(t, before)
			if _, err := ParseYAMLBytesWithoutEnvExpansion(document); err == nil || !strings.Contains(err.Error(), "run `vllm-sr config migrate`") {
				t.Fatalf("the earlier form loads, or fails without a migrate hint: %v", err)
			}
			migrated, _, err := MigrateConfig(document)
			if err != nil {
				t.Fatalf("migrate: %v", err)
			}
			old, err := ParseYAMLBytesWithoutEnvExpansion(migrated)
			if err != nil {
				t.Fatalf("the migrated form does not load: %v", err)
			}
			current, err := loadDocExample(t, after)
			if err != nil {
				t.Fatalf("the new form does not load: %v", err)
			}
			keys := append([]string{"domain", "jailbreak", "pii", "fact_check", "user_feedback", "hallucination", "embedding"}, ruleKeys(old)...)
			keys = append(keys, ruleKeys(current)...)
			was, now := effectiveDeployments(old, keys), effectiveDeployments(current, keys)
			for _, key := range sortedStringKeys(was) {
				if was[key] != now[key] {
					t.Errorf("%s: the earlier form resolves to %q, the new form to %q", key, was[key], now[key])
				}
			}
		})
	}
	if pairs == 0 {
		t.Fatal("the migration page has no before/after pairs")
	}
}

// The model runtime's migration page shows what `vllm-sr config migrate`
// writes for its legacy.yaml, in both languages.
func TestModelRuntimeMigrationExampleIsWhatConfigMigrateWrites(t *testing.T) {
	page := filepath.Join("model-runtime", "deploy", "migrate.md")
	for _, path := range []string{
		filepath.Join(docsRoot(), "docs", page),
		filepath.Join(docsRoot(), "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", page),
	} {
		examples := map[string]docExample{}
		for _, example := range readDocExamples(t, path) {
			examples[example.title] = example
		}
		legacy, documented := examples["legacy.yaml"], examples["legacy.migrated.yaml"]
		if legacy.text == "" || documented.text == "" {
			t.Errorf("%s has no legacy.yaml and legacy.migrated.yaml blocks", path)
			continue
		}
		if _, err := ParseYAMLBytesWithoutEnvExpansion([]byte(legacy.text)); err == nil || !strings.Contains(err.Error(), "run `vllm-sr config migrate`") {
			t.Errorf("%s: legacy.yaml loads, or fails without a migrate hint: %v", path, err)
		}
		migrated, _, err := MigrateConfig([]byte(legacy.text))
		if err != nil {
			t.Fatalf("%s: migrate: %v", path, err)
		}
		var got, want interface{}
		if err := yaml.Unmarshal(migrated, &got); err != nil {
			t.Fatal(err)
		}
		if err := yaml.Unmarshal([]byte(documented.text), &want); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: legacy.migrated.yaml is not what config migrate writes:\n%s", path, migrated)
		}
	}
}

func TestModelCatalogDocExamplesMatchTheirTranslations(t *testing.T) {
	for _, page := range modelCatalogDocPages {
		english := readDocExamples(t, filepath.Join(docsRoot(), "docs", page))
		path := filepath.Join(docsRoot(), "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", page)
		translated := map[string]string{}
		for _, example := range readDocExamples(t, path) {
			translated[example.title] = example.text
		}
		for _, example := range english {
			text, ok := translated[example.title]
			switch {
			case !ok:
				t.Errorf("%s has no example %q", path, example.title)
			case text != example.text:
				t.Errorf("%s: example %q differs from the English page", path, example.title)
			}
		}
	}
}

func ruleKeys(cfg *RouterConfig) []string {
	var keys []string
	for _, rule := range cfg.SystemOneRules {
		keys = append(keys, "systemone."+rule.Name)
	}
	for _, rule := range cfg.ApplicationRules() {
		keys = append(keys, rule.Type+"."+rule.Rule.Name)
	}
	return keys
}
