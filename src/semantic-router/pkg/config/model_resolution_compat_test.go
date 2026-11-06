package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// The golden files list, for each configuration, the model every active
// signal of every scope resolves to and the model_runtime deployments the
// Router starts. testdata/model_resolution/deprecated holds one fixture for
// each deprecated way to choose a model; its golden is what the same
// fixture resolved to before the fields were deprecated, except where the
// release notes say otherwise: a systemone decision model now answers the
// built-in signals (07), and a hazard part without a hazard deployment uses
// the pre-declared hazard deployment instead of failing to load under a
// systemone decision model (08). The maintained golden
// pins the shipped configurations. UPDATE_MODEL_RESOLUTION=1 rewrites both.

const (
	deprecatedFixtures   = "testdata/model_resolution/deprecated"
	deprecatedGolden     = "testdata/model_resolution/deprecated.golden.json"
	maintainedGolden     = "testdata/model_resolution/maintained.golden.json"
	maintainedRepoPrefix = "../../../../"
)

var maintainedConfigDirs = []string{
	"config",
	filepath.Join("e2e", "config"),
	filepath.Join("e2e", "profiles"),
	filepath.Join("deploy", "helm"),
	filepath.Join("deploy", "kubernetes"),
	filepath.Join("deploy", "operator", "config", "samples"),
}

// resolutionSummary is a configuration's effective models: an error, or each
// scope's active signals and the deployments it starts.
type resolutionSummary struct {
	Error    string            `json:"error,omitempty"`
	Signals  map[string]string `json:"signals,omitempty"`
	Started  []string          `json:"started,omitempty"`
	Warnings int               `json:"deprecated_fields,omitempty"`
}

func resolvedIdentity(name string, d ModelDeployment) string {
	d = d.WithDefaults()
	switch d.Provider {
	case ModelRuntimeProvider:
		if !d.Managed() {
			return fmt.Sprintf("model_runtime %s %s", d.Endpoint, d.ServedModel(name))
		}
		artifact := d.Artifact
		if path := d.LegacyModelPath(); path != "" && RegisteredModel(path) == nil {
			artifact = path
		}
		return strings.TrimSpace(fmt.Sprintf("model_runtime %s@%s %s replicas=%d", artifact, d.Revision, d.Device, len(d.Replicas)))
	case ClassifyProvider, OpenAIChatProvider, LegacyHTTPProvider:
		return strings.TrimSpace(fmt.Sprintf("remote %s %s", strings.TrimSuffix(strings.TrimRight(d.Endpoint, "/"), "/v1"), d.Model))
	}
	return strings.TrimSpace(fmt.Sprintf("%s %s %s", d.Provider, d.Endpoint, d.Model))
}

func summarizeResolution(cfg *RouterConfig) resolutionSummary {
	summary := resolutionSummary{Signals: map[string]string{}}
	for _, warning := range cfg.ModelResourceWarnings {
		if warning.Code == warningDeprecatedModelField {
			summary.Warnings++
		}
	}
	if _, err := compileModelBindings(cfg); err != nil {
		summary.Error = "plan: " + err.Error()
		return summary
	}
	for recipe, scoped := range cfg.consumerScopes() {
		for _, consumer := range scoped.scopeConsumers() {
			if !TaskConsumerInUse(scoped, recipe, consumer) {
				continue
			}
			key := string(recipe) + "/" + BindingKey(consumer)
			resolved, ok, err := scoped.ResolveTask(consumer)
			var deferred *DeferredModelError
			switch {
			case errors.As(err, &deferred):
				summary.Signals[key] = "fails when it runs: " + deferred.Path
			case err != nil:
				summary.Signals[key] = "error: " + err.Error()
			case !ok:
				summary.Signals[key] = "none"
			case resolved.Deployment.MissingLocalModel() != nil:
				summary.Signals[key] = "fails when it runs: " + resolved.Deployment.LegacyModelPath()
			default:
				summary.Signals[key] = resolvedIdentity(resolved.Name, resolved.Deployment)
			}
		}
		for _, rule := range scoped.DecisionRules {
			name := scoped.DecisionQuestionDeployment(rule)
			summary.Signals[string(recipe)+"/decision."+rule.Name] = resolvedIdentity(name, cfg.ModelDeployments[name])
		}
	}
	for name, deployment := range ModelRuntimeDeploymentsInUse(cfg) {
		summary.Started = append(summary.Started, resolvedIdentity(name, deployment))
	}
	sort.Strings(summary.Started)
	return summary
}

func summarizeFile(path string) resolutionSummary {
	data, err := os.ReadFile(path)
	if err != nil {
		return resolutionSummary{Error: err.Error()}
	}
	cfg, err := ParseYAMLBytesWithoutEnvExpansion(data)
	if err != nil {
		return resolutionSummary{Error: err.Error()}
	}
	return summarizeResolution(cfg)
}

func compareResolutionGolden(t *testing.T, golden string, got map[string]resolutionSummary) {
	t.Helper()
	if os.Getenv("UPDATE_MODEL_RESOLUTION") == "1" {
		encoded, err := json.MarshalIndent(got, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(golden, append(encoded, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	data, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("%v; run with UPDATE_MODEL_RESOLUTION=1 to write it", err)
	}
	var want map[string]resolutionSummary
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	for _, key := range sortedSummaryKeys(want, got) {
		left, right := want[key], got[key]
		encodedLeft, _ := json.Marshal(left)
		encodedRight, _ := json.Marshal(right)
		if string(encodedLeft) != string(encodedRight) {
			t.Errorf("%s resolves differently:\n want %s\n  got %s", key, encodedLeft, encodedRight)
		}
	}
}

func sortedSummaryKeys(maps ...map[string]resolutionSummary) []string {
	seen := map[string]bool{}
	var keys []string
	for _, m := range maps {
		for key := range m {
			if !seen[key] {
				seen[key] = true
				keys = append(keys, key)
			}
		}
	}
	sort.Strings(keys)
	return keys
}

func TestDeprecatedModelFieldsResolveAsBefore(t *testing.T) {
	files, err := filepath.Glob(filepath.Join(deprecatedFixtures, "*.yaml"))
	if err != nil || len(files) == 0 {
		t.Fatalf("no fixtures: %v", err)
	}
	got := map[string]resolutionSummary{}
	for _, file := range files {
		summary := summarizeFile(file)
		if summary.Error == "" && summary.Warnings == 0 {
			t.Errorf("%s writes deprecated fields but loads without a deprecation warning", file)
		}
		got[filepath.Base(file)] = summary
	}
	compareResolutionGolden(t, deprecatedGolden, got)
}

func TestMaintainedConfigsResolveToTheirModels(t *testing.T) {
	got := map[string]resolutionSummary{}
	repo := os.DirFS(maintainedRepoPrefix)
	for _, dir := range maintainedConfigDirs {
		err := fs.WalkDir(repo, dir, func(path string, entry fs.DirEntry, err error) error {
			if err == nil && path != dir && strings.HasPrefix(entry.Name(), ".") {
				// Local state, such as the .vllm-sr directory `vllm-sr serve`
				// writes next to a config, is not a maintained asset.
				if entry.IsDir() {
					return fs.SkipDir
				}
				return nil
			}
			if err != nil || entry.IsDir() || (!strings.HasSuffix(path, ".yaml") && !strings.HasSuffix(path, ".yml")) {
				return err
			}
			data, err := fs.ReadFile(repo, path)
			if err != nil {
				return err
			}
			for index, document := range strings.Split(string(data), "\n---\n") {
				config, ok := routerConfigDocument([]byte(document))
				if !ok {
					continue
				}
				key := path
				if index > 0 {
					key = fmt.Sprintf("%s#%d", key, index)
				}
				cfg, parseErr := ParseYAMLBytesWithoutEnvExpansion(config)
				if parseErr != nil {
					got[key] = resolutionSummary{Error: parseErr.Error()}
					continue
				}
				got[key] = summarizeResolution(cfg)
			}
			return nil
		})
		if err != nil && !errors.Is(err, fs.ErrNotExist) {
			t.Fatal(err)
		}
	}
	for path, summary := range got {
		if summary.Warnings > 0 {
			t.Errorf("%s writes %d deprecated model fields; write the canonical form", path, summary.Warnings)
		}
	}
	compareResolutionGolden(t, maintainedGolden, got)
}

// routerConfigDocument returns a YAML document's Router configuration: the
// document itself, or the config a Helm or e2e values file embeds.
func routerConfigDocument(data []byte) ([]byte, bool) {
	var probe map[string]interface{}
	if yaml.Unmarshal(data, &probe) != nil || probe == nil {
		return nil, false
	}
	isConfig := func(m map[string]interface{}) bool {
		return m["version"] != nil || m["routing"] != nil || m["global"] != nil || m["recipes"] != nil || m["entrypoints"] != nil || m["providers"] != nil
	}
	if inner, ok := probe["config"].(map[string]interface{}); ok && isConfig(inner) {
		encoded, err := yaml.Marshal(inner)
		return encoded, err == nil
	}
	return data, isConfig(probe)
}
