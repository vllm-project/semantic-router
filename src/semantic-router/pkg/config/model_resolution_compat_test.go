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

// The golden file lists, for each shipped configuration, the model every
// active signal of every scope resolves to and the model_runtime deployments
// the Router starts. UPDATE_MODEL_RESOLUTION=1 rewrites it.

const (
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
	Error   string            `json:"error,omitempty"`
	Signals map[string]string `json:"signals,omitempty"`
	Started []string          `json:"started,omitempty"`
}

func resolvedIdentity(name string, d ModelDeployment) string {
	d = d.WithDefaults()
	switch d.Provider {
	case ModelRuntimeProvider:
		if !d.Managed() {
			return fmt.Sprintf("model_runtime %s %s", d.Endpoint, d.ServedModel(name))
		}
		return strings.TrimSpace(fmt.Sprintf("model_runtime %s@%s %s replicas=%d", d.Artifact, d.Revision, d.Device, len(d.Replicas)))
	case OpenAIChatProvider:
		return strings.TrimSpace(fmt.Sprintf("remote %s %s", strings.TrimSuffix(strings.TrimRight(d.Endpoint, "/"), "/v1"), d.Model))
	}
	return strings.TrimSpace(fmt.Sprintf("%s %s %s", d.Provider, d.Endpoint, d.Model))
}

func summarizeResolution(cfg *RouterConfig) resolutionSummary {
	summary := resolutionSummary{Signals: map[string]string{}}
	if err := validateModelBindingContracts(cfg); err != nil {
		summary.Error = "plan: " + err.Error()
		return summary
	}
	for recipe, scoped := range cfg.consumerScopes() {
		for _, requirement := range EmbeddingRequirements(scoped, false) {
			summary.Signals[string(recipe)+"/embedding"] = resolvedIdentity(requirement.Deployment, cfg.ModelDeployments[requirement.Deployment])
		}
		for _, rule := range scoped.SystemOneRules {
			name := scoped.SystemOneRuleDeployment(rule)
			summary.Signals[string(recipe)+"/systemone."+rule.Name] = resolvedIdentity(name, cfg.ModelDeployments[name])
		}
		for _, rule := range scoped.ApplicationRules() {
			name := scoped.SystemOneRuleDeployment(rule.Rule)
			summary.Signals[string(recipe)+"/"+rule.Type+"."+rule.Rule.Name] = resolvedIdentity(name, cfg.ModelDeployments[name])
		}
	}
	for name, deployment := range ModelRuntimeDeploymentsInUse(cfg) {
		summary.Started = append(summary.Started, resolvedIdentity(name, deployment))
	}
	sort.Strings(summary.Started)
	return summary
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

// realModelProfiles are the only E2E profiles that start a model. Every other
// profile answers its learned signals and embeddings from provider-mocker
// fixtures, since E2E pins contracts and the release benchmarks measure
// model quality; a real model there only adds a download and a slow start.
var realModelProfiles = map[string]bool{
	"model-runtime":       true,
	"model-runtime-real":  true,
	"model-runtime-vela3": true,
}

func TestE2EProfilesStartNoModelOutsideTheRuntimeProfiles(t *testing.T) {
	data, err := os.ReadFile(maintainedGolden)
	if err != nil {
		t.Fatal(err)
	}
	var summaries map[string]resolutionSummary
	if err := json.Unmarshal(data, &summaries); err != nil {
		t.Fatal(err)
	}
	prefix := filepath.Join("e2e", "profiles") + string(filepath.Separator)
	for _, key := range sortedSummaryKeys(summaries) {
		profile, file, ok := strings.Cut(strings.TrimPrefix(key, prefix), string(filepath.Separator))
		if !strings.HasPrefix(key, prefix) || !ok || !strings.HasPrefix(file, "values") {
			continue
		}
		if started := summaries[key].Started; len(started) > 0 && !realModelProfiles[profile] {
			t.Errorf("%s starts %v; bind its signals and embeddings to the provider-mocker fixtures", key, started)
		}
	}
}
