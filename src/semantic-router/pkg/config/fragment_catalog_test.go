package config

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestConfigFragmentCatalogCoversSupportedRoutingSurfaces(t *testing.T) {
	root := repoRootFromTestFile(t)
	configRoot := filepath.Join(root, "config")

	for _, signalType := range SupportedSignalTypes() {
		dir := filepath.Join(configRoot, "signal", fragmentDirName(signalType))
		requireYAMLFilesInDir(t, dir)
	}

	requiredDecisionCategories := []string{"single", "and", "or", "not", "composite"}
	for _, category := range requiredDecisionCategories {
		dir := filepath.Join(configRoot, "decision", category)
		requireYAMLFilesInDir(t, dir)
	}

	requiredAlgorithmFragments := map[string]string{
		"automix":       filepath.Join("selection", "automix.yaml"),
		"confidence":    filepath.Join("looper", "confidence.yaml"),
		"elo":           filepath.Join("selection", "elo.yaml"),
		"gmtrouter":     filepath.Join("selection", "gmtrouter.yaml"),
		"hybrid":        filepath.Join("selection", "hybrid.yaml"),
		"kmeans":        filepath.Join("selection", "kmeans.yaml"),
		"knn":           filepath.Join("selection", "knn.yaml"),
		"latency_aware": filepath.Join("selection", "latency-aware.yaml"),
		"ratings":       filepath.Join("looper", "ratings.yaml"),
		"remom":         filepath.Join("looper", "remom.yaml"),
		"rl_driven":     filepath.Join("selection", "rl-driven.yaml"),
		"router_dc":     filepath.Join("selection", "router-dc.yaml"),
		"static":        filepath.Join("selection", "static.yaml"),
		"svm":           filepath.Join("selection", "svm.yaml"),
	}
	for _, algorithmType := range SupportedDecisionAlgorithmTypes() {
		relPath, ok := requiredAlgorithmFragments[algorithmType]
		if !ok {
			t.Fatalf("missing fragment mapping for algorithm type %q", algorithmType)
		}
		requireYAMLFile(t, filepath.Join(configRoot, "algorithm", relPath))
	}

	for _, pluginType := range SupportedDecisionPluginTypes() {
		dir := filepath.Join(configRoot, "plugin", fragmentDirName(pluginType))
		requireYAMLFilesInDir(t, dir)
	}
}

func TestConfigFragmentsAreValidYAML(t *testing.T) {
	root := repoRootFromTestFile(t)
	configRoot := filepath.Join(root, "config")

	err := filepath.Walk(configRoot, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || !strings.HasSuffix(info.Name(), ".yaml") {
			return nil
		}

		data, err := os.ReadFile(path) //nolint:gosec // G122: test walks static config tree, no symlink risk
		if err != nil {
			return err
		}
		var doc interface{}
		if err := yaml.Unmarshal(data, &doc); err != nil {
			t.Fatalf("failed to parse YAML fragment %s: %v", path, err)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk config fragment catalog: %v", err)
	}
}

func TestConfigFragmentsAvoidRetiredDomainAliases(t *testing.T) {
	root := repoRootFromTestFile(t)
	configRoot := filepath.Join(root, "config")

	err := filepath.Walk(configRoot, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || !strings.HasSuffix(info.Name(), ".yaml") {
			return nil
		}

		data, err := os.ReadFile(path) //nolint:gosec // G122: test walks static config tree, no symlink risk
		if err != nil {
			return err
		}
		content := string(data)
		for _, forbidden := range []string{"computer_science", "name: technical\n"} {
			if strings.Contains(content, forbidden) {
				t.Fatalf("%s still contains retired domain alias %q", path, forbidden)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk config fragment catalog: %v", err)
	}
}

func repoRootFromTestFile(t *testing.T) string {
	t.Helper()
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve test filename")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "../../../../"))
}

func requireYAMLFilesInDir(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("failed to read fragment dir %s: %v", dir, err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if strings.HasSuffix(entry.Name(), ".yaml") {
			return
		}
	}
	t.Fatalf("fragment dir %s does not contain any YAML files", dir)
}

func requireYAMLFile(t *testing.T, path string) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("expected fragment file %s: %v", path, err)
	}
	if info.IsDir() {
		t.Fatalf("expected fragment file %s, found directory", path)
	}
}

func fragmentDirName(name string) string {
	return strings.ReplaceAll(name, "_", "-")
}
