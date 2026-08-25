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
	fragmentsRoot := filepath.Join(configRoot, "fragments")

	for _, signalType := range SupportedSignalTypes() {
		dir := filepath.Join(fragmentsRoot, "signal", fragmentDirName(signalType))
		requireYAMLFilesInDir(t, dir)
	}

	requiredDecisionCategories := []string{"single", "and", "or", "not", "composite"}
	for _, category := range requiredDecisionCategories {
		dir := filepath.Join(fragmentsRoot, "decision", category)
		requireYAMLFilesInDir(t, dir)
	}

	requiredAlgorithmFragments := map[string]string{
		"automix":       filepath.Join("selection", "automix.yaml"),
		"confidence":    filepath.Join("looper", "confidence.yaml"),
		"fusion":        filepath.Join("looper", "fusion.yaml"),
		"hybrid":        filepath.Join("selection", "hybrid.yaml"),
		"kmeans":        filepath.Join("selection", "kmeans.yaml"),
		"knn":           filepath.Join("selection", "knn.yaml"),
		"latency_aware": filepath.Join("selection", "latency-aware.yaml"),
		"mlp":           filepath.Join("selection", "mlp.yaml"),
		"multi_factor":  filepath.Join("selection", "multi-factor.yaml"),
		"ratings":       filepath.Join("looper", "ratings.yaml"),
		"remom":         filepath.Join("looper", "remom.yaml"),
		"router_dc":     filepath.Join("selection", "router-dc.yaml"),
		"static":        filepath.Join("selection", "static.yaml"),
		"svm":           filepath.Join("selection", "svm.yaml"),
		"workflows":     filepath.Join("looper", "workflows.yaml"),
		"prompt":        filepath.Join("selection", "prompt.yaml"),
	}
	for _, algorithmType := range SupportedDecisionAlgorithmTypes() {
		relPath, ok := requiredAlgorithmFragments[algorithmType]
		if !ok {
			t.Fatalf("missing fragment mapping for algorithm type %q", algorithmType)
		}
		requireYAMLFile(t, filepath.Join(fragmentsRoot, "algorithm", relPath))
	}

	for _, pluginType := range SupportedDecisionPluginTypes() {
		dir := filepath.Join(fragmentsRoot, "plugin", fragmentDirName(pluginType))
		requireYAMLFilesInDir(t, dir)
	}
}

func TestConfigFragmentsStayUnderUnifiedDirectory(t *testing.T) {
	root := repoRootFromTestFile(t)
	configRoot := filepath.Join(root, "config")
	fragmentsRoot := filepath.Join(configRoot, "fragments")

	for _, category := range []string{"signal", "decision", "algorithm", "plugin"} {
		requireDirectory(t, filepath.Join(fragmentsRoot, category))
		if _, err := os.Stat(filepath.Join(configRoot, category)); !os.IsNotExist(err) {
			t.Fatalf("legacy fragment directory config/%s must not exist", category)
		}
	}
}

func TestConfigFragmentsAreValidYAML(t *testing.T) {
	root := repoRootFromTestFile(t)
	configRoot := filepath.Join(root, "config", "fragments")

	err := filepath.Walk(configRoot, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || !strings.HasSuffix(info.Name(), ".yaml") {
			return nil
		}

		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var doc map[string]interface{}
		if err := yaml.Unmarshal(data, &doc); err != nil {
			t.Fatalf("failed to parse YAML fragment %s: %v", path, err)
		}
		if _, retired := doc["document"]; retired {
			t.Fatalf("fragment %s uses retired top-level document; use routing", path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk config fragment catalog: %v", err)
	}
}

func TestConfigFragmentsAvoidRetiredDomainAliases(t *testing.T) {
	root := repoRootFromTestFile(t)
	configRoot := filepath.Join(root, "config", "fragments")

	err := filepath.Walk(configRoot, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || !strings.HasSuffix(info.Name(), ".yaml") {
			return nil
		}

		data, err := os.ReadFile(path)
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

func requireDirectory(t *testing.T, path string) {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("expected directory %s: %v", path, err)
	}
	if !info.IsDir() {
		t.Fatalf("expected directory %s, found file", path)
	}
}

func fragmentDirName(name string) string {
	return strings.ReplaceAll(name, "_", "-")
}
