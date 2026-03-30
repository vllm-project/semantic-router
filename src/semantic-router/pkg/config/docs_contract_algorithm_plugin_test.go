package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

var algorithmTutorialBuckets = map[string]string{
	"automix":       "selection",
	"confidence":    "looper",
	"elo":           "selection",
	"gmtrouter":     "selection",
	"hybrid":        "selection",
	"kmeans":        "selection",
	"knn":           "selection",
	"latency-aware": "selection",
	"ratings":       "looper",
	"remom":         "looper",
	"rl-driven":     "selection",
	"router-dc":     "selection",
	"static":        "selection",
	"svm":           "selection",
}

var retiredAlgorithmTutorialDocs = []string{
	repoRel("website", "docs", "tutorials", "algorithm", "selection.md"),
	repoRel("website", "docs", "tutorials", "algorithm", "looper.md"),
}

var pluginTutorialBuckets = map[string]string{
	"content-safety":     "safety-and-generation",
	"fast-response":      "response-and-mutation",
	"hallucination":      "safety-and-generation",
	"header-mutation":    "response-and-mutation",
	"image-gen":          "response-and-mutation",
	"jailbreak":          "safety-and-generation",
	"memory":             "retrieval-and-memory",
	"pii":                "safety-and-generation",
	"rag":                "retrieval-and-memory",
	"request-params":     "response-and-mutation",
	"response-jailbreak": "safety-and-generation",
	"router-replay":      "retrieval-and-memory",
	"semantic-cache":     "retrieval-and-memory",
	"system-prompt":      "response-and-mutation",
	"tools":              "response-and-mutation",
}

var retiredPluginTutorialDocs = []string{
	repoRel("website", "docs", "tutorials", "plugin", "response-and-mutation.md"),
	repoRel("website", "docs", "tutorials", "plugin", "retrieval-and-memory.md"),
	repoRel("website", "docs", "tutorials", "plugin", "safety-and-generation.md"),
}

func assertAlgorithmTutorialDocsMatchConfigHierarchy(t *testing.T, root string) {
	t.Helper()

	configRoot := filepath.Join(root, repoRel("config", "algorithm"))
	remaining := copyStringStringMap(algorithmTutorialBuckets)
	seen := 0

	err := filepath.Walk(configRoot, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || filepath.Ext(path) != ".yaml" {
			return nil
		}

		family := filepath.Base(filepath.Dir(path))
		stem := trimExt(filepath.Base(path))
		expectedBucket, ok := remaining[stem]
		if !ok {
			t.Fatalf("%s is missing a tutorial bucket mapping for algorithm %q", configRoot, stem)
		}
		if expectedBucket != family {
			t.Fatalf("algorithm tutorial mapping says %q belongs in %q, but config fragment is under %q", stem, expectedBucket, family)
		}

		docPath := repoRel("website", "docs", "tutorials", "algorithm", family, stem+".md")
		if _, err := os.Stat(filepath.Join(root, docPath)); err != nil {
			t.Fatalf("%s should exist for %s: %v", docPath, path, err)
		}

		delete(remaining, stem)
		seen++
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk %s: %v", configRoot, err)
	}
	if seen == 0 {
		t.Fatalf("%s should contain algorithm fragments", configRoot)
	}
	for algorithm := range remaining {
		t.Fatalf("algorithm tutorial mapping declares %q, but config/algorithm is missing it", algorithm)
	}

	assertPathsDoNotExist(t, root, retiredAlgorithmTutorialDocs)
}

func assertPluginTutorialDocsMatchConfigHierarchy(t *testing.T, root string) {
	t.Helper()

	configRoot := filepath.Join(root, repoRel("config", "plugin"))
	entries, err := os.ReadDir(configRoot)
	if err != nil {
		t.Fatalf("failed to read %s: %v", configRoot, err)
	}

	remaining := copyStringStringMap(pluginTutorialBuckets)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		plugin := entry.Name()
		if _, ok := remaining[plugin]; !ok {
			t.Fatalf("%s is missing a tutorial bucket mapping for plugin %q", configRoot, plugin)
		}

		docPath := repoRel("website", "docs", "tutorials", "plugin", plugin+".md")
		if _, err := os.Stat(filepath.Join(root, docPath)); err != nil {
			t.Fatalf("%s should exist for config/plugin/%s: %v", docPath, plugin, err)
		}
		delete(remaining, plugin)
	}

	for plugin := range remaining {
		t.Fatalf("plugin tutorial mapping declares %q, but config/plugin/%s is missing", plugin, plugin)
	}

	assertPathsDoNotExist(t, root, retiredPluginTutorialDocs)
}

func algorithmTutorialSidebarEntries() []string {
	entries := []string{}
	names := sortedStringKeys(algorithmTutorialBuckets)
	for _, name := range names {
		entries = append(entries, fmt.Sprintf("'tutorials/algorithm/%s/%s'", algorithmTutorialBuckets[name], name))
	}
	return entries
}

func pluginTutorialSidebarEntries() []string {
	entries := []string{}
	names := sortedStringKeys(pluginTutorialBuckets)
	for _, name := range names {
		entries = append(entries, fmt.Sprintf("'tutorials/plugin/%s'", name))
	}
	return entries
}

func sortedStringKeys(src map[string]string) []string {
	keys := make([]string, 0, len(src))
	for key := range src {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func trimExt(name string) string {
	return name[:len(name)-len(filepath.Ext(name))]
}
