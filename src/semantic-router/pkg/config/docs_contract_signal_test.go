package config

import (
	"os"
	"path"
	"path/filepath"
	"sort"
	"testing"
)

// signalTutorialBuckets maps each signal family to the directory of its
// tutorial under tutorials/signal: the layer it belongs to, or none for the
// systemone primitive.
var signalTutorialBuckets = map[string]string{
	"safety":         "application",
	"authz":          "heuristic",
	"complexity":     "application",
	"context":        "heuristic",
	"conversation":   "heuristic",
	"domain":         "application",
	"fact-check":     "application",
	"hallucination":  "application",
	"jailbreak":      "application",
	"keyword":        "heuristic",
	"language":       "heuristic",
	"modality":       "application",
	"pii":            "application",
	"user-feedback":  "application",
	"metadata":       "heuristic",
	"input-modality": "heuristic",
	"systemone":      "",
	"refusal":        "application",
	"turn-relation":  "application",
}

var retiredSignalTutorialDocs = []string{
	repoRel("website", "docs", "tutorials", "signal", "routing.md"),
	repoRel("website", "docs", "tutorials", "signal", "safety.md"),
	repoRel("website", "docs", "tutorials", "signal", "operational.md"),
	repoRel("website", "docs", "tutorials", "signal", "heuristic", "topic-continuity.md"),
	repoRel("website", "docs", "tutorials", "signal", "heuristic", "action.md"),
	repoRel("website", "docs", "tutorials", "signal", "heuristic", "event.md"),
	repoRel("website", "docs", "tutorials", "signal", "heuristic", "structure.md"),
	repoRel("website", "docs", "tutorials", "signal", "learned"),
	repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "tutorials", "signal", "learned"),
}

func assertSignalTutorialDocsMatchConfigHierarchy(t *testing.T, root string) {
	t.Helper()

	configRoot := filepath.Join(root, repoRel("config", "fragments", "signal"))
	entries, err := os.ReadDir(configRoot)
	if err != nil {
		t.Fatalf("failed to read %s: %v", configRoot, err)
	}

	remaining := copyStringStringMap(signalTutorialBuckets)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		bucket, ok := remaining[entry.Name()]
		if !ok {
			t.Fatalf("%s is missing a tutorial bucket mapping for signal family %q", configRoot, entry.Name())
		}
		docPath := repoRel("website", "docs", "tutorials", "signal", bucket, entry.Name()+".md")
		fullDocPath := filepath.Join(root, docPath)
		if _, err := os.Stat(fullDocPath); err != nil {
			t.Fatalf("%s should exist for config/fragments/signal/%s: %v", docPath, entry.Name(), err)
		}
		delete(remaining, entry.Name())
	}

	for family := range remaining {
		t.Fatalf("signal tutorial mapping declares %q, but config/fragments/signal/%s is missing", family, family)
	}

	assertPathsDoNotExist(t, root, retiredSignalTutorialDocs)
}

func signalTutorialSidebarEntries() []string {
	entries := []string{}
	families := make([]string, 0, len(signalTutorialBuckets))
	for family := range signalTutorialBuckets {
		families = append(families, family)
	}
	sort.Strings(families)
	for _, family := range families {
		entries = append(entries, "'"+path.Join("tutorials", "signal", signalTutorialBuckets[family], family)+"'")
	}
	return entries
}

func copyStringStringMap(src map[string]string) map[string]string {
	dst := make(map[string]string, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}
