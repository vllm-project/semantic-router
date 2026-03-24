package config

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

var signalTutorialBuckets = map[string]string{
	"authz":         "heuristic",
	"complexity":    "learned",
	"context":       "heuristic",
	"domain":        "learned",
	"embedding":     "learned",
	"fact-check":    "learned",
	"jailbreak":     "learned",
	"keyword":       "heuristic",
	"language":      "heuristic",
	"modality":      "learned",
	"structure":     "heuristic",
	"pii":           "learned",
	"preference":    "learned",
	"user-feedback": "learned",
}

var retiredSignalTutorialDocs = []string{
	repoRel("website", "docs", "tutorials", "signal", "routing.md"),
	repoRel("website", "docs", "tutorials", "signal", "safety.md"),
	repoRel("website", "docs", "tutorials", "signal", "operational.md"),
}

func assertSignalTutorialDocsMatchConfigHierarchy(t *testing.T, root string) {
	t.Helper()

	configRoot := filepath.Join(root, repoRel("config", "signal"))
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
			t.Fatalf("%s should exist for config/signal/%s: %v", docPath, entry.Name(), err)
		}
		delete(remaining, entry.Name())
	}

	for family := range remaining {
		t.Fatalf("signal tutorial mapping declares %q, but config/signal/%s is missing", family, family)
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
		entries = append(entries, fmt.Sprintf("'tutorials/signal/%s/%s'", signalTutorialBuckets[family], family))
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
