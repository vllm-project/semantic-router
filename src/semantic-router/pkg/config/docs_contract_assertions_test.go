package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func assertDocsContainAll(t *testing.T, root string, docs []docNeedles) {
	t.Helper()
	for _, doc := range docs {
		assertStringContainsAll(t, readRepoFile(t, root, doc.path), doc.path, doc.needles)
	}
}

func assertDocsDoNotContainAny(t *testing.T, root string, docs []docNeedles) {
	t.Helper()
	for _, doc := range docs {
		assertStringContainsNone(t, readRepoFile(t, root, doc.path), doc.path, doc.needles)
	}
}

func assertTutorialSidebarTaxonomy(t *testing.T, root string) {
	t.Helper()
	sidebarPath := repoRel("website", "sidebars.ts")
	content := readRepoFile(t, root, sidebarPath)
	required := append([]string(nil), latestTutorialSidebarRequired...)
	required = append(required, signalTutorialSidebarEntries()...)
	required = append(required, algorithmTutorialSidebarEntries()...)
	required = append(required, pluginTutorialSidebarEntries()...)
	assertStringContainsAll(t, content, sidebarPath, required)
	assertStringContainsNone(t, content, sidebarPath, latestTutorialSidebarForbidden)
}

func assertTutorialFilesContainRequiredSections(t *testing.T, root string) {
	t.Helper()
	tutorialRoot := filepath.Join(root, repoRel("website", "docs", "tutorials"))
	err := filepath.Walk(tutorialRoot, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}
		contentBytes, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		assertStringContainsAll(t, string(contentBytes), path, latestTutorialRequiredSections)
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk latest tutorial files: %v", err)
	}
}

func assertTutorialRootDirectories(t *testing.T, root string) {
	t.Helper()
	tutorialRoot := filepath.Join(root, repoRel("website", "docs", "tutorials"))
	entries, err := os.ReadDir(tutorialRoot)
	if err != nil {
		t.Fatalf("failed to read latest tutorial root: %v", err)
	}

	allowed := copyStringBoolMap(latestTutorialAllowedDirectories)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if !allowed[entry.Name()] {
			t.Fatalf("%s contains retired top-level directory %q", tutorialRoot, entry.Name())
		}
		delete(allowed, entry.Name())
	}
	for remaining := range allowed {
		t.Fatalf("%s is missing required top-level directory %q", tutorialRoot, remaining)
	}
}

func assertMarkdownTreeDoesNotContainAny(t *testing.T, root string, forbidden []string) {
	t.Helper()
	if _, err := os.Stat(root); os.IsNotExist(err) {
		return
	}
	err := filepath.Walk(root, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}
		contentBytes, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		assertStringContainsNone(t, string(contentBytes), path, forbidden)
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk tutorial docs under %s: %v", root, err)
	}
}

func assertPathsDoNotExist(t *testing.T, root string, relPaths []string) {
	t.Helper()
	for _, relPath := range relPaths {
		fullPath := filepath.Join(root, relPath)
		if _, err := os.Stat(fullPath); err == nil {
			t.Fatalf("%s should not exist; let latest docs fall back to the canonical current source instead", relPath)
		} else if !os.IsNotExist(err) {
			t.Fatalf("failed to stat %s: %v", relPath, err)
		}
	}
}

func tutorialDocRoots(root string) []string {
	return []string{
		filepath.Join(root, repoRel("website", "docs", "tutorials")),
		filepath.Join(root, repoRel("website", "i18n", "zh-Hans", "docusaurus-plugin-content-docs", "current", "tutorials")),
	}
}

func readRepoFile(t *testing.T, root string, relPath string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(root, relPath))
	if err != nil {
		t.Fatalf("failed to read %s: %v", relPath, err)
	}
	return string(data)
}

func assertStringContainsAll(t *testing.T, content string, label string, needles []string) {
	t.Helper()
	for _, needle := range needles {
		if !strings.Contains(content, needle) {
			t.Fatalf("%s is missing required text %q", label, needle)
		}
	}
}

func assertStringContainsNone(t *testing.T, content string, label string, needles []string) {
	t.Helper()
	for _, needle := range needles {
		if strings.Contains(content, needle) {
			t.Fatalf("%s still contains retired text %q", label, needle)
		}
	}
}

func copyStringBoolMap(source map[string]bool) map[string]bool {
	clone := make(map[string]bool, len(source))
	for key, value := range source {
		clone[key] = value
	}
	return clone
}
