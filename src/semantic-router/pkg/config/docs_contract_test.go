package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConfigContractDocsStayAligned(t *testing.T) {
	root := repoRootFromTestFile(t)

	testCases := []struct {
		path     string
		required []string
	}{
		{
			path: "config/README.md",
			required: []string{
				"`config/config.yaml`",
				"`config/signal/`",
				"`config/decision/`",
				"`config/algorithm/`",
				"`config/plugin/`",
				"`tutorials/global/`",
				"`go test ./pkg/config/...`",
			},
		},
		{
			path: "website/docs/installation/configuration.md",
			required: []string{
				"`version/listeners/providers/routing/global`",
				"`routing.modelCards`",
				"`config/algorithm/`",
				"`tutorials/global/`",
				"vllm-sr config migrate --config old-config.yaml",
				"v0.3. The steady-state file is `config.yaml`",
			},
		},
		{
			path: "website/docs/proposals/unified-config-contract-v0-3.md",
			required: []string{
				"version:\nlisteners:\nproviders:\nrouting:\nglobal:",
				"`routing.modelCards`",
				"`config/algorithm/`",
				"`providers.models`",
				"vllm-sr init",
			},
		},
	}

	for _, tc := range testCases {
		data, err := os.ReadFile(filepath.Join(root, tc.path))
		if err != nil {
			t.Fatalf("failed to read %s: %v", tc.path, err)
		}
		content := string(data)
		for _, needle := range tc.required {
			if !strings.Contains(content, needle) {
				t.Fatalf("%s is missing required config-contract text %q", tc.path, needle)
			}
		}
	}
}

func TestCurrentTutorialDocsDoNotReferenceRemovedConfigFiles(t *testing.T) {
	root := repoRootFromTestFile(t)
	docRoots := []string{
		filepath.Join(root, "website/docs/tutorials"),
		filepath.Join(root, "website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials"),
	}
	forbidden := []string{
		"router-config.yaml",
		"router-defaults.yaml",
	}

	for _, docRoot := range docRoots {
		if _, err := os.Stat(docRoot); os.IsNotExist(err) {
			continue
		}
		err := filepath.Walk(docRoot, func(path string, info os.FileInfo, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if info.IsDir() || filepath.Ext(path) != ".md" {
				return nil
			}

			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			content := string(data)
			for _, needle := range forbidden {
				if strings.Contains(content, needle) {
					t.Fatalf("%s still references removed config file %q", path, needle)
				}
			}
			return nil
		})
		if err != nil {
			t.Fatalf("failed to walk tutorial docs under %s: %v", docRoot, err)
		}
	}
}

func TestLatestTutorialTaxonomyMatchesConfigHierarchy(t *testing.T) {
	root := repoRootFromTestFile(t)

	sidebarPath := filepath.Join(root, "website/sidebars.ts")
	sidebarData, err := os.ReadFile(sidebarPath)
	if err != nil {
		t.Fatalf("failed to read %s: %v", sidebarPath, err)
	}
	sidebar := string(sidebarData)
	for _, needle := range []string{
		"label: 'Signals'",
		"label: 'Decisions'",
		"label: 'Algorithms'",
		"label: 'Plugins'",
		"label: 'Global'",
		"'tutorials/signal/overview'",
		"'tutorials/decision/overview'",
		"'tutorials/algorithm/overview'",
		"'tutorials/plugin/overview'",
		"'tutorials/global/overview'",
	} {
		if !strings.Contains(sidebar, needle) {
			t.Fatalf("website/sidebars.ts is missing tutorial taxonomy entry %q", needle)
		}
	}

	requiredSections := []string{
		"## Overview",
		"## Key Advantages",
		"## What Problem Does It Solve?",
		"## When to Use",
		"## Configuration",
	}
	for _, forbidden := range []string{
		"'tutorials/intelligent-route/",
		"'tutorials/content-safety/",
		"'tutorials/semantic-cache/",
		"'tutorials/observability/",
		"'tutorials/response-api/",
		"'tutorials/performance-tuning/",
		"'tutorials/runtime/",
	} {
		if strings.Contains(sidebar, forbidden) {
			t.Fatalf("website/sidebars.ts still exposes retired latest-nav entry %q", forbidden)
		}
	}

	testCases := []struct {
		path     string
		required []string
	}{
		{
			path: "website/docs/tutorials/signal/overview.md",
			required: []string{
				"`config/signal/`",
				"[Routing Signals](./routing)",
			},
		},
		{
			path: "website/docs/tutorials/decision/overview.md",
			required: []string{
				"`config/decision/`",
				"`decision.algorithm`",
				"`decision.plugins`",
			},
		},
		{
			path: "website/docs/tutorials/algorithm/overview.md",
			required: []string{
				"`config/algorithm/`",
				"[Selection](./selection)",
				"[Looper](./looper)",
			},
		},
		{
			path: "website/docs/tutorials/plugin/overview.md",
			required: []string{
				"`config/plugin/`",
				"`routing.decisions[].plugins`",
			},
		},
		{
			path: "website/docs/tutorials/global/overview.md",
			required: []string{
				"`global:`",
				"`signal/`",
			},
		},
	}
	for _, tc := range testCases {
		data, readErr := os.ReadFile(filepath.Join(root, tc.path))
		if readErr != nil {
			t.Fatalf("failed to read %s: %v", tc.path, readErr)
		}
		content := string(data)
		for _, needle := range tc.required {
			if !strings.Contains(content, needle) {
				t.Fatalf("%s is missing required tutorial-taxonomy text %q", tc.path, needle)
			}
		}
	}

	err = filepath.Walk(filepath.Join(root, "website/docs/tutorials"), func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		content := string(data)
		for _, heading := range requiredSections {
			if !strings.Contains(content, heading) {
				t.Fatalf("%s is missing required tutorial section %q", path, heading)
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("failed to walk latest tutorial files: %v", err)
	}

	entries, err := os.ReadDir(filepath.Join(root, "website/docs/tutorials"))
	if err != nil {
		t.Fatalf("failed to read latest tutorial root: %v", err)
	}
	allowed := map[string]bool{
		"signal":    true,
		"decision":  true,
		"algorithm": true,
		"plugin":    true,
		"global":    true,
	}
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		if !allowed[entry.Name()] {
			t.Fatalf("website/docs/tutorials contains retired top-level directory %q", entry.Name())
		}
		delete(allowed, entry.Name())
	}
	for remaining := range allowed {
		t.Fatalf("website/docs/tutorials is missing required top-level directory %q", remaining)
	}
}
