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
				"`go test ./pkg/config/...`",
			},
		},
		{
			path: "website/docs/installation/configuration.md",
			required: []string{
				"`version/listeners/providers/routing/global`",
				"`routing.modelCards`",
				"`config/algorithm/`",
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
