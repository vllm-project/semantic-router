package modeldownload

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDownloadReleasePinDoesNotLeakToCustomRepository(t *testing.T) {
	original := config.DefaultModelRegistry
	t.Cleanup(func() { config.DefaultModelRegistry = original })
	const path = "models/versioned-test-model"
	const repo = "example/versioned-test-model"
	const revision = "0123456789abcdef0123456789abcdef01234567"
	config.DefaultModelRegistry = append(append([]config.ModelSpec{}, original...), config.ModelSpec{
		LocalPath: path, RepoID: repo, Revision: revision,
	})
	cfg := &config.RouterConfig{}
	cfg.CategoryModel.ModelID = path
	cfg.CategoryModel.CategoryMappingPath = path + "/category_mapping.json"
	cfg.Decisions = []config.Decision{{Name: "domain-route", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "billing"}}}
	cfg.MoMRegistry = map[string]string{path: repo}
	for _, tc := range []struct{ repo, revision string }{
		{repo, revision}, {"user/custom-checkpoint", "main"},
	} {
		cfg.MoMRegistry[path] = tc.repo
		specs, err := BuildModelSpecs(cfg)
		if err != nil || len(specs) != 1 {
			t.Fatalf("specs=%v, err=%v", specs, err)
		}
		if specs[0].Revision != tc.revision {
			t.Fatalf("repo=%s revision=%s, want %s", tc.repo, specs[0].Revision, tc.revision)
		}
		if tc.revision != "main" {
			args := buildDownloadArgs(specs[0])
			found := false
			for i := 0; i+1 < len(args); i++ {
				found = found || (args[i] == "--revision" && args[i+1] == revision)
			}
			if !found {
				t.Fatal("immutable revision did not reach the HF download command")
			}
		}
	}
}
