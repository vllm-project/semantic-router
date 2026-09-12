package modeldownload

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestVelaClassifierDownloadsRetainReleaseRevisionsAndMappings(t *testing.T) {
	const prefix = "models/Vela-1.0-Encoder-307M-"
	cfg := &config.RouterConfig{MoMRegistry: config.ToLegacyRegistry()}
	cfg.CategoryModel.ModelID = prefix + "Domain"
	cfg.CategoryModel.CategoryMappingPath = prefix + "Domain/category_mapping.json"
	cfg.PIIModel.ModelID = prefix + "PII"
	cfg.PIIModel.PIIMappingPath = prefix + "PII/pii_mapping.json"
	cfg.HallucinationMitigation.FactCheckModel.ModelID = prefix + "FactCheck"
	cfg.FactCheckRules = []config.FactCheckRule{{Name: "needs_fact_check"}}
	cfg.ModalityDetector.Enabled = true
	cfg.ModalityDetector.Method = config.ModalityDetectionClassifier
	cfg.ModalityDetector.Classifier = &config.ModalityClassifierConfig{ModelPath: prefix + "Modality"}
	cfg.Decisions = []config.Decision{
		{Name: "subject-route", Rules: config.RuleNode{Type: config.SignalTypeDomain, Name: "economics"}},
		{Name: "privacy-route", Rules: config.RuleNode{Type: config.SignalTypePII, Name: "email"}},
		{Name: "verified-route", Rules: config.RuleNode{Type: config.SignalTypeFactCheck, Name: "needs_fact_check"}},
	}

	specs, err := BuildModelSpecs(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(specs) != 4 {
		t.Fatalf("got %d download specs, want all four configured classifiers: %+v", len(specs), specs)
	}
	for _, spec := range specs {
		model := config.GetModelByPath(spec.LocalPath)
		if model == nil || model.Revision == "" || spec.Revision != model.Revision || spec.RepoID != model.RepoID {
			t.Fatalf("download did not retain release identity: %+v", spec)
		}
		args := buildDownloadArgs(spec)
		i := slices.Index(args, "--revision")
		if i < 0 || i+1 >= len(args) || args[i+1] != model.Revision {
			t.Fatalf("download command lost immutable revision: %v", args)
		}
		if !slices.Contains(spec.RequiredFiles, "config.json") {
			t.Fatalf("download lacks model config: %+v", spec)
		}
		mapping := map[string]string{
			prefix + "Domain": "category_mapping.json",
			prefix + "PII":    "pii_mapping.json",
		}[spec.LocalPath]
		if mapping != "" && !slices.Contains(spec.RequiredFiles, mapping) {
			t.Fatalf("download lacks runtime mapping %q: %+v", mapping, spec)
		}
	}
}

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
