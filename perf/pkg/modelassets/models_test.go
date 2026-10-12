package modelassets

import (
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolveServesThePinnedCatalogModel(t *testing.T) {
	t.Setenv("VLLM_SR_DOMAIN_MODEL", "")
	artifact, err := Resolve("domain", t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	spec := config.GetModelByPath(config.Vela1SystemModels().DomainClassifier)
	if artifact.RepoID != spec.RepoID || artifact.Revision != spec.Revision || artifact.Path != "" {
		t.Fatalf("domain must serve its pinned Vela 1.0 specialist: %+v", artifact)
	}
	deployment := artifact.Deployment("cpu", config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"})
	if !deployment.Managed() || deployment.Artifact != spec.RepoID || deployment.Revision != spec.Revision || deployment.Device != "cpu" {
		t.Fatalf("pinned model must run as a managed model_runtime deployment: %+v", deployment)
	}
	if _, err = Resolve("unknown", t.TempDir()); err == nil {
		t.Fatal("an unknown task resolved to a model")
	}
}

func TestOverrideServesTheLocalPackage(t *testing.T) {
	root := t.TempDir()
	t.Setenv("VLLM_SR_PII_MODEL", "snapshot")
	artifact, err := Resolve("pii", root)
	if err != nil || artifact.Path != filepath.Join(root, "snapshot") {
		t.Fatalf("override must name a package under the root: %+v %v", artifact, err)
	}
	deployment := artifact.Deployment("cpu", config.ModelInputBudget{})
	if deployment.Artifact != artifact.Path || deployment.Revision != "" {
		t.Fatalf("a local package has no Hub revision: %+v", deployment)
	}
}
