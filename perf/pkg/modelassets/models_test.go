package modelassets

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestResolveServesThePinnedDeclaredModel(t *testing.T) {
	for name, deployment := range map[string]string{
		"decision":  config.DefaultDecisionDeployment,
		"embedding": config.EmbeddingDefaultDeployment,
	} {
		t.Setenv("VLLM_SR_"+strings.ToUpper(name)+"_MODEL", "")
		artifact, err := Resolve(name, t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		declared := config.DefaultModelDeployments()[deployment]
		if artifact.RepoID != declared.Artifact || artifact.Revision != declared.Revision || artifact.Path != "" {
			t.Fatalf("%s must serve the model the %s deployment pins: %+v", name, deployment, artifact)
		}
		served := artifact.Deployment("cpu", config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"})
		if !served.Managed() || served.Artifact != declared.Artifact || served.Revision != declared.Revision || served.Device != "cpu" {
			t.Fatalf("pinned model must run as a managed model_runtime deployment: %+v", served)
		}
	}
	if _, err := Resolve("unknown", t.TempDir()); err == nil {
		t.Fatal("an unknown task resolved to a model")
	}
}

func TestOverrideServesTheLocalPackage(t *testing.T) {
	root := t.TempDir()
	t.Setenv("VLLM_SR_DECISION_MODEL", "snapshot")
	artifact, err := Resolve("decision", root)
	if err != nil || artifact.Path != filepath.Join(root, "snapshot") {
		t.Fatalf("override must name a package under the root: %+v %v", artifact, err)
	}
	deployment := artifact.Deployment("cpu", config.ModelInputBudget{})
	if deployment.Artifact != artifact.Path || deployment.Revision != "" {
		t.Fatalf("a local package has no Hub revision: %+v", deployment)
	}
}
