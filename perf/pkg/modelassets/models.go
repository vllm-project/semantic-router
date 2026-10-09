// Package modelassets resolves the performance models from the router's canonical catalog.
package modelassets

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Artifact is one benchmark model: a built-in Hub repository at its pinned
// revision, or the local package that VLLM_SR_<NAME>_MODEL names.
type Artifact struct {
	Name     string
	RepoID   string
	Revision string
	// Path is the absolute local package of an override; empty serves the
	// pinned Hub repository.
	Path string
}

// Resolve returns the catalog model of a benchmark task: domain, pii,
// jailbreak or embedding. The classifier tasks measure the per-task
// label_distribution.v1 and token_spans.v1 contracts, which the Vela 1.0
// specialists serve; the default Vela 2.0 0.3B deployment does not.
func Resolve(name, root string) (Artifact, error) {
	specialists := config.Vela1SystemModels()
	paths := map[string]string{
		"domain":    specialists.DomainClassifier,
		"pii":       specialists.PIIClassifier,
		"jailbreak": specialists.PromptGuard,
		"embedding": config.DefaultGlobalConfig().EmbeddingModels.MmBertModelPath,
	}
	spec := config.GetModelByPath(paths[name])
	if spec == nil || spec.RepoID == "" || spec.Revision == "" {
		return Artifact{}, fmt.Errorf("missing pinned performance artifact %q", name)
	}
	artifact := Artifact{Name: name, RepoID: spec.RepoID, Revision: spec.Revision}
	if override := os.Getenv("VLLM_SR_" + strings.ToUpper(name) + "_MODEL"); override != "" {
		if !filepath.IsAbs(override) {
			override = filepath.Join(root, override)
		}
		artifact.Path = override
	}
	return artifact, nil
}

// Deployment is the model_runtime deployment that serves the artifact.
func (a Artifact) Deployment(device string, input config.ModelInputBudget) config.ModelDeployment {
	deployment := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Device: device, Input: input}
	if a.Path != "" {
		deployment.Artifact = a.Path
	} else {
		deployment.Artifact, deployment.Revision = a.RepoID, a.Revision
	}
	return deployment.WithDefaults()
}
