// Package modelassets resolves the performance models from the router's pre-declared deployments.
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

// Resolve returns the model of a benchmark task, as a deployment every
// configuration declares serves it: decision, the decision model that answers
// the Router's question-style signals (primary), or embedding, the model
// recall embeds with (vela-embedding, embeddings.default).
func Resolve(name, root string) (Artifact, error) {
	deployments := map[string]string{
		"decision":  config.DefaultDecisionDeployment,
		"embedding": config.EmbeddingDefaultDeployment,
	}
	declared, ok := config.DefaultModelDeployments()[deployments[name]]
	if !ok || declared.Artifact == "" || declared.Revision == "" {
		return Artifact{}, fmt.Errorf("missing pinned performance artifact %q", name)
	}
	artifact := Artifact{Name: name, RepoID: declared.Artifact, Revision: declared.Revision}
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
