package config

import (
	"fmt"
	"net/url"
	"path/filepath"
	"regexp"
	"strings"
)

// ModelRuntimeProvider serves a deployment through the built-in model runtime
// (src/model-runtime). Without an endpoint the Router starts and supervises
// the runtime process on a private Unix socket; with one it attaches to an
// engine it does not manage.
const ModelRuntimeProvider = "model_runtime"

var (
	modelRuntimeProfiles = []string{"exact", "shared_context", "batching", "max_speed"}
	modelRuntimeDevice   = regexp.MustCompile(`^(auto|cpu|cuda|rocm)(:[0-9]+)?$`)
	modelRuntimeRevision = regexp.MustCompile(`^[0-9a-f]{40}$`)
	hubRepositoryID      = regexp.MustCompile(`^[A-Za-z0-9][\w.-]*/[\w.-]+$`)
)

// IsModelRuntime reports whether the deployment is served by the built-in model runtime.
func (d ModelDeployment) IsModelRuntime() bool {
	return d.Provider == ModelRuntimeProvider
}

// Managed reports whether the Router starts and supervises the runtime process.
func (d ModelDeployment) Managed() bool {
	return d.IsModelRuntime() && strings.TrimSpace(d.Endpoint) == ""
}

// ModelRuntimeDeploymentsInUse returns the model_runtime deployments that a
// decision signal or a decision algorithm references, in the top-level routing
// surface or any recipe, with defaults applied. Unreferenced deployments are
// never started.
func ModelRuntimeDeploymentsInUse(cfg *RouterConfig) map[string]ModelDeployment {
	used := make(map[string]ModelDeployment)
	if cfg == nil {
		return used
	}
	mark := func(name string) {
		if deployment, ok := cfg.ModelDeployments[name]; ok && deployment.IsModelRuntime() {
			used[name] = deployment.WithDefaults()
		}
	}
	scan := func(signals Signals, decisions []Decision) {
		for _, rule := range signals.DecisionRules {
			mark(rule.Deployment)
		}
		for _, decision := range decisions {
			if decision.Algorithm != nil && decision.Algorithm.Decision != nil &&
				strings.EqualFold(strings.TrimSpace(decision.Algorithm.Type), DecisionAlgorithmDecision) {
				mark(decision.Algorithm.Decision.Deployment)
			}
		}
	}
	scan(cfg.Signals, cfg.Decisions)
	for _, recipe := range cfg.Recipes {
		scan(recipe.Profile.Signals, recipe.Profile.Decisions)
	}
	return used
}

// ModelRuntimeProfiles lists the numerics profiles a model_runtime deployment may select.
func ModelRuntimeProfiles() []string {
	return append([]string(nil), modelRuntimeProfiles...)
}

func (d ModelDeployment) validateModelRuntime() error {
	if d.ExternalModel != "" {
		return fmt.Errorf("model_runtime deployments cannot set external_model")
	}
	if d.CustomOpsProfile != "" || d.CompilationCacheDir != "" {
		return fmt.Errorf("model_runtime deployments do not use ONNX Runtime custom ops or compilation caches")
	}
	if d.Precision != "native" {
		return fmt.Errorf("model_runtime deployments run the package's own dtype policy (precision native)")
	}
	if d.Input.Overflow != "reject" || d.Input.MaxTokens != 0 {
		return fmt.Errorf("decision models reject over-length input and never truncate; remove input")
	}
	if !modelRuntimeDevice.MatchString(d.Device) {
		return fmt.Errorf("device must be auto, cpu, cuda[:N] or rocm[:N]")
	}
	if !stringSliceContains(modelRuntimeProfiles, d.Profile) {
		return fmt.Errorf("profile must be one of %s", strings.Join(modelRuntimeProfiles, ", "))
	}
	if d.Revision != "" && !modelRuntimeRevision.MatchString(d.Revision) {
		return fmt.Errorf("revision must be a 40-hex commit")
	}
	if strings.TrimSpace(d.Endpoint) != "" {
		return validateModelRuntimeEndpoint(d.Endpoint)
	}
	artifact := strings.TrimSpace(d.Artifact)
	if artifact == "" {
		return fmt.Errorf("a managed model_runtime deployment requires artifact (a Hub repository or an absolute package path)")
	}
	if !filepath.IsAbs(artifact) && !hubRepositoryID.MatchString(artifact) {
		return fmt.Errorf("artifact must be a Hub repository ID or an absolute package path")
	}
	if filepath.IsAbs(artifact) && d.Revision != "" {
		return fmt.Errorf("revision applies only to Hub repositories")
	}
	return nil
}

func validateModelRuntimeEndpoint(endpoint string) error {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("endpoint: %w", err)
	}
	switch parsed.Scheme {
	case "unix":
		if !filepath.IsAbs(parsed.Path) || parsed.Host != "" {
			return fmt.Errorf("endpoint unix:// needs an absolute socket path (unix:///run/vllm-sr/runtime.sock)")
		}
	case "http", "https":
		if parsed.Host == "" {
			return fmt.Errorf("endpoint needs a host")
		}
	default:
		return fmt.Errorf("endpoint must use unix://, http:// or https://")
	}
	return nil
}
