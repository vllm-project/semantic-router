// Package modelruntime provides the E2E profile for the built-in model runtime:
// runtimes the Router manages and one it attaches to, on tiny random-weight
// fixture packages, with startup readiness, every task binding, decision
// signals, the decision selector, request bundles, supervision and fail-open
// asserted against the runtimes' own answers.
package modelruntime

import (
	"context"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	runtimeclient "github.com/vllm-project/semantic-router/e2e/pkg/modelruntime"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	profileDir = "e2e/profiles/model-runtime/"
	valuesFile = profileDir + "values.yaml"
	// FilesConfigMap holds the fixture script and the attached models file;
	// the Router pod and the attached runtime mount it at /opt/vsr-e2e.
	FilesConfigMap = "model-runtime-e2e"
)

// configMapFiles are the profile files the runtimes read, by ConfigMap key.
var configMapFiles = map[string]string{
	"runtime_with_fixtures.py": profileDir + "runtime_with_fixtures.py",
	"attached-models.yaml":     profileDir + "attached-models.yaml",
}

// The attached runtime starts before the Router so that its deployments can
// become ready while the Router prepares its generation.
var prerequisiteManifests = []string{
	profileDir + "manifests/attached-runtime.yaml",
}

var resourceManifests = []string{
	"e2e/profiles/ai-gateway/gateway-resources/backend.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	"e2e/profiles/ai-gateway/gateway-resources/responses-route.yaml",
}

// Profile validates the Router with managed and attached model runtimes.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates the model-runtime profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "model-runtime",
			SemanticRouterValuesFile: valuesFile,
			PrerequisiteManifests:    prerequisiteManifests,
			ResourceManifests:        resourceManifests,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: runtimeclient.RouterNamespace, Name: "model-runtime-attached"},
				{Namespace: "default", Name: "vllm-llama3-8b-instruct"},
			},
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string { return "model-runtime" }

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests managed and attached model runtimes: startup readiness, task bindings, decision signals and selection, bundles, supervision and fail-open"
}

// Setup publishes the runtimes' files, then deploys the gateway stack.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	data := make(map[string]string, len(configMapFiles))
	for key, path := range configMapFiles {
		content, err := os.ReadFile(path)
		if err != nil {
			return fmt.Errorf("read %s: %w", path, err)
		}
		data[key] = string(content)
	}
	if err := runtimeclient.ApplyConfigMap(ctx, opts.KubeClient, runtimeclient.RouterNamespace, FilesConfigMap, data); err != nil {
		return err
	}
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the gateway stack and the runtimes' files.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	err := p.stack.Teardown(ctx, opts)
	if opts.KubeClient != nil {
		if cleanup := runtimeclient.DeleteConfigMap(ctx, opts.KubeClient, runtimeclient.RouterNamespace, FilesConfigMap); cleanup != nil && opts.Verbose {
			fmt.Printf("[model-runtime] Warning: failed to delete %s: %v\n", FilesConfigMap, cleanup)
		}
	}
	return err
}

// GetTestCases returns the profile's contracts. Startup readiness runs first,
// before cases that restart runtimes add readiness changes to the log.
func (p *Profile) GetTestCases() []string {
	return []string{
		"model-runtime-startup-readiness",
		"model-runtime-lifecycle",
		"model-runtime-task-signals",
		"model-runtime-embeddings-rerank",
		"decision-runtime-routing",
		"decision-runtime-set-span",
		"decision-prior-user-turns",
		"model-runtime-bundles",
		"model-runtime-long-history",
		"model-runtime-fail-open",
		"model-runtime-supervision",
		"model-runtime-load-retry",
		"model-runtime-load-retry-isolation",
	}
}

// GetServiceConfig returns the gateway access configuration.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
