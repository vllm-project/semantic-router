package remoteembedding

import (
	"context"
	"fmt"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	valuesFile        = "e2e/profiles/remote-embedding/values.yaml"
	mockManifest      = "e2e/profiles/remote-embedding/manifests/mock-embedding.yaml"
	mockNamespace     = "default"
	mockDeployment    = "mock-embedding"
	mockReadyTimeout  = 5 * time.Minute
	mockReadyInterval = 2 * time.Second
)

var gatewayResources = []string{
	"deploy/kubernetes/routing-strategies/aigw-resources/base-model.yaml",
	"deploy/kubernetes/routing-strategies/aigw-resources/gwapi-resources.yaml",
}

// Profile validates text embedding routing through an external OpenAI-compatible provider.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates the remote embedding E2E profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "remote-embedding",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        gatewayResources,
			WaitDeployments: []helpers.DeploymentRef{
				{Namespace: "default", Name: "vllm-llama3-8b-instruct"},
			},
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "remote-embedding"
}

// Description returns the profile contract.
func (p *Profile) Description() string {
	return "Tests OpenAI-compatible remote embedding provider startup and deterministic text embedding routing"
}

// Setup deploys the mock provider before the router because startup probes call it immediately.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.stack.ApplyPrerequisites(ctx, opts); err != nil {
		return err
	}
	if err := applyManifest(ctx, opts.KubeConfig, "apply", mockManifest); err != nil {
		return fmt.Errorf("apply mock embedding provider: %w", err)
	}
	if err := waitForMockProvider(ctx, opts); err != nil {
		return err
	}
	if err := p.stack.DeployCore(ctx, opts); err != nil {
		return err
	}
	if err := p.stack.ApplyResources(ctx, opts); err != nil {
		return err
	}
	return p.stack.Verify(ctx, opts)
}

// Teardown removes gateway resources, the mock provider, and shared releases.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	if err := p.stack.CleanupResources(ctx, opts); err != nil && opts.Verbose {
		fmt.Printf("[remote-embedding] cleanup resources: %v\n", err)
	}
	p.stack.UninstallCore(ctx, opts)
	if err := applyManifest(ctx, opts.KubeConfig, "delete", mockManifest, "--ignore-not-found=true"); err != nil && opts.Verbose {
		fmt.Printf("[remote-embedding] cleanup mock provider: %v\n", err)
	}
	return nil
}

// GetTestCases returns the strict remote embedding contract test.
func (p *Profile) GetTestCases() []string {
	return []string{"remote-embedding-routing"}
}

// GetServiceConfig returns the shared gateway service configuration.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}

func waitForMockProvider(ctx context.Context, opts *framework.SetupOptions) error {
	if opts.KubeClient == nil {
		return fmt.Errorf("kube client is required to verify the mock embedding provider")
	}
	deadline := time.Now().Add(mockReadyTimeout)
	for time.Now().Before(deadline) {
		if err := helpers.VerifyServicePodsRunning(ctx, opts.KubeClient, mockNamespace, mockDeployment, opts.Verbose); err == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(mockReadyInterval):
		}
	}
	return fmt.Errorf("mock embedding provider did not become ready within %s", mockReadyTimeout)
}

func applyManifest(ctx context.Context, kubeConfig string, action string, manifest string, extraArgs ...string) error {
	args := []string{action, "-f", manifest}
	args = append(args, extraArgs...)
	args = append(args, "--kubeconfig", kubeConfig)
	return exec.CommandContext(ctx, "kubectl", args...).Run()
}
