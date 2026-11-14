package responseapishared

import (
	"context"
	"fmt"
	"os/exec"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
)

var sharedResourceManifests = []string{
	"deploy/kubernetes/response-api/provider-mocker.yaml",
	"deploy/kubernetes/response-api/gwapi-resources.yaml",
}

// RedisProfile implements the shared setup for Redis-backed Response API profiles.
type RedisProfile struct {
	name        string
	description string
	redisReady  []string
	stack       *gatewaystack.Stack
}

// NewRedisProfile constructs a shared Redis-backed Response API profile.
// redisReady is the `kubectl wait` condition and resource that mean Redis
// accepts connections, such as {"--for=condition=available", "deployment/redis"}.
func NewRedisProfile(name, description, valuesFile, redisManifest string, redisReady ...string) *RedisProfile {
	return &RedisProfile{
		name:        name,
		description: description,
		redisReady:  redisReady,
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     name,
			SemanticRouterValuesFile: valuesFile,
			PrerequisiteManifests:    []string{redisManifest},
			ResourceManifests:        sharedResourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *RedisProfile) Name() string {
	return p.name
}

// Description returns the profile description.
func (p *RedisProfile) Description() string {
	return p.description
}

// Setup deploys Redis and waits until it is ready before the Router starts:
// the Router opens its Response API store once, at startup, and answers 503
// for the rest of its life when Redis was not there yet.
func (p *RedisProfile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.stack.ApplyPrerequisites(ctx, opts); err != nil {
		return err
	}
	args := append([]string{"--kubeconfig", opts.KubeConfig, "-n", "default", "wait", "--timeout=5m"}, p.redisReady...)
	if output, err := exec.CommandContext(ctx, "kubectl", args...).CombinedOutput(); err != nil {
		return fmt.Errorf("wait for Redis: %w: %s", err, output)
	}
	if err := p.stack.DeployCore(ctx, opts); err != nil {
		return err
	}
	if err := p.stack.ApplyResources(ctx, opts); err != nil {
		return err
	}
	return p.stack.Verify(ctx, opts)
}

// Teardown removes the shared gateway stack and Redis prerequisite.
func (p *RedisProfile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *RedisProfile) GetTestCases() []string {
	return []string{
		"response-api-create",
		"response-api-get",
		"response-api-delete",
		"response-api-input-items",
		"response-api-conversation-chaining",
		"response-api-edge-empty-input",
		"response-api-edge-large-input",
		"response-api-edge-special-characters",
		"response-api-edge-concurrent-requests",
		"response-api-ttl-expiry",
		"response-api-restart-recovery",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *RedisProfile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
