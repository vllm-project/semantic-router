package stickytoolselectionredis

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"
	aigateway "github.com/vllm-project/semantic-router/e2e/profiles/ai-gateway"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const redisManifest = "deploy/kubernetes/response-api/redis.yaml"

// Profile isolates the destructive Redis availability contract from the
// baseline AI Gateway matrix while reusing its values and gateway resources.
type Profile struct {
	stack *gatewaystack.Stack
}

// NewProfile creates the Redis-backed sticky tool-selection profile.
func NewProfile() *Profile {
	stackConfig := aigateway.BaseStackConfig("sticky-tool-selection-redis")
	stackConfig.PrerequisiteManifests = []string{redisManifest}
	stackConfig.WaitDeployments = append(stackConfig.WaitDeployments, helpers.DeploymentRef{
		Namespace: "default",
		Name:      "redis",
	})
	stackConfig.SemanticRouterSet = map[string]string{
		"config.global.stores.tool_sessions.backend":       "redis",
		"config.global.stores.tool_sessions.ttl_seconds":   "900",
		"config.global.stores.tool_sessions.redis.address": "redis.default.svc.cluster.local:6379",
	}
	return &Profile{stack: gatewaystack.New(stackConfig)}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "sticky-tool-selection-redis"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests Redis-backed sticky tool selection across restart and store unavailability"
}

// Setup deploys the shared gateway stack and Redis prerequisite.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.Setup(ctx, opts)
}

// Teardown removes the shared gateway stack and Redis prerequisite.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return append([]string(nil), testmatrix.StickyToolSelectionRedisContract...)
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}
