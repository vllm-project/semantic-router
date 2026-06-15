package routingstrategies

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const valuesFile = "e2e/profiles/routing-strategies/values.yaml"

var resourceManifests = []string{
	"deploy/kubernetes/routing-strategies/aigw-resources/base-model.yaml",
	"deploy/kubernetes/routing-strategies/aigw-resources/gwapi-resources.yaml",
}

// Profile implements the Routing Strategies test profile.
type Profile struct {
	verbose         bool
	stack           *gatewaystack.Stack
	mcpStdioProcess *exec.Cmd
	mcpHTTPProcess  *exec.Cmd
}

// NewProfile creates a new Routing Strategies profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     "routing-strategies",
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "routing-strategies"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests different routing strategies including keyword-based routing"
}

// Setup deploys the shared gateway stack, then starts optional MCP servers.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose

	if err := p.stack.Setup(ctx, opts); err != nil {
		return err
	}

	if err := p.startMCPServers(ctx); err != nil {
		p.log("Warning: MCP servers not started: %v", err)
		p.log("MCP-related tests will be skipped")
	}

	return nil
}

// Teardown stops MCP servers, then tears down the shared gateway stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose

	if p.mcpStdioProcess != nil && p.mcpStdioProcess.Process != nil {
		_ = p.mcpStdioProcess.Process.Kill()
	}
	if p.mcpHTTPProcess != nil && p.mcpHTTPProcess.Process != nil {
		_ = p.mcpHTTPProcess.Process.Kill()
	}

	return p.stack.Teardown(ctx, opts)
}

// GetTestCases returns the list of test cases for this profile.
func (p *Profile) GetTestCases() []string {
	return []string{
		"keyword-routing",
		"entropy-routing",
		"routing-fallback",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[Routing-Strategies] "+format+"\n", args...)
	}
}

func (p *Profile) startMCPServers(ctx context.Context) error {
	p.log("Starting MCP classification servers")

	if _, err := exec.LookPath("python3"); err != nil {
		p.log("Warning: python3 not found, skipping MCP server startup")
		p.log("MCP tests will be skipped or may fail")
		return nil
	}

	p.log("Starting stdio MCP server (keyword-based)")
	p.mcpStdioProcess = exec.CommandContext(ctx,
		"python3",
		"deploy/examples/mcp-classifier-server/server_keyword.py")
	if p.verbose {
		p.mcpStdioProcess.Stdout = os.Stdout
		p.mcpStdioProcess.Stderr = os.Stderr
	}
	if err := p.mcpStdioProcess.Start(); err != nil {
		p.log("Warning: failed to start stdio MCP server: %v", err)
	} else {
		p.log("Stdio MCP server started (PID: %d)", p.mcpStdioProcess.Process.Pid)
	}

	p.log("Starting HTTP MCP server (embedding-based)")
	p.mcpHTTPProcess = exec.CommandContext(ctx,
		"python3",
		"deploy/examples/mcp-classifier-server/server_embedding.py",
		"--port", "8090")
	if p.verbose {
		p.mcpHTTPProcess.Stdout = os.Stdout
		p.mcpHTTPProcess.Stderr = os.Stderr
	}
	if err := p.mcpHTTPProcess.Start(); err != nil {
		p.log("Warning: failed to start HTTP MCP server: %v", err)
		if p.mcpStdioProcess == nil || p.mcpStdioProcess.Process == nil {
			return fmt.Errorf("failed to start any MCP servers: %w", err)
		}
		p.log("Continuing with only stdio MCP server")
	} else {
		p.log("HTTP MCP server started (PID: %d)", p.mcpHTTPProcess.Process.Pid)
	}

	p.log("Waiting for MCP servers to initialize...")
	time.Sleep(3 * time.Second)

	p.log("MCP servers started successfully")
	return nil
}
