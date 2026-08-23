package managementcomposition

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managedruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

type backgroundWorkerFunc func(context.Context) error

func (function backgroundWorkerFunc) Run(ctx context.Context) error { return function(ctx) }

func TestNewFactoryRequiresManagedRouterAuthentication(t *testing.T) {
	for _, test := range []struct {
		name   string
		mutate func(*config.RouterConfig)
	}{
		{name: "standalone", mutate: func(cfg *config.RouterConfig) {
			cfg.ControlPlane.Mode = config.ControlPlaneModeStandalone
		}},
		{name: "legacy bearer", mutate: func(cfg *config.RouterConfig) {
			cfg.ManagementAPI.Auth.Mode = config.ManagementAuthModeBearer
		}},
		{name: "static role", mutate: func(cfg *config.RouterConfig) {
			cfg.ManagementAPI.Auth.Roles = config.DefaultManagementAPIRoles()
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := managedFactoryConfig()
			test.mutate(&cfg)
			if _, err := NewFactory(&cfg, Options{}); err == nil {
				t.Fatal("NewFactory() should fail closed")
			}
		})
	}
}

func TestNewFactoryRejectsPartialIdentityVerifierOverrides(t *testing.T) {
	cfg := managedFactoryConfig()
	if _, err := NewFactory(&cfg, Options{
		AssertionVerifier: managementauth.DenyAllSubjectAssertionVerifier{},
	}); err == nil || !strings.Contains(err.Error(), "complete set") {
		t.Fatalf("NewFactory() error = %v, want partial override rejection", err)
	}
}

func TestNewFactoryDefersProductionAssertionVerifierComposition(t *testing.T) {
	cfg := managedFactoryConfig()
	factory, err := NewFactory(&cfg, Options{})
	if err != nil || factory == nil || factory.assertionVerifier != nil {
		t.Fatalf("NewFactory() = %#v, %v", factory, err)
	}
}

func TestNewFactoryCapturesCredentialRevealPolicy(t *testing.T) {
	cfg := managedFactoryConfig()
	cfg.Access.Credentials.Reveal.Enabled = true
	factory, err := NewFactory(&cfg, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if !factory.defaultRevealable {
		t.Fatal("managed API-key reveal policy was not captured from Router configuration")
	}

	// Factory construction must own a value copy. Later caller mutations cannot
	// silently change the Management API's credential-retention policy.
	cfg.Access.Credentials.Reveal.Enabled = false
	if !factory.defaultRevealable {
		t.Fatal("managed API-key reveal policy changed after factory construction")
	}
}

func TestNewFactoryCapturesUsageBacklogAdmissionLimit(t *testing.T) {
	cfg := managedFactoryConfig()
	cfg.Access.Enforcement.MaxUsageBacklog = 321
	factory, err := NewFactory(&cfg, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if factory.maxUsageBacklog != 321 {
		t.Fatalf("max usage backlog = %d, want 321", factory.maxUsageBacklog)
	}
	cfg.Access.Enforcement.MaxUsageBacklog = 999
	if factory.maxUsageBacklog != 321 {
		t.Fatal("runtime diagnostics backlog limit changed after factory construction")
	}
}

func TestNewFactoryUsesOnlyThePublicInferenceFrontDoor(t *testing.T) {
	cfg := managedFactoryConfig()
	cfg.Looper.Endpoint = "http://physical-backend.internal/v1"
	factory, err := NewFactory(&cfg, Options{})
	if err != nil {
		t.Fatal(err)
	}
	if factory.agentInferenceEndpoint != cfg.Agent.PublicInferenceEndpoint {
		t.Fatalf("Agent endpoint = %q, want public inference endpoint %q", factory.agentInferenceEndpoint, cfg.Agent.PublicInferenceEndpoint)
	}
}

func TestNewFactoryRejectsMissingAgentPublicInferenceEndpoint(t *testing.T) {
	cfg := managedFactoryConfig()
	cfg.Agent.PublicInferenceEndpoint = ""
	if _, err := NewFactory(&cfg, Options{}); err == nil || !strings.Contains(err.Error(), "public inference endpoint") {
		t.Fatalf("NewFactory() error = %v, want public inference endpoint rejection", err)
	}
}

func TestFactoryBuildRejectsIncompleteProcessDependencies(t *testing.T) {
	cfg := managedFactoryConfig()
	factory, err := NewFactory(&cfg, Options{})
	if err != nil {
		t.Fatal(err)
	}
	_, err = factory.Build(context.Background(), managedruntime.ManagementDependencies{})
	if err == nil || !strings.Contains(err.Error(), "dependencies are incomplete") {
		t.Fatalf("Build() error = %v", err)
	}
}

func TestApplicationRunTreatsUnexpectedCleanWorkerExitAsFailure(t *testing.T) {
	application := &application{workers: []backgroundWorker{
		backgroundWorkerFunc(func(context.Context) error { return nil }),
		backgroundWorkerFunc(func(ctx context.Context) error {
			<-ctx.Done()
			return ctx.Err()
		}),
	}}
	done := make(chan error, 1)
	go func() { done <- application.Run(context.Background()) }()
	select {
	case err := <-done:
		if err == nil || errors.Is(err, context.Canceled) ||
			!strings.Contains(err.Error(), "exited before cancellation") {
			t.Fatalf("Run() error = %v, want unexpected worker-exit failure", err)
		}
	case <-time.After(time.Second):
		t.Fatal("Run() did not cancel sibling workers after a clean exit")
	}
}

func managedFactoryConfig() config.RouterConfig {
	cfg := config.DefaultGlobalConfig()
	cfg.ControlPlane.Mode = config.ControlPlaneModeManaged
	cfg.Agent.PublicInferenceEndpoint = "http://public-inference.internal/v1/chat/completions"
	cfg.ManagementAPI.Auth.Mode = config.ManagementAuthModeRouter
	cfg.ManagementAPI.Auth.Roles = nil
	cfg.ManagementAPI.Auth.Tokens = nil
	cfg.AccessRuntimeStore = &config.AccessRuntimeStoreConfig{
		Type:  config.AccessRuntimeStoreTypeRedis,
		Redis: config.RedisAccessRuntimeStoreConfig{KeyPrefix: "vllm-sr:test:"},
	}
	return cfg
}
