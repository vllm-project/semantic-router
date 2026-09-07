package main

import (
	"flag"
	"net"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBoolFlagOverrideOnlyWhenExplicitlySet(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	value := fs.Bool("management-remote-exposure", false, "")
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	if got := boolFlagOverride(fs, "management-remote-exposure", *value); got != nil {
		t.Fatalf("unset flag override = %v, want nil", *got)
	}

	fs = flag.NewFlagSet("test", flag.ContinueOnError)
	value = fs.Bool("management-remote-exposure", false, "")
	if err := fs.Parse([]string{"-management-remote-exposure=true"}); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	got := boolFlagOverride(fs, "management-remote-exposure", *value)
	if got == nil || !*got {
		t.Fatalf("explicit true override = %v, want true", got)
	}

	fs = flag.NewFlagSet("test", flag.ContinueOnError)
	value = fs.Bool("management-remote-exposure", false, "")
	if err := fs.Parse([]string{"-management-remote-exposure=false"}); err != nil {
		t.Fatalf("Parse() error = %v", err)
	}
	got = boolFlagOverride(fs, "management-remote-exposure", *value)
	if got == nil || *got {
		t.Fatalf("explicit false override = %v, want false", got)
	}
}

func TestResolveRuntimeManagementOptionsUsesConfigListener(t *testing.T) {
	cfg := &config.RouterConfig{ManagementAPI: config.ManagementAPIConfig{
		BindAddress: "0.0.0.0",
		Port:        9090,
		Auth:        config.ManagementAPIAuthConfig{Mode: config.ManagementAuthModeDisabled},
	}}
	t.Setenv(config.ManagementInternalListenerEnv, "true")
	resolved, err := resolveRuntimeManagementOptions(runtimeOptions{enableAPI: true}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.apiBind != "0.0.0.0" || resolved.apiPort != 9090 || resolved.managementAuthMode != config.ManagementAuthModeDisabled {
		t.Fatalf("resolved options = %#v", resolved)
	}
}

func TestResolveRuntimeManagementOptionsRejectsInvalidConfiguredPort(t *testing.T) {
	cfg := &config.RouterConfig{ManagementAPI: config.ManagementAPIConfig{
		BindAddress: "127.0.0.1",
		Port:        70000,
		Auth:        config.ManagementAPIAuthConfig{Mode: config.ManagementAuthModeDisabled},
	}}
	if _, err := resolveRuntimeManagementOptions(runtimeOptions{enableAPI: true}, cfg); err == nil {
		t.Fatal("invalid management port must fail before the startup goroutine")
	}
}

func TestResolveRuntimeManagementOptionsRejectsRouterServicePortConflict(t *testing.T) {
	cfg := &config.RouterConfig{ManagementAPI: config.ManagementAPIConfig{
		BindAddress: "127.0.0.1",
		Port:        50051,
		Auth:        config.ManagementAPIAuthConfig{Mode: config.ManagementAuthModeDisabled},
	}}
	if _, err := resolveRuntimeManagementOptions(runtimeOptions{
		enableAPI:   true,
		port:        50051,
		metricsPort: 9190,
	}, cfg); err == nil {
		t.Fatal("management port conflict must fail before startup")
	}
}

func TestStartProfilingServerKeepsExplicitPortZeroEphemeral(t *testing.T) {
	cfg := &config.RouterConfig{}
	cfg.Observability.Profiling = config.ProfilingConfig{Enabled: true, Port: 0, Bind: "127.0.0.1"}

	// Reserving the default profiling port means a 0 -> 6060 rewrite would be
	// rejected as a service conflict instead of taking an ephemeral port.
	hooks := startProfilingForTest(t, cfg, runtimeOptions{port: config.DefaultProfilingPort})
	if len(hooks) != 1 {
		t.Fatalf("profiling shutdown hooks = %d, want 1", len(hooks))
	}
}

func TestStartProfilingServerReusesMetricsPortWhenMetricsDisabled(t *testing.T) {
	metricsPort := freePort(t)
	cfg := &config.RouterConfig{}
	metricsDisabled := false
	cfg.Observability.Metrics.Enabled = &metricsDisabled
	cfg.Observability.Profiling = config.ProfilingConfig{Enabled: true, Port: metricsPort, Bind: "127.0.0.1"}

	hooks := startProfilingForTest(t, cfg, runtimeOptions{port: 50051, metricsPort: metricsPort})
	if len(hooks) != 1 {
		t.Fatalf("profiling shutdown hooks = %d, want 1 when the metrics server is disabled", len(hooks))
	}
}

func TestStartProfilingServerRejectsLiveMetricsPort(t *testing.T) {
	metricsPort := freePort(t)
	cfg := &config.RouterConfig{}
	cfg.Observability.Profiling = config.ProfilingConfig{Enabled: true, Port: metricsPort, Bind: "127.0.0.1"}

	hooks := startProfilingForTest(t, cfg, runtimeOptions{port: 50051, metricsPort: metricsPort})
	if len(hooks) != 0 {
		t.Fatalf("profiling shutdown hooks = %d, want 0 when the port collides with the metrics server", len(hooks))
	}
}

func startProfilingForTest(t *testing.T, cfg *config.RouterConfig, opts runtimeOptions) []func() {
	t.Helper()
	hooks := make([]func(), 0)
	t.Cleanup(func() { runShutdownHooks(&hooks) })
	startProfilingServerIfEnabled(cfg, opts, &hooks)
	return hooks
}

func freePort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to reserve a free port: %v", err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatalf("failed to release the reserved port: %v", err)
	}
	return port
}

func TestRunShutdownHooksCompletesInRegistrationOrder(t *testing.T) {
	completed := make([]string, 0, 2)
	hooks := []func(){
		func() { completed = append(completed, "replay-and-vector-stores") },
		func() { completed = append(completed, "runtime-resources") },
	}

	runShutdownHooks(&hooks)

	want := []string{"replay-and-vector-stores", "runtime-resources"}
	if !reflect.DeepEqual(completed, want) {
		t.Fatalf("shutdown hook order = %v, want %v", completed, want)
	}
}
