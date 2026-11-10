//go:build !windows

package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"math/rand/v2"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
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

func startProfilingForTest(t *testing.T, cfg *config.RouterConfig, opts runtimeOptions) []func(context.Context) error {
	t.Helper()
	hooks := make([]func(context.Context) error, 0)
	t.Cleanup(func() {
		if err := runShutdownHooks(context.Background(), &hooks); err != nil {
			t.Errorf("shutdown profiling server: %v", err)
		}
	})
	startProfilingServerIfEnabled(cfg, opts, &hooks)
	return hooks
}

// givenPorts are the ports freePort has returned in this test binary.
var givenPorts sync.Map

// freePort returns a port that is free on every address and that freePort
// has not returned before. The Router binds it later, so it comes from
// outside the kernel's ephemeral range: a released ephemeral port can be
// given to any other socket on the host, such as another test package's
// listener or outgoing connection, before the Router binds it.
func freePort(t *testing.T) int {
	t.Helper()
	low, high := nonEphemeralPorts()
	for range 100 {
		port := low + rand.IntN(high-low+1)
		if _, given := givenPorts.LoadOrStore(port, struct{}{}); given {
			continue
		}
		listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			continue
		}
		if err := listener.Close(); err != nil {
			t.Fatalf("failed to release the reserved port: %v", err)
		}
		return port
	}
	t.Fatalf("no free port between %d and %d", low, high)
	return 0
}

// nonEphemeralPorts is a port range the kernel does not assign to sockets
// on its own: above Linux's ip_local_port_range, or below the IANA dynamic
// range that other systems assign from.
func nonEphemeralPorts() (low, high int) {
	if data, err := os.ReadFile("/proc/sys/net/ipv4/ip_local_port_range"); err == nil {
		var first, last int
		if _, err := fmt.Sscan(string(data), &first, &last); err == nil && last < 64536 {
			return last + 1, 65535
		}
	}
	return 32768, 49151
}

func TestRunShutdownHooksCompletesInRegistrationOrder(t *testing.T) {
	completed := make([]string, 0, 2)
	hooks := []func(context.Context) error{
		func(context.Context) error {
			completed = append(completed, "replay-and-vector-stores")
			return nil
		},
		func(context.Context) error {
			completed = append(completed, "runtime-resources")
			return nil
		},
	}

	if err := runShutdownHooks(context.Background(), &hooks); err != nil {
		t.Fatalf("runShutdownHooks() error = %v", err)
	}

	want := []string{"replay-and-vector-stores", "runtime-resources"}
	if !reflect.DeepEqual(completed, want) {
		t.Fatalf("shutdown hook order = %v, want %v", completed, want)
	}
}

// Exercise the actual startup loader in a subprocess because invalid config
// terminates the process. Returning from this boundary would allow startup to
// create registries, servers, model runtimes, and background workers.
func TestStartupRejectsInvalidKubernetesGlobals(t *testing.T) {
	const configPathEnv = "VSR_TEST_STARTUP_VALIDATION_CONFIG"
	if path := os.Getenv(configPathEnv); path != "" {
		if _, err := logging.InitLoggerFromEnv(); err != nil {
			t.Fatal(err)
		}
		loadRuntimeConfigOrFatal(path)
		os.Exit(0)
	}

	cases := []struct {
		name   string
		global string
		want   string
	}{
		{"cache", "  plugins: [{type: response_cache, configuration: {enabled: true, semantic: {match: {recall: {top_k: 1, min_similarity: 1.5}}}}}]", "min_similarity"},
		{"memory", "  stores: {memory: {default_similarity_threshold: 1.5}}", "default_similarity_threshold"},
		{"admission", "  model_catalog: {admission: {primary: {max_concurrency: 1, max_queue: -1}}}", "max_queue"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "config.yaml")
			raw := "version: v0.5\nglobal:\n  router: {config_source: kubernetes}\n" + tc.global + "\n"
			if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			defer cancel()
			//nolint:gosec // G204: runs the current Go test binary with a fixed test selector.
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestStartupRejectsInvalidKubernetesGlobals$")
			cmd.Env = append(os.Environ(), configPathEnv+"="+path)
			output, err := cmd.CombinedOutput()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
				t.Fatalf("startup loader must exit with code 1, got %v\n%s", err, output)
			}
			for _, want := range []string{"runtime_config_load_failed", tc.want} {
				if !strings.Contains(string(output), want) {
					t.Errorf("startup output must contain %q:\n%s", want, output)
				}
			}
		})
	}
}
