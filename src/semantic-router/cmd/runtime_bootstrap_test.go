package main

import (
	"flag"
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
