package main

import (
	"context"
	"flag"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type runtimeManagedAPIStub struct{}

func (runtimeManagedAPIStub) Register(*http.ServeMux)     {}
func (runtimeManagedAPIStub) Ready(context.Context) error { return nil }
func (runtimeManagedAPIStub) Run(ctx context.Context) error {
	<-ctx.Done()
	return ctx.Err()
}

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

func TestStartAPIServerWaitsForListenerAndShutsDown(t *testing.T) {
	port := reserveRuntimeManagementPort(t)
	cfg := config.DefaultGlobalConfig()
	cfg.ManagementAPI.BindAddress = "127.0.0.1"
	cfg.ManagementAPI.Port = port
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	lifecycle, err := startAPIServerIfEnabled(ctx, runtimeOptions{
		enableAPI: true, apiPort: port, apiBind: "127.0.0.1", port: 50051, metricsPort: 9190,
	}, routerruntime.NewRegistry(&cfg), nil)
	if err != nil {
		t.Fatalf("startAPIServerIfEnabled() error = %v", err)
	}
	response, err := http.Get("http://" + net.JoinHostPort("127.0.0.1", portStringForRuntimeTest(port)) + "/health")
	if err != nil {
		t.Fatalf("listener was not accepting requests after startup returned: %v", err)
	}
	response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("listener health status = %d", response.StatusCode)
	}
	shutdownContext, stopShutdown := context.WithTimeout(context.Background(), 5*time.Second)
	defer stopShutdown()
	if err := lifecycle.Close(shutdownContext); err != nil {
		t.Fatalf("listener shutdown error = %v", err)
	}
}

func TestStartAPIServerRejectsInvalidManagedTLSBeforeReturning(t *testing.T) {
	directory := t.TempDir()
	certificateFile := filepath.Join(directory, "certificate.pem")
	privateKeyFile := filepath.Join(directory, "private-key.pem")
	if err := os.WriteFile(certificateFile, []byte("invalid certificate"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(privateKeyFile, []byte("invalid private key"), 0o600); err != nil {
		t.Fatal(err)
	}
	port := reserveRuntimeManagementPort(t)
	cfg := config.DefaultGlobalConfig()
	cfg.ControlPlane.Mode = config.ControlPlaneModeManaged
	cfg.ManagementAPI.BindAddress = "127.0.0.1"
	cfg.ManagementAPI.Port = port
	cfg.ManagementAPI.Auth.Mode = config.ManagementAuthModeRouter
	cfg.ManagementAPI.Auth.Roles = nil
	cfg.ManagementAPI.Auth.TokenSigningKeyringFile = "/unused/management-signing"
	cfg.ManagementAPI.Auth.ServiceAccountHMACKeyringFile = "/unused/service-account-hmac"
	cfg.ManagementAPI.Auth.InvitationHMACKeyringFile = "/unused/invitation-hmac"
	cfg.ManagementAPI.Auth.ControlPlaneHMACKeyringFile = "/unused/control-plane-hmac"
	cfg.ManagementAPI.Auth.ResponseKEKKeyringFile = "/unused/response-kek"
	cfg.ManagementAPI.TLS.CertificateFile = certificateFile
	cfg.ManagementAPI.TLS.PrivateKeyFile = privateKeyFile
	_, err := startAPIServerIfEnabled(context.Background(), runtimeOptions{
		enableAPI: true, apiPort: port, apiBind: "127.0.0.1", port: 50051, metricsPort: 9190,
	}, routerruntime.NewRegistry(&cfg), runtimeManagedAPIStub{})
	if err == nil || !strings.Contains(err.Error(), "invalid or do not match") {
		t.Fatalf("invalid managed TLS startup error = %v", err)
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

func reserveRuntimeManagementPort(t *testing.T) int {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	return listener.Addr().(*net.TCPAddr).Port
}

func portStringForRuntimeTest(port int) string {
	return strconv.Itoa(port)
}
