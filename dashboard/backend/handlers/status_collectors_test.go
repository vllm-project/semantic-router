package handlers

import (
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
	"github.com/vllm-project/semantic-router/dashboard/backend/statusstore"
)

func closedHTTPAddress(t *testing.T) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen on unused port: %v", err)
	}
	address := "http://" + listener.Addr().String()
	if err := listener.Close(); err != nil {
		t.Fatalf("close listener: %v", err)
	}
	return address
}

func answeringRouter(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health", "/v1/models":
			w.WriteHeader(http.StatusOK)
		case "/startup-status":
			_, _ = w.Write([]byte(`{"phase":"ready","ready":true}`))
		case "/api/v1/inventory/models":
			_, _ = w.Write([]byte(`{"models":[],"summary":{}}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func writeRuntimeConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "runtime-config.yaml")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestManagedStackStatusReportsServicesThatAnswerTheirProbes(t *testing.T) {
	invocations := trapContainerCLIs(t)
	t.Setenv("VLLM_SR_GATEWAY", "extproc")
	router := answeringRouter(t)
	envoy := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) }))
	t.Cleanup(envoy.Close)

	status := collectManagedStackStatus("", StackState{}, router.URL, envoy.URL)
	if status.Overall != "healthy" || status.DeploymentType != "docker" {
		t.Fatalf("status = %q (%q), want healthy docker: %+v", status.Overall, status.DeploymentType, status.Services)
	}
	var names []string
	for _, service := range status.Services {
		names = append(names, service.Name)
		if !service.Healthy || service.Status != "running" {
			t.Errorf("service %+v, want running", service)
		}
	}
	if want := []string{"Routing access", "Router", "Envoy", "Dashboard"}; !reflect.DeepEqual(names, want) {
		t.Fatalf("services = %v, want %v", names, want)
	}
	if calls := invocations(); len(calls) != 0 {
		t.Fatalf("status ran a container CLI: %q", calls)
	}
}

func TestManagedStandaloneStackStatusReportsNoEnvoy(t *testing.T) {
	t.Setenv("VLLM_SR_GATEWAY", "standalone")
	router := answeringRouter(t)

	status := collectManagedStackStatus("", StackState{}, router.URL, router.URL)
	var names []string
	for _, service := range status.Services {
		names = append(names, service.Name)
	}
	if want := []string{"Routing access", "Router", "Dashboard"}; !reflect.DeepEqual(names, want) {
		t.Fatalf("services = %v, want %v", names, want)
	}
	if status.Overall != "healthy" || !status.Services[0].Healthy {
		t.Fatalf("routing access through the Router's listener = %+v", status.Services[0])
	}
}

// Without a container runtime to ask, a service that does not answer reports
// what the stack's files say: it waits for setup, `vllm-sr serve` is starting
// it, or nothing runs it. A stack the CLI did not create, such as a Kubernetes
// deployment, gets no `vllm-sr serve` hint.
func TestManagedStackStatusTellsStoppedServicesApart(t *testing.T) {
	setupConfig := "version: v0.3\nsetup:\n  mode: true\n"
	activatedConfig := "version: v0.3\nlisteners: []\n"
	for _, test := range []struct {
		name, config, heartbeat            string
		staleHeartbeat, pending, notServed bool
		status, message                    string
	}{
		{name: "setup", config: setupConfig, heartbeat: `{"pid": 1, "state": "waiting"}`, status: "standby", message: "Waiting for setup: activate a config in the Dashboard"},
		{name: "setup_without_cli", config: setupConfig, status: "standby", message: "Waiting for setup: activate a config in the Dashboard"},
		{name: "serve_starting", config: activatedConfig, heartbeat: `{"pid": 1, "state": "starting"}`, status: "starting", message: "Starting: `vllm-sr serve` is starting it"},
		{name: "serve_applying_setup", config: activatedConfig, heartbeat: `{"pid": 1, "state": "waiting"}`, pending: true, status: "starting", message: "Starting: `vllm-sr serve` is starting it"},
		{name: "saved_for_serve", config: activatedConfig, pending: true, status: "not running", message: "Not running: run `vllm-sr serve` to apply the saved configuration"},
		{name: "serve_gone", config: activatedConfig, heartbeat: `{"pid": 1, "state": "starting"}`, staleHeartbeat: true, status: "not running", message: "Not running: run `vllm-sr serve`"},
		{name: "stopped", config: activatedConfig, status: "not running", message: "Not running: run `vllm-sr serve`"},
		{name: "not_served_by_cli", config: activatedConfig, heartbeat: `{"pid": 1, "state": "starting"}`, pending: true, notServed: true, status: "not running", message: "Not running"},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("VLLM_SR_GATEWAY", "extproc")
			t.Setenv("TARGET_ENVOY_ADMIN_URL", "")
			t.Setenv("TARGET_ENVOY_URL", closedHTTPAddress(t))
			configPath := writeRuntimeConfig(t, test.config)
			if test.notServed {
				t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", "")
			} else {
				t.Setenv("VLLM_SR_RUNTIME_CONFIG_PATH", configPath)
			}
			if test.heartbeat != "" {
				heartbeat := pendingActivationPath(configPath, serveHeartbeatSuffix)
				if err := os.WriteFile(heartbeat, []byte(test.heartbeat), 0o644); err != nil {
					t.Fatal(err)
				}
				if test.staleHeartbeat {
					old := time.Now().Add(-2 * serveHeartbeatFreshness)
					if err := os.Chtimes(heartbeat, old, old); err != nil {
						t.Fatal(err)
					}
				}
			}
			if test.pending {
				if err := recordPendingActivation(configPath, []byte(test.config), activationReasonSetup, ""); err != nil {
					t.Fatal(err)
				}
			}
			stack := StackState{ConfigPath: configPath, Setup: setupmode.New(configPath, false)}
			stopped := closedHTTPAddress(t)

			status := collectManagedStackStatus("", stack, stopped, stopped)
			if status.Overall != "degraded" {
				t.Errorf("overall = %q, want degraded", status.Overall)
			}
			wantHistory := statusstore.StateUnavailable
			if test.status == "starting" {
				wantHistory = statusstore.StateStarting
			}
			for _, service := range status.Services[1:3] {
				if service.Healthy || service.Status != test.status || service.Message != test.message {
					t.Errorf("%s = %+v, want %q: %q", service.Name, service, test.status, test.message)
				}
				if got := statusObservations([]ServiceStatus{service})[0].State; got != wantHistory {
					t.Errorf("%s history = %q, want %q", service.Name, got, wantHistory)
				}
			}
		})
	}
}

func TestCollectHostStatusReportsDashboardWhenRouterIsUnavailable(t *testing.T) {
	status := collectHostStatus("", closedHTTPAddress(t), "")
	if status.Overall != "not_running" {
		t.Fatalf("overall status = %q, want not_running", status.Overall)
	}
	if len(status.Services) != 3 {
		t.Fatalf("service count = %d, want 3 (%#v)", len(status.Services), status.Services)
	}

	routingAccess := status.Services[0]
	if routingAccess.Name != "Routing access" || routingAccess.Healthy || routingAccess.Status != "unavailable" {
		t.Fatalf("routing access service = %#v, want unavailable", routingAccess)
	}

	router := status.Services[1]
	if router.Name != "Router" || router.Healthy || router.Status != "not running" {
		t.Fatalf("router service = %#v, want Router not running and unhealthy", router)
	}

	dashboard := status.Services[2]
	if dashboard.Name != "Dashboard" || !dashboard.Healthy || dashboard.Status != "running" {
		t.Fatalf("dashboard service = %#v, want Dashboard running and healthy", dashboard)
	}
}
