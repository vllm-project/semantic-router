package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

func TestResolveCapabilityReportsFirstFailingDimension(t *testing.T) {
	failing := fail("code", "fix it")
	for _, test := range []struct {
		name  string
		facts capabilityFacts
		want  CapabilityState
	}{
		{"all pass", capabilityFacts{enabled: pass(), configured: pass(), authorized: pass(), reachable: pass(), healthy: pass()}, CapabilityAvailable},
		{"disabled wins", capabilityFacts{enabled: failing, configured: failing, authorized: failing, reachable: failing, healthy: failing}, CapabilityDisabled},
		{"unconfigured before unauthorized", capabilityFacts{enabled: pass(), configured: failing, authorized: failing, reachable: failing, healthy: failing}, CapabilityUnconfigured},
		{"unauthorized before unreachable", capabilityFacts{enabled: pass(), configured: pass(), authorized: failing, reachable: failing, healthy: failing}, CapabilityUnauthorized},
		{"unreachable before degraded", capabilityFacts{enabled: pass(), configured: pass(), authorized: pass(), reachable: failing, healthy: failing}, CapabilityUnreachable},
		{"degraded", capabilityFacts{enabled: pass(), configured: pass(), authorized: pass(), reachable: pass(), healthy: failing}, CapabilityDegraded},
	} {
		t.Run(test.name, func(t *testing.T) {
			capability := resolveCapability(test.facts)
			if capability.State != test.want {
				t.Fatalf("state = %q, want %q", capability.State, test.want)
			}
			if test.want == CapabilityAvailable && (capability.Reason != "" || capability.Remediation != "") {
				t.Errorf("available capability carries reason %q", capability.Reason)
			}
			if test.want != CapabilityAvailable && capability.Reason != "code" {
				t.Errorf("reason = %q, want code", capability.Reason)
			}
		})
	}
}

type capabilityFixture struct {
	cfg   *config.Config
	setup *setupmode.Resolver
}

func newCapabilityFixture(t *testing.T, setupMode bool) capabilityFixture {
	t.Helper()
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	body := "version: v0.3\n"
	if setupMode {
		body += "setup:\n  mode: true\n"
	}
	if err := os.WriteFile(configPath, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CAPABILITY_FIXTURE_SR_BENCH_TOKEN", "fixture-token")
	return capabilityFixture{
		cfg: &config.Config{
			AbsConfigPath: configPath, RuntimeConfigWritable: true,
			SRBenchTokenEnv: "CAPABILITY_FIXTURE_SR_BENCH_TOKEN", MCPEnabled: true,
			MLPipelineEnabled: true,
		},
		setup: setupmode.New(configPath, false),
	}
}

func (f capabilityFixture) collect(probes map[string]probeResult, perms ...string) map[string]Capability {
	permissions := map[string]bool{}
	for _, permission := range perms {
		permissions[permission] = true
	}
	byID := map[string]Capability{}
	for _, capability := range collectCapabilities(f.cfg, f.setup, probes, permissions).Capabilities {
		byID[capability.ID] = capability
	}
	return byID
}

var allCapabilityPermissions = []string{
	auth.PermConfigDeploy, auth.PermConfigRead, auth.PermEvalRead, auth.PermMlPipeline, auth.PermMcpRead,
}

func healthyProbes() map[string]probeResult {
	now := time.Now().UTC()
	return map[string]probeResult{
		probeRouter: {ok: true, checkedAt: now}, probeSRBench: {ok: true, checkedAt: now}, probeMLService: {ok: true, checkedAt: now},
	}
}

func writeServeHeartbeat(t *testing.T, configPath string) {
	t.Helper()
	if err := os.WriteFile(pendingActivationPath(configPath, serveHeartbeatSuffix), []byte(`{"state":"waiting"}`), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestCollectCapabilitiesReportsEveryCapability(t *testing.T) {
	fixture := newCapabilityFixture(t, true)
	writeServeHeartbeat(t, fixture.cfg.AbsConfigPath)
	got := fixture.collect(healthyProbes(), allCapabilityPermissions...)
	for _, id := range []string{"setup", "config", "evaluation", "mlPipeline", "mcp"} {
		if got[id].State != CapabilityAvailable {
			t.Errorf("%s = %+v, want available", id, got[id])
		}
	}
	if got["config"].CheckedAt == nil || got["evaluation"].CheckedAt == nil {
		t.Errorf("probed capabilities omit checkedAt: %+v", got)
	}
	if got["mcp"].CheckedAt != nil {
		t.Errorf("unprobed capability reports checkedAt")
	}
}

func TestCollectCapabilitiesReasons(t *testing.T) {
	covered := map[string]bool{}
	t.Cleanup(func() {
		for _, reason := range CapabilityReasons {
			if !covered[reason] {
				t.Errorf("reason %q has no test case", reason)
			}
		}
	})
	for _, test := range []struct {
		name   string
		id     string
		mutate func(*testing.T, *capabilityFixture, map[string]probeResult) []string
		state  CapabilityState
		reason string
	}{
		{"setup completed", "setup", func(_ *testing.T, f *capabilityFixture, _ map[string]probeResult) []string {
			*f = newCapabilityFixtureLike(f, false)
			return allCapabilityPermissions
		}, CapabilityDisabled, reasonSetupInactive},
		{"setup without deploy permission", "setup", func(t *testing.T, f *capabilityFixture, _ map[string]probeResult) []string {
			writeServeHeartbeat(t, f.cfg.AbsConfigPath)
			return []string{auth.PermConfigRead}
		}, CapabilityUnauthorized, reasonPermissionRequired},
		{"setup on a read-only Dashboard", "setup", func(t *testing.T, f *capabilityFixture, _ map[string]probeResult) []string {
			writeServeHeartbeat(t, f.cfg.AbsConfigPath)
			f.cfg.ReadonlyMode = true
			return allCapabilityPermissions
		}, CapabilityUnconfigured, reasonServerReadonly},
		{"setup with no serve attached", "setup", func(*testing.T, *capabilityFixture, map[string]probeResult) []string {
			return allCapabilityPermissions
		}, CapabilityDegraded, reasonServeNotAttached},
		{"unreadable config", "config", func(t *testing.T, f *capabilityFixture, _ map[string]probeResult) []string {
			if err := os.Remove(f.cfg.AbsConfigPath); err != nil {
				t.Fatal(err)
			}
			return allCapabilityPermissions
		}, CapabilityUnconfigured, reasonConfigUnreadable},
		{"router down", "config", func(_ *testing.T, _ *capabilityFixture, probes map[string]probeResult) []string {
			probes[probeRouter] = probeResult{checkedAt: time.Now()}
			return allCapabilityPermissions
		}, CapabilityUnreachable, reasonRouterUnreachable},
		{"router probe pending", "config", func(_ *testing.T, _ *capabilityFixture, probes map[string]probeResult) []string {
			delete(probes, probeRouter)
			return allCapabilityPermissions
		}, CapabilityUnreachable, reasonProbePending},
		{"server readonly", "config", func(_ *testing.T, f *capabilityFixture, _ map[string]probeResult) []string {
			f.cfg.ReadonlyMode = true
			return allCapabilityPermissions
		}, CapabilityDegraded, reasonServerReadonly},
		{"runtime config not writable", "config", func(_ *testing.T, f *capabilityFixture, _ map[string]probeResult) []string {
			f.cfg.RuntimeConfigWritable = false
			return allCapabilityPermissions
		}, CapabilityDegraded, reasonRuntimeConfigNotWritable},
		{"sr-bench token missing", "evaluation", func(_ *testing.T, f *capabilityFixture, _ map[string]probeResult) []string {
			f.cfg.SRBenchTokenEnv = "CAPABILITY_FIXTURE_UNSET_TOKEN"
			return allCapabilityPermissions
		}, CapabilityUnconfigured, reasonSRBenchAuthUnconfigured},
		{"sr-bench down", "evaluation", func(_ *testing.T, _ *capabilityFixture, probes map[string]probeResult) []string {
			probes[probeSRBench] = probeResult{checkedAt: time.Now()}
			return allCapabilityPermissions
		}, CapabilityUnreachable, reasonUpstreamUnreachable},
		{"evaluation without permission", "evaluation", func(*testing.T, *capabilityFixture, map[string]probeResult) []string {
			return []string{auth.PermConfigRead}
		}, CapabilityUnauthorized, reasonPermissionRequired},
		{"ml pipeline disabled", "mlPipeline", func(_ *testing.T, f *capabilityFixture, _ map[string]probeResult) []string {
			f.cfg.MLPipelineEnabled = false
			return allCapabilityPermissions
		}, CapabilityDisabled, reasonMLPipelineDisabled},
		{"ml sidecar down", "mlPipeline", func(_ *testing.T, f *capabilityFixture, probes map[string]probeResult) []string {
			f.cfg.MLServiceURL = "http://ml-service:8686"
			probes[probeMLService] = probeResult{checkedAt: time.Now()}
			return allCapabilityPermissions
		}, CapabilityUnreachable, reasonUpstreamUnreachable},
		{"mcp disabled", "mcp", func(_ *testing.T, f *capabilityFixture, _ map[string]probeResult) []string {
			f.cfg.MCPEnabled = false
			return allCapabilityPermissions
		}, CapabilityDisabled, reasonMCPDisabled},
	} {
		covered[test.reason] = true
		t.Run(test.name, func(t *testing.T) {
			fixture := newCapabilityFixture(t, true)
			probes := healthyProbes()
			perms := test.mutate(t, &fixture, probes)
			got := fixture.collect(probes, perms...)[test.id]
			if got.State != test.state || got.Reason != test.reason {
				t.Fatalf("%s = state %q reason %q, want %q %q", test.id, got.State, got.Reason, test.state, test.reason)
			}
			if !slices.Contains(CapabilityReasons, got.Reason) {
				t.Errorf("reason %q is not listed in CapabilityReasons", got.Reason)
			}
			if got.Remediation == "" {
				t.Errorf("reason %q has no remediation", got.Reason)
			}
		})
	}
}

// newCapabilityFixtureLike rewrites the fixture's config in place with or
// without setup mode, keeping its other settings.
func newCapabilityFixtureLike(f *capabilityFixture, setupMode bool) capabilityFixture {
	body := "version: v0.3\n"
	if setupMode {
		body += "setup:\n  mode: true\n"
	}
	_ = os.WriteFile(f.cfg.AbsConfigPath, []byte(body), 0o600)
	return capabilityFixture{cfg: f.cfg, setup: setupmode.New(f.cfg.AbsConfigPath, false)}
}

func TestCapabilitiesHandlerUsesCallerPermissionsAndHidesSecrets(t *testing.T) {
	t.Setenv("CAPABILITY_TEST_TOKEN", "super-secret-token")
	fixture := newCapabilityFixture(t, true)
	fixture.cfg.SRBenchURL = "http://sr-bench.internal:8090"
	fixture.cfg.SRBenchTokenEnv = "CAPABILITY_TEST_TOKEN"
	fixture.cfg.RouterAPIURL = "http://router.internal:8080"
	prober := newCapabilityProber(nil, nil, time.Hour)
	handler := CapabilitiesHandler(fixture.cfg, fixture.setup, prober)

	request := httptest.NewRequest(http.MethodGet, "/api/v1/capabilities", nil)
	request = request.WithContext(auth.WithAuthContext(request.Context(), auth.AuthContext{
		UserID: "reader", Perms: map[string]bool{auth.PermConfigRead: true, auth.PermEvalRead: true},
	}))
	recorder := httptest.NewRecorder()
	handler(recorder, request)
	if recorder.Code != http.StatusOK || recorder.Header().Get("Cache-Control") != "no-store" {
		t.Fatalf("status = %d cache-control = %q", recorder.Code, recorder.Header().Get("Cache-Control"))
	}
	for _, secret := range []string{"super-secret-token", "sr-bench.internal", "router.internal", fixture.cfg.AbsConfigPath} {
		if strings.Contains(recorder.Body.String(), secret) {
			t.Errorf("response leaks %q: %s", secret, recorder.Body.String())
		}
	}
	var response CapabilitiesResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	states := map[string]CapabilityState{}
	for _, capability := range response.Capabilities {
		states[capability.ID] = capability.State
	}
	if states["setup"] != CapabilityUnauthorized || states["mcp"] != CapabilityUnauthorized {
		t.Errorf("reader states = %v, want setup and mcp unauthorized", states)
	}
	if states["config"] != CapabilityUnreachable {
		t.Errorf("config before the first probe = %q, want unreachable", states["config"])
	}
}

func TestCapabilityProberDoesNotBlockReads(t *testing.T) {
	release := make(chan struct{})
	prober := newCapabilityProber(nil, []capabilityProbe{{
		name: probeSRBench,
		check: func(context.Context) bool {
			<-release
			return true
		},
	}}, time.Hour)
	prober.Start()
	t.Cleanup(func() { _ = prober.Close() })

	started := time.Now()
	if _, ok := prober.Snapshot()[probeSRBench]; ok {
		t.Fatal("snapshot reports a probe that has not completed")
	}
	if elapsed := time.Since(started); elapsed > 100*time.Millisecond {
		t.Fatalf("snapshot waited %s on a pending probe", elapsed)
	}

	close(release)
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if result, ok := prober.Snapshot()[probeSRBench]; ok {
			if !result.ok || result.checkedAt.IsZero() {
				t.Fatalf("result = %+v", result)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("probe result never published")
}

func TestCapabilityProberCloseWithoutStart(t *testing.T) {
	if err := newCapabilityProber(nil, nil, time.Hour).Close(); err != nil {
		t.Fatal(err)
	}
}

func TestProbeHTTPSendsBearerAndRejectsRedirects(t *testing.T) {
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/redirect" {
			http.Redirect(w, r, "/health", http.StatusFound)
			return
		}
		authorization = r.Header.Get("Authorization")
	}))
	t.Cleanup(server.Close)

	if !probeHTTP(context.Background(), server.URL+"/health", "token") || authorization != "Bearer token" {
		t.Fatalf("probe failed or sent authorization %q", authorization)
	}
	if probeHTTP(context.Background(), server.URL+"/redirect", "token") {
		t.Fatal("probe followed a redirect")
	}
	server.Close()
	if probeHTTP(context.Background(), server.URL+"/health", "") {
		t.Fatal("probe succeeded against a stopped server")
	}
}
