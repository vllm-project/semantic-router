package handlers

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

// CapabilityState is the first failing dimension of a capability, or available.
type CapabilityState string

const (
	CapabilityDisabled     CapabilityState = "disabled"
	CapabilityUnconfigured CapabilityState = "unconfigured"
	CapabilityUnauthorized CapabilityState = "unauthorized"
	CapabilityUnreachable  CapabilityState = "unreachable"
	CapabilityDegraded     CapabilityState = "degraded"
	CapabilityAvailable    CapabilityState = "available"
)

// CapabilityStates lists every state in precedence order.
var CapabilityStates = []CapabilityState{
	CapabilityDisabled, CapabilityUnconfigured, CapabilityUnauthorized,
	CapabilityUnreachable, CapabilityDegraded, CapabilityAvailable,
}

// Stable reason codes. Remediation text may change; these may not.
const (
	reasonSetupInactive            = "setup_inactive"
	reasonConfigUnreadable         = "config_unreadable"
	reasonPermissionRequired       = "permission_required"
	reasonServeNotAttached         = "serve_not_attached"
	reasonRouterUnreachable        = "router_unreachable"
	reasonProbePending             = "probe_pending"
	reasonServerReadonly           = "server_readonly"
	reasonRuntimeConfigNotWritable = "runtime_config_not_writable"
	reasonSRBenchAuthUnconfigured  = "sr_bench_auth_unconfigured"
	reasonUpstreamUnreachable      = "upstream_unreachable"
	reasonMLPipelineDisabled       = "ml_pipeline_disabled"
	reasonMCPDisabled              = "mcp_disabled"
)

// CapabilityReasons lists every reason code a capability can report.
var CapabilityReasons = []string{
	reasonSetupInactive, reasonConfigUnreadable, reasonPermissionRequired, reasonServeNotAttached,
	reasonRouterUnreachable, reasonProbePending, reasonServerReadonly, reasonRuntimeConfigNotWritable,
	reasonSRBenchAuthUnconfigured, reasonUpstreamUnreachable, reasonMLPipelineDisabled, reasonMCPDisabled,
}

// Capability is the runtime state of one Dashboard feature for the caller.
type Capability struct {
	ID          string          `json:"id"`
	Enabled     bool            `json:"enabled"`
	Configured  bool            `json:"configured"`
	Reachable   bool            `json:"reachable"`
	Authorized  bool            `json:"authorized"`
	Degraded    bool            `json:"degraded"`
	State       CapabilityState `json:"state"`
	Reason      string          `json:"reason,omitempty"`
	Remediation string          `json:"remediation,omitempty"`
	CheckedAt   *time.Time      `json:"checkedAt,omitempty"`
}

// CapabilitiesResponse is the body of GET /api/v1/capabilities.
type CapabilitiesResponse struct {
	Capabilities []Capability `json:"capabilities"`
}

// capabilityCheck is one dimension; reason and remediation explain a failure.
type capabilityCheck struct {
	ok          bool
	reason      string
	remediation string
}

func pass() capabilityCheck { return capabilityCheck{ok: true} }

func fail(reason, remediation string) capabilityCheck {
	return capabilityCheck{reason: reason, remediation: remediation}
}

type capabilityFacts struct {
	id                                         string
	enabled, configured, authorized, reachable capabilityCheck
	healthy                                    capabilityCheck
	checkedAt                                  *time.Time
}

func resolveCapability(facts capabilityFacts) Capability {
	capability := Capability{
		ID:         facts.id,
		Enabled:    facts.enabled.ok,
		Configured: facts.configured.ok,
		Authorized: facts.authorized.ok,
		Reachable:  facts.reachable.ok,
		Degraded:   !facts.healthy.ok,
		State:      CapabilityAvailable,
		CheckedAt:  facts.checkedAt,
	}
	for _, dimension := range []struct {
		state CapabilityState
		check capabilityCheck
	}{
		{CapabilityDisabled, facts.enabled},
		{CapabilityUnconfigured, facts.configured},
		{CapabilityUnauthorized, facts.authorized},
		{CapabilityUnreachable, facts.reachable},
		{CapabilityDegraded, facts.healthy},
	} {
		if !dimension.check.ok {
			capability.State = dimension.state
			capability.Reason = dimension.check.reason
			capability.Remediation = dimension.check.remediation
			break
		}
	}
	return capability
}

// CapabilitiesHandler reports each Dashboard feature's runtime state for the
// caller. Probe results come from prober's cache, so a read never waits on an
// upstream.
func CapabilitiesHandler(cfg *config.Config, setupResolver *setupmode.Resolver, prober *CapabilityProber) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var perms map[string]bool
		if ac, ok := auth.AuthFromContext(r); ok {
			perms = ac.Perms
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_ = json.NewEncoder(w).Encode(collectCapabilities(cfg, setupResolver, prober.Snapshot(), perms))
	}
}

func collectCapabilities(cfg *config.Config, setupResolver *setupmode.Resolver, probes map[string]probeResult, perms map[string]bool) CapabilitiesResponse {
	configReadable := fileReadable(cfg.AbsConfigPath)
	facts := []capabilityFacts{
		setupFacts(cfg, setupResolver, configReadable, perms),
		configFacts(cfg, configReadable, probes, perms),
		evaluationFacts(cfg, probes, perms),
		mlPipelineFacts(cfg, probes, perms),
		mcpFacts(cfg, perms),
	}
	response := CapabilitiesResponse{Capabilities: make([]Capability, 0, len(facts))}
	for _, f := range facts {
		response.Capabilities = append(response.Capabilities, resolveCapability(f))
	}
	return response
}

func setupFacts(cfg *config.Config, setupResolver *setupmode.Resolver, configReadable bool, perms map[string]bool) capabilityFacts {
	facts := capabilityFacts{
		id: "setup", enabled: pass(), configured: pass(), reachable: pass(), healthy: pass(),
		authorized: permitted(perms, auth.PermConfigDeploy),
	}
	if !configReadable {
		facts.enabled = fail(reasonConfigUnreadable, "The router config file cannot be read.")
		facts.configured = facts.enabled
		return facts
	}
	if !setupResolver.Active() {
		facts.enabled = fail(reasonSetupInactive, "Setup has been completed; the config no longer declares setup mode.")
	}
	if cfg.ReadonlyMode {
		facts.configured = fail(reasonServerReadonly, "The Dashboard runs in read-only mode; setup activation is disabled.")
	}
	if !configActivationDeferred() && !serveAttached(cfg.AbsConfigPath) {
		facts.healthy = fail(reasonServeNotAttached, "Activation will be saved; run `vllm-sr serve` to start the Router.")
	}
	return facts
}

func configFacts(cfg *config.Config, configReadable bool, probes map[string]probeResult, perms map[string]bool) capabilityFacts {
	facts := capabilityFacts{
		id: "config", enabled: pass(), configured: pass(), healthy: pass(),
		authorized: permitted(perms, auth.PermConfigRead),
	}
	if !configReadable {
		facts.configured = fail(reasonConfigUnreadable, "The router config file cannot be read.")
	}
	facts.reachable, facts.checkedAt = probeCheck(probes, probeRouter, reasonRouterUnreachable, "The Router management API did not report ready.")
	switch {
	case cfg.ReadonlyMode:
		facts.healthy = fail(reasonServerReadonly, "The Dashboard runs in read-only mode; config changes are disabled.")
	case !cfg.RuntimeConfigWritable:
		facts.healthy = fail(reasonRuntimeConfigNotWritable, "The runtime config is not writable; config changes cannot be saved.")
	}
	return facts
}

func evaluationFacts(cfg *config.Config, probes map[string]probeResult, perms map[string]bool) capabilityFacts {
	facts := capabilityFacts{
		id: "evaluation", enabled: pass(), configured: pass(), reachable: pass(), healthy: pass(),
		authorized: permitted(perms, auth.PermEvalRead),
	}
	if srBenchToken(cfg) == "" {
		facts.configured = fail(reasonSRBenchAuthUnconfigured, "Set the sr-bench service token for the Dashboard.")
		return facts
	}
	facts.reachable, facts.checkedAt = probeCheck(probes, probeSRBench, reasonUpstreamUnreachable, "The sr-bench service did not answer its health check.")
	return facts
}

func mlPipelineFacts(cfg *config.Config, probes map[string]probeResult, perms map[string]bool) capabilityFacts {
	facts := capabilityFacts{
		id: "mlPipeline", enabled: pass(), configured: pass(), reachable: pass(), healthy: pass(),
		authorized: permitted(perms, auth.PermMlPipeline),
	}
	if !cfg.MLPipelineEnabled {
		facts.enabled = fail(reasonMLPipelineDisabled, "Enable the ML pipeline with ML_PIPELINE_ENABLED=true.")
		return facts
	}
	if cfg.MLServiceURL != "" {
		facts.reachable, facts.checkedAt = probeCheck(probes, probeMLService, reasonUpstreamUnreachable, "The ML service did not answer its health check.")
	}
	return facts
}

func mcpFacts(cfg *config.Config, perms map[string]bool) capabilityFacts {
	facts := capabilityFacts{
		id: "mcp", enabled: pass(), configured: pass(), reachable: pass(), healthy: pass(),
		authorized: permitted(perms, auth.PermMcpRead),
	}
	if !cfg.MCPEnabled {
		facts.enabled = fail(reasonMCPDisabled, "Enable MCP with MCP_ENABLED=true.")
	}
	return facts
}

// srBenchToken is the service token the Dashboard forwards to sr-bench; the
// Evaluation capability is unconfigured without it.
func srBenchToken(cfg *config.Config) string {
	return strings.TrimSpace(os.Getenv(cfg.SRBenchTokenEnv))
}

func permitted(perms map[string]bool, permission string) capabilityCheck {
	if perms[permission] {
		return pass()
	}
	return fail(reasonPermissionRequired, "Your role lacks the "+permission+" permission.")
}

// probeCheck turns a cached probe into the reachable dimension. A probe that
// has not completed yet fails closed.
func probeCheck(probes map[string]probeResult, name, reason, remediation string) (capabilityCheck, *time.Time) {
	result, ok := probes[name]
	if !ok {
		return fail(reasonProbePending, "The first health check has not completed yet."), nil
	}
	checkedAt := result.checkedAt
	if !result.ok {
		return fail(reason, remediation), &checkedAt
	}
	return pass(), &checkedAt
}

func fileReadable(path string) bool {
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	_ = file.Close()
	return true
}
