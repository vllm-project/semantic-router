package config

import (
	"fmt"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ReliabilityHeaders are the per-request headers through which the Router
// asks Envoy's router to apply a decision's reliability block. The block is
// their only writer.
var ReliabilityHeaders = []string{
	"x-envoy-upstream-rq-timeout-ms",
	"x-envoy-upstream-rq-per-try-timeout-ms",
	"x-envoy-max-retries",
	"x-envoy-retry-on",
	"x-envoy-retriable-status-codes",
}

func isReliabilityHeader(name string) bool {
	return slices.Contains(ReliabilityHeaders, strings.ToLower(strings.TrimSpace(name)))
}

// dropReliabilityHeaderMutations removes the reliability headers from every
// decision's header_mutation plugin, with a warning. Envoy's ext_proc ignored
// such entries before its filter allowed these headers, so no working config
// changes behavior, and the reliability block stays the one retry authority.
func dropReliabilityHeaderMutations(cfg *RouterConfig) {
	scopes := [][]Decision{cfg.Decisions}
	for i := range cfg.Recipes {
		scopes = append(scopes, cfg.Recipes[i].Profile.Decisions)
	}
	for _, decisions := range scopes {
		for i := range decisions {
			for j := range decisions[i].Plugins {
				dropReliabilityHeaderMutation(decisions[i].Name, &decisions[i].Plugins[j])
			}
		}
	}
}

func dropReliabilityHeaderMutation(decision string, plugin *DecisionPlugin) {
	if NormalizeDecisionPluginType(plugin.Type) != DecisionPluginHeaderMutation || plugin.Configuration == nil {
		return
	}
	var mutation HeaderMutationPluginConfig
	if plugin.Configuration.DecodeInto(&mutation) != nil {
		return
	}
	var dropped []string
	keep := func(name string) bool {
		if isReliabilityHeader(name) {
			dropped = append(dropped, strings.ToLower(strings.TrimSpace(name)))
			return false
		}
		return true
	}
	keepPairs := func(pairs []HeaderPair) []HeaderPair {
		return slices.DeleteFunc(pairs, func(pair HeaderPair) bool { return !keep(pair.Name) })
	}
	mutation.Add, mutation.Update = keepPairs(mutation.Add), keepPairs(mutation.Update)
	mutation.Delete = slices.DeleteFunc(mutation.Delete, func(name string) bool { return !keep(name) })
	if len(dropped) == 0 {
		return
	}
	payload, err := NewStructuredPayload(mutation)
	if err != nil {
		return
	}
	plugin.Configuration = payload
	logging.ComponentWarnEvent("config", "reliability_header_mutation_dropped", map[string]interface{}{
		"decision": decision,
		"headers":  dropped,
		"reason":   "a decision's reliability block is the only writer of these headers",
	})
}

// DecisionReliability overrides, for the requests a decision routes, the
// timeouts and retries of the provider model that serves them. The fields
// mean what they mean in the provider reliability block. Timeouts and the
// retry count replace the provider model's, and a 0s timeout disables it;
// retry_on and retriable_status_codes add to the provider model's, as Envoy's
// per-request headers do.
type DecisionReliability struct {
	TotalTimeout         string `yaml:"total_timeout,omitempty" json:"total_timeout,omitempty"`
	PerTryTimeout        string `yaml:"per_try_timeout,omitempty" json:"per_try_timeout,omitempty"`
	IdleTimeout          string `yaml:"idle_timeout,omitempty" json:"idle_timeout,omitempty"`
	FirstByteTimeout     string `yaml:"first_byte_timeout,omitempty" json:"first_byte_timeout,omitempty"`
	RetryCount           *int   `yaml:"retry_count,omitempty" json:"retry_count,omitempty"`
	RetryOn              string `yaml:"retry_on,omitempty" json:"retry_on,omitempty"`
	RetriableStatusCodes []int  `yaml:"retriable_status_codes,omitempty" json:"retriable_status_codes,omitempty"`
	RetryBackOffBase     string `yaml:"retry_back_off_base,omitempty" json:"retry_back_off_base,omitempty"`
	RetryBackOffMax      string `yaml:"retry_back_off_max,omitempty" json:"retry_back_off_max,omitempty"`
	RetryAfterMax        string `yaml:"retry_after_max,omitempty" json:"retry_after_max,omitempty"`
}

// NativeOnlyFields names the fields set that Envoy cannot apply to a single
// request, so only the native gateway honors them.
func (r *DecisionReliability) NativeOnlyFields() []string {
	if r == nil {
		return nil
	}
	var fields []string
	for _, field := range []struct{ name, value string }{
		{"idle_timeout", r.IdleTimeout},
		{"first_byte_timeout", r.FirstByteTimeout},
		{"retry_back_off_base", r.RetryBackOffBase},
		{"retry_back_off_max", r.RetryBackOffMax},
		{"retry_after_max", r.RetryAfterMax},
	} {
		if strings.TrimSpace(field.value) != "" {
			fields = append(fields, field.name)
		}
	}
	return fields
}

// CapabilityNativeReliability is the decision reliability fields that only
// the native gateway honors. Behind Envoy a decision overrides only what
// Envoy's per-request headers carry.
const CapabilityNativeReliability = "decision_reliability_native_fields"

func init() {
	GatewayCapabilities.MustRegister(CapabilityNativeReliability, GatewayCapability{
		Modes: []GatewayMode{GatewayStandalone},
		Uses: func(cfg *RouterConfig) []CapabilityUse {
			var uses []CapabilityUse
			for _, at := range cfg.RoutingDecisionsAt() {
				if fields := at.Decision.Reliability.NativeOnlyFields(); len(fields) > 0 {
					uses = append(uses, CapabilityUse{
						Path: at.Path + ".reliability",
						Subject: fmt.Sprintf("decision '%s': reliability.%s",
							at.Decision.Name, strings.Join(fields, ", reliability.")),
					})
				}
			}
			return uses
		},
	})
}

func validateDecisionReliability(decision Decision) error {
	r := decision.Reliability
	if r == nil {
		return nil
	}
	path := func(field string) string {
		return fmt.Sprintf("decision '%s': reliability.%s", decision.Name, field)
	}
	if r.RetryCount != nil && (*r.RetryCount < 0 || *r.RetryCount > 5) {
		return fmt.Errorf("%s must be between 0 and 5", path("retry_count"))
	}
	for _, code := range r.RetriableStatusCodes {
		if code < 100 || code > 599 {
			return fmt.Errorf("%s has %d; expected 100..599", path("retriable_status_codes"), code)
		}
	}
	for _, field := range []struct {
		name, value string
		zeroOK      bool
	}{
		{"total_timeout", r.TotalTimeout, true},
		{"per_try_timeout", r.PerTryTimeout, true},
		{"idle_timeout", r.IdleTimeout, true},
		{"first_byte_timeout", r.FirstByteTimeout, true},
		{"retry_after_max", r.RetryAfterMax, false},
	} {
		if _, err := reliabilityDuration(path(field.name), field.value, field.zeroOK); err != nil {
			return err
		}
	}
	base, err := reliabilityDuration(path("retry_back_off_base"), r.RetryBackOffBase, false)
	if err != nil {
		return err
	}
	maximum, err := reliabilityDuration(path("retry_back_off_max"), r.RetryBackOffMax, false)
	if err != nil {
		return err
	}
	if base == 0 {
		base = DefaultProviderRetryBackOffBase
	}
	if maximum != 0 && maximum < base {
		return fmt.Errorf("%s must not be below retry_back_off_base", path("retry_back_off_max"))
	}
	return nil
}
