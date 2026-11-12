package config

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// validateHeaderMutationContracts refuses every header the header_mutation
// plugins of global.plugins, a recipe's plugins and its decisions add, update
// or delete that the Router or Envoy owns (headers.RouterOwned), one line per
// header, at the layer that writes it.
func validateHeaderMutationContracts(cfg *RouterConfig) error {
	var failures []error
	for _, plugin := range cfg.GlobalPlugins {
		failures = append(failures, reservedHeaderMutations("global.plugins["+DecisionPluginHeaderMutation+"].configuration", plugin)...)
	}
	for _, plugin := range cfg.Plugins {
		failures = append(failures, reservedHeaderMutations(cfg.ScopePath("plugins["+DecisionPluginHeaderMutation+"].configuration"), plugin)...)
	}
	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]
		for index, plugin := range decision.DeclaredPlugins() {
			base := cfg.ScopePath(fmt.Sprintf("decisions[%s].plugins[%d].configuration", decision.Name, index))
			failures = append(failures, reservedHeaderMutations(base, plugin)...)
		}
	}
	return errors.Join(failures...)
}

// reservedHeaderMutations are the headers the Router owns that plugin, when it
// is a header_mutation plugin configured at base, writes.
func reservedHeaderMutations(base string, plugin DecisionPlugin) []error {
	if NormalizeDecisionPluginType(plugin.Type) != DecisionPluginHeaderMutation || plugin.Configuration == nil {
		return nil
	}
	var mutation HeaderMutationPluginConfig
	if plugin.Configuration.DecodeInto(&mutation) != nil {
		return nil // the plugin payload check reports it
	}
	var failures []error
	check := func(path, name string) {
		if headers.RouterOwned(name) {
			failures = append(failures, CheckError(path, CodeReservedHeader, reservedHeaderMessage(name)))
		}
	}
	for _, pair := range mutation.Add {
		check(base+".add["+pair.Name+"]", pair.Name)
	}
	for _, pair := range mutation.Update {
		check(base+".update["+pair.Name+"]", pair.Name)
	}
	for j, name := range mutation.Delete {
		check(base+".delete["+strconv.Itoa(j)+"]", name)
	}
	return failures
}

func reservedHeaderMessage(name string) string {
	if isReliabilityHeader(name) {
		return fmt.Sprintf("%s carries the decision's retries and timeouts, which header_mutation cannot set; write them in decisions[].reliability", name)
	}
	return fmt.Sprintf("%s is owned by the Router or Envoy, which header_mutation cannot write; name a header the Router does not own", name)
}
