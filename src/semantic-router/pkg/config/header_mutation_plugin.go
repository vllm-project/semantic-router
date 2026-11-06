package config

import (
	"errors"
	"fmt"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// validateHeaderMutationContracts refuses every header a routing scope's
// header_mutation plugins add, update or delete that the Router or Envoy
// owns (headers.RouterOwned), one line per header.
func validateHeaderMutationContracts(cfg *RouterConfig) error {
	var failures []error
	for i := range cfg.Decisions {
		decision := &cfg.Decisions[i]
		for index, plugin := range decision.Plugins {
			if NormalizeDecisionPluginType(plugin.Type) != DecisionPluginHeaderMutation || plugin.Configuration == nil {
				continue
			}
			var mutation HeaderMutationPluginConfig
			if plugin.Configuration.DecodeInto(&mutation) != nil {
				continue // the plugin payload check reports it
			}
			base := cfg.ScopePath(fmt.Sprintf("decisions[%s].plugins[%d].configuration", decision.Name, index))
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
		}
	}
	return errors.Join(failures...)
}

func reservedHeaderMessage(name string) string {
	if isReliabilityHeader(name) {
		return fmt.Sprintf("%s carries the decision's retries and timeouts, which header_mutation cannot set; write them in decisions[].reliability", name)
	}
	return fmt.Sprintf("%s is owned by the Router or Envoy, which header_mutation cannot write; name a header the Router does not own", name)
}
