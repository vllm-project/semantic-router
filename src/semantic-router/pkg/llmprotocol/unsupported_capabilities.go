package llmprotocol

import (
	"fmt"
	"slices"
	"strings"
)

// declarableUnsupported are the capabilities a model may list in
// unsupported_capabilities: the sampling controls the Router then leaves out,
// and the developer role, whose messages it then sends as system messages.
var declarableUnsupported = []Capability{
	CapabilitySamplingMinP,
	CapabilitySamplingLogitBias,
	CapabilityDeveloperRole,
}

// UnsupportedCapabilityNames lists the capabilities a model may declare it
// lacks.
func UnsupportedCapabilityNames() []string {
	names := make([]string, 0, len(declarableUnsupported))
	for _, capability := range declarableUnsupported {
		names = append(names, Capabilities(capability).Names()...)
	}
	return names
}

// ParseUnsupportedCapabilities reads the capabilities a model declares it
// lacks. Only the names UnsupportedCapabilityNames lists are accepted.
func ParseUnsupportedCapabilities(names []string) (CapabilitySet, error) {
	allowed := UnsupportedCapabilityNames()
	var set CapabilitySet
	for _, name := range names {
		if !slices.Contains(allowed, name) {
			return CapabilitySet{}, fmt.Errorf("%q cannot be declared unsupported, use one of: %s", name, strings.Join(allowed, ", "))
		}
		parsed, err := ParseCapabilities([]string{name})
		if err != nil {
			return CapabilitySet{}, err
		}
		set.bits |= parsed.bits
	}
	return set, nil
}
