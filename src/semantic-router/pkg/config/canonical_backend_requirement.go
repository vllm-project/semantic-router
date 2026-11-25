package config

import (
	"fmt"
	"reflect"
	"strings"

	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
)

// validateRouterOwnedPhysicalBackend keeps provider selection explicit when
// Semantic Router owns the downstream listener. Metadata-only ExtProc configs
// deliberately remain backendless because their external gateway owns
// transport and credentials. Built-in virtual models also remain backendless:
// their recipe resolves a pool of physical model aliases at request time. An
// entry that states only facts describes a model for routing, not a backend.
func validateRouterOwnedPhysicalBackend(
	registry *modelcatalog.Registry,
	model CanonicalProviderModel,
	routerOwnsTransport bool,
	modelIndex int,
) error {
	if !routerOwnsTransport || len(model.BackendRefs) != 0 || model.Deployment != "" || model.factsOnly() {
		return nil
	}
	if catalogID := strings.TrimSpace(model.Catalog); catalogID != "" {
		card, ok := registry.Model(catalogID)
		if !ok {
			// The catalog builder reports the more precise unknown-card error.
			return nil
		}
		if card.Kind == "virtual" {
			return nil
		}
	}
	return fmt.Errorf(
		"providers.models[%d] %q is a physical model used by a router-owned listener and must define backend_refs with an explicit Provider ID; api_format selects only the upstream wire format",
		modelIndex,
		model.Name,
	)
}

// factsOnly reports an entry that states a name and facts, and no transport.
func (m CanonicalProviderModel) factsOnly() bool {
	transport := m
	transport.ModelFacts = ModelFacts{}
	return reflect.DeepEqual(transport, CanonicalProviderModel{Name: m.Name})
}
