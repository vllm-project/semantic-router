package catalog

import (
	"fmt"
	"strings"
)

const providerModelIDKindDeploymentName = "deployment_name"

// compileModelProviders materializes the runtime bindings for one request-facing
// model alias. Every backend must agree on one wire protocol because routing
// selects the endpoint after the request has already been projected.
func (registry *Registry) compileModelProviders(
	bindings []ModelProviderBinding,
	modelPath string,
	providers map[string]EffectiveProvider,
	card EffectiveModelCard,
) ([]EffectiveModelProvider, error) {
	result := make([]EffectiveModelProvider, 0, len(bindings))
	seen := map[string]struct{}{}
	selectedProtocol := ""
	for index, binding := range bindings {
		path := fmt.Sprintf("%s.providers[%d]", modelPath, index)
		effective, err := registry.compileModelProvider(binding, path, providers, card, seen, selectedProtocol)
		if err != nil {
			return nil, err
		}
		selectedProtocol = effective.Binding.Protocol
		result = append(result, effective)
	}
	return result, nil
}

func (registry *Registry) compileModelProvider(
	binding ModelProviderBinding,
	path string,
	providers map[string]EffectiveProvider,
	card EffectiveModelCard,
	seen map[string]struct{},
	selectedProtocol string,
) (EffectiveModelProvider, error) {
	provider, ok := providers[binding.Name]
	if !ok {
		return EffectiveModelProvider{}, fmt.Errorf("%s.name %q does not reference providers[].name", path, binding.Name)
	}
	if _, duplicate := seen[binding.Name]; duplicate {
		return EffectiveModelProvider{}, fmt.Errorf("%s.name %q is duplicated", path, binding.Name)
	}
	seen[binding.Name] = struct{}{}
	if binding.Protocol == "" {
		inferred, err := registry.inferModelProviderProtocol(binding, path, provider, card, selectedProtocol)
		if err != nil {
			return EffectiveModelProvider{}, err
		}
		binding.Protocol = inferred
	}
	if err := validateModelProviderProtocol(binding.Protocol, path, selectedProtocol, provider); err != nil {
		return EffectiveModelProvider{}, err
	}
	catalogBinding := registry.findCatalogBinding(provider.Definition.ID, card.Card.ID, binding.ModelID, binding.Protocol)
	if binding.ModelID == "" && catalogBinding != nil {
		if catalogBindingRequiresExplicitModelID(*catalogBinding) {
			return EffectiveModelProvider{}, fmt.Errorf(
				"%s cannot infer model_id for provider %q because its catalog mapping uses an operator-defined deployment name; set the containing providers.models[].provider_model_id or external_model_ids[%q]",
				path, provider.Definition.ID, provider.Definition.ID,
			)
		}
		binding.ModelID = catalogBinding.ID
	}
	if card.Card.Kind == "physical" && binding.ModelID == "" {
		if protocols := registry.catalogBindingProtocols(provider.Definition.ID, card.Card.ID, ""); len(protocols) > 0 {
			return EffectiveModelProvider{}, fmt.Errorf(
				"%s.protocol %q has no catalog binding for model %q through provider %q; available protocols: %s",
				path, binding.Protocol, card.Card.ID, provider.Definition.ID, strings.Join(protocols, ", "),
			)
		}
		return EffectiveModelProvider{}, fmt.Errorf("%s.model_id is required when the provider catalog has no matching model", path)
	}
	return EffectiveModelProvider{Binding: binding, Provider: provider, CatalogBinding: catalogBinding}, nil
}

func (registry *Registry) inferModelProviderProtocol(
	binding ModelProviderBinding,
	path string,
	provider EffectiveProvider,
	card EffectiveModelCard,
	selectedProtocol string,
) (string, error) {
	protocols := registry.catalogBindingProtocols(provider.Definition.ID, card.Card.ID, binding.ModelID)
	if len(protocols) == 0 {
		if binding.ModelID == "" && card.Card.Kind == "physical" {
			return "", fmt.Errorf(
				"%s.protocol cannot be inferred: provider %q has no catalog binding for model %q; set model_id and protocol explicitly",
				path, provider.Definition.ID, card.Card.ID,
			)
		}
		if selectedProtocol != "" {
			return selectedProtocol, nil
		}
		return provider.Definition.DefaultProtocol, nil
	}
	if len(protocols) == 1 {
		return protocols[0], nil
	}
	if selectedProtocol != "" {
		if contains(protocols, selectedProtocol) {
			return selectedProtocol, nil
		}
		return "", fmt.Errorf(
			"%s.protocol cannot use %q selected by another backend; provider %q exposes model %q through %s",
			path, selectedProtocol, provider.Definition.ID, card.Card.ID, strings.Join(protocols, ", "),
		)
	}
	if contains(protocols, provider.Definition.DefaultProtocol) {
		return provider.Definition.DefaultProtocol, nil
	}
	return "", fmt.Errorf(
		"%s.protocol is ambiguous: provider %q exposes model %q through %s and its default protocol is not one of them; set protocol explicitly",
		path, provider.Definition.ID, card.Card.ID, strings.Join(protocols, ", "),
	)
}

func (registry *Registry) catalogBindingProtocols(providerID, modelID, nativeModelID string) []string {
	provider, ok := registry.providers[providerID]
	if !ok {
		return nil
	}
	seen := map[string]struct{}{}
	protocols := make([]string, 0, len(provider.Protocols))
	for _, binding := range provider.Models {
		if binding.Catalog != modelID || !catalogBindingAcceptsModelID(binding, nativeModelID) {
			continue
		}
		for _, protocol := range binding.Protocols {
			if _, duplicate := seen[protocol]; duplicate {
				continue
			}
			seen[protocol] = struct{}{}
			protocols = append(protocols, protocol)
		}
	}
	return protocols
}

func validateModelProviderProtocol(protocol, path, selected string, provider EffectiveProvider) error {
	if !contains(provider.Definition.Protocols, protocol) {
		return fmt.Errorf("%s.protocol %q is not supported by provider %q", path, protocol, provider.Definition.ID)
	}
	if !contains(provider.Definition.SupportedOperations, protocol+"#create") {
		return fmt.Errorf("%s.protocol %q cannot create requests through provider %q", path, protocol, provider.Definition.ID)
	}
	if selected != "" && protocol != selected {
		return fmt.Errorf("%s.protocol %q conflicts with %q; one alias must use one wire protocol", path, protocol, selected)
	}
	return nil
}

func (registry *Registry) findCatalogBinding(providerID, modelID, nativeModelID, protocol string) *CatalogModelBinding {
	provider, ok := registry.providers[providerID]
	if !ok {
		return nil
	}
	for _, catalogBinding := range provider.Models {
		if catalogBinding.Catalog != modelID || !contains(catalogBinding.Protocols, protocol) {
			continue
		}
		if !catalogBindingAcceptsModelID(catalogBinding, nativeModelID) {
			continue
		}
		copy := catalogBinding
		copy.Protocols = append([]string(nil), catalogBinding.Protocols...)
		copy.Restrictions = cloneMap(catalogBinding.Restrictions)
		return &copy
	}
	return nil
}

func catalogBindingAcceptsModelID(binding CatalogModelBinding, modelID string) bool {
	return modelID == "" || binding.ID == modelID || catalogBindingRequiresExplicitModelID(binding)
}

func catalogBindingRequiresExplicitModelID(binding CatalogModelBinding) bool {
	kind, _ := binding.Restrictions["provider_model_id_kind"].(string)
	return kind == providerModelIDKindDeploymentName
}

func contains(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
