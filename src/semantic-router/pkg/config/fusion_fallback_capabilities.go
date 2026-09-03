/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package config

import (
	"sort"
	"strings"
)

// fusionFallbackCapabilityGap reports why a fallback target cannot be shown to
// cover the capabilities its panel declares.
type fusionFallbackCapabilityGap struct {
	// missing lists panel capabilities the target does not declare.
	missing []string
	// targetUndeclared is set when the panel declares capabilities and the
	// target declares none, so compatibility cannot be established at all.
	targetUndeclared bool
}

// fusionFallbackCapabilityCheck compares a fallback target against the
// capabilities its effective panel declares.
//
// The panel's declared union is the requirement: the fallback answers in place
// of the whole panel, so it must serve at least what the panel could. The rule
// is fail-closed. When the panel declares capabilities, a target declaring none
// is rejected rather than assumed capable, because absent metadata does not
// establish compatibility and treating it as such would let any target pass by
// omitting its declaration. When the panel declares nothing, there is no
// requirement and a target declaring nothing is accepted.
func fusionFallbackCapabilityCheck(
	panelCapabilities [][]string,
	targetCapabilities []string,
) fusionFallbackCapabilityGap {
	required := make([]string, 0)
	for _, declared := range panelCapabilities {
		for _, capability := range declared {
			normalized := strings.TrimSpace(capability)
			if normalized == "" {
				continue
			}
			if !containsString(required, normalized) {
				required = append(required, normalized)
			}
		}
	}
	if len(required) == 0 {
		return fusionFallbackCapabilityGap{}
	}

	covered := make(map[string]bool, len(targetCapabilities))
	for _, capability := range targetCapabilities {
		if normalized := strings.TrimSpace(capability); normalized != "" {
			covered[normalized] = true
		}
	}
	if len(covered) == 0 {
		return fusionFallbackCapabilityGap{targetUndeclared: true}
	}

	missing := make([]string, 0, len(required))
	for _, capability := range required {
		if !covered[capability] {
			missing = append(missing, capability)
		}
	}
	sort.Strings(missing)
	return fusionFallbackCapabilityGap{missing: missing}
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

// modelExecutionIdentity names what a runtime identity will actually dispatch.
//
// It is deliberately coarser than the spelling and finer than the owner. A
// catalog key and the provider external IDs that entry declares all dispatch the
// same model, so they share one execution identity: falling back from one to
// another would retry the model that just failed. A LoRA adapter is a separate
// executable variant served by the same base, so it keeps the base plus the
// adapter name and remains a legitimate fallback source for its base.
type modelExecutionIdentity struct {
	base    string
	adapter string
}

// modelIdentityResolution reports how a runtime model identity maps onto the
// catalog: its single owner, or every owner that claims it.
type modelIdentityResolution struct {
	owner        string
	capabilities []string
	execution    modelExecutionIdentity

	// ambiguous lists every catalog key claiming the identity, sorted, when more
	// than one does.
	ambiguous []string
}

func (r modelIdentityResolution) resolved() bool {
	return r.owner != "" && len(r.ambiguous) == 0
}

// resolveModelIdentity maps a runtime model identity onto its catalog entry.
//
// A decision can name a model by catalog key, by provider external model ID, or
// by a LoRA adapter name that inherits its base model's capabilities. Exact keys
// win before any alias search, so an entry is never shadowed by another
// declaring its name as an alias.
//
// Neither alias form is required to be unique across model cards, so indirect
// matches are collected across the whole catalog rather than short-circuiting.
// That is what keeps the result independent of map iteration order, and it is
// why ambiguity is reported instead of resolved: picking an owner would let a
// safety check pass on an identity that was never proven.
func resolveModelIdentity(catalog map[string]ModelParams, name string) modelIdentityResolution {
	trimmed := strings.TrimSpace(name)
	if len(catalog) == 0 || trimmed == "" {
		return modelIdentityResolution{}
	}
	if params, ok := catalog[trimmed]; ok {
		return modelIdentityResolution{
			owner:        trimmed,
			capabilities: params.Capabilities,
			execution:    modelExecutionIdentity{base: trimmed},
		}
	}

	owners := make([]string, 0, 2)
	claims := make(map[string]identityClaim, 2)
	for candidate, params := range catalog {
		if claim := modelParamsClaimsIdentity(params, trimmed); claim != identityClaimNone {
			owners = append(owners, candidate)
			claims[candidate] = claim
		}
	}
	switch len(owners) {
	case 0:
		return modelIdentityResolution{}
	case 1:
		owner := owners[0]
		execution := modelExecutionIdentity{base: owner}
		if claims[owner] == identityClaimLoRA {
			execution.adapter = trimmed
		}
		return modelIdentityResolution{
			owner:        owner,
			capabilities: catalog[owner].Capabilities,
			execution:    execution,
		}
	default:
		sort.Strings(owners)
		return modelIdentityResolution{ambiguous: owners}
	}
}

// identityClaim names which alias form an entry used to claim an identity. The
// two forms differ in what they dispatch, so the distinction has to survive.
type identityClaim int

const (
	identityClaimNone identityClaim = iota
	identityClaimExternalID
	identityClaimLoRA
)

// modelParamsClaimsIdentity reports how an entry declares name, if at all: as a
// provider external ID for itself, or as a LoRA adapter it serves.
func modelParamsClaimsIdentity(params ModelParams, name string) identityClaim {
	for _, externalID := range params.ExternalModelIDs {
		if externalID == name {
			return identityClaimExternalID
		}
	}
	for _, adapter := range params.LoRAs {
		if adapter.Name == name {
			return identityClaimLoRA
		}
	}
	return identityClaimNone
}

// fusionFallbackCapabilityResolution reports every reason a panel and target
// may fail the capability rule: identities with no metadata, identities
// claimed by more than one catalog entry, and the capability gap among those
// that resolved cleanly.
type fusionFallbackCapabilityResolution struct {
	// unresolved lists model identities with no metadata, in the order given.
	// Any entry here is a rejection: compatibility is unknown, not established.
	unresolved []string

	// ambiguous describes identities owned by several catalog entries, each
	// rendered with its sorted owners so the diagnostic is stable.
	ambiguous []string

	// conflictingPanelModels lists panel entries that dispatch the same model as
	// the target, in panel order. Retrying a model that just failed in the panel
	// is not recovery, and comparing spellings alone misses the case where a
	// panel names the target by one of its provider external IDs.
	conflictingPanelModels []string

	gap fusionFallbackCapabilityGap
}

// resolveFusionFallbackCapabilities applies the fail-closed capability rule to a
// recipe's panel and fallback target at config load.
//
// Every identity is resolved before any comparison, and both failure modes are
// rejections. Skipping an unresolvable panel member would shrink the required
// union, and an empty union approves any target.
func resolveFusionFallbackCapabilities(
	catalog map[string]ModelParams,
	panelModels []string,
	target string,
) fusionFallbackCapabilityResolution {
	var resolution fusionFallbackCapabilityResolution
	panelCapabilities := make([][]string, 0, len(panelModels))
	panelIdentities := make([]modelIdentityResolution, 0, len(panelModels))
	panelNames := make([]string, 0, len(panelModels))
	for _, panelModel := range panelModels {
		identity := resolveModelIdentity(catalog, panelModel)
		switch {
		case len(identity.ambiguous) > 0:
			resolution.ambiguous = append(resolution.ambiguous,
				describeAmbiguousIdentity(panelModel, identity.ambiguous))
		case !identity.resolved():
			resolution.unresolved = append(resolution.unresolved, strings.TrimSpace(panelModel))
		default:
			panelCapabilities = append(panelCapabilities, identity.capabilities)
			panelIdentities = append(panelIdentities, identity)
			panelNames = append(panelNames, strings.TrimSpace(panelModel))
		}
	}

	targetIdentity := resolveModelIdentity(catalog, target)
	switch {
	case len(targetIdentity.ambiguous) > 0:
		resolution.ambiguous = append(resolution.ambiguous,
			describeAmbiguousIdentity(target, targetIdentity.ambiguous))
	case !targetIdentity.resolved():
		resolution.unresolved = append(resolution.unresolved, strings.TrimSpace(target))
	}
	if len(resolution.unresolved) > 0 || len(resolution.ambiguous) > 0 {
		return resolution
	}

	resolution.conflictingPanelModels = conflictingExecutionIdentities(
		panelNames, panelIdentities, targetIdentity)
	if len(resolution.conflictingPanelModels) > 0 {
		return resolution
	}

	resolution.gap = fusionFallbackCapabilityCheck(panelCapabilities, targetIdentity.capabilities)
	return resolution
}

func describeAmbiguousIdentity(name string, owners []string) string {
	return strings.TrimSpace(name) + " (claimed by " + strings.Join(owners, ", ") + ")"
}

// conflictingExecutionIdentities returns the panel names that dispatch the same
// model as the target.
func conflictingExecutionIdentities(
	panelNames []string,
	panelIdentities []modelIdentityResolution,
	target modelIdentityResolution,
) []string {
	if !target.resolved() {
		return nil
	}
	var conflicts []string
	for index, identity := range panelIdentities {
		if identity.resolved() && identity.execution == target.execution {
			conflicts = append(conflicts, panelNames[index])
		}
	}
	return conflicts
}
