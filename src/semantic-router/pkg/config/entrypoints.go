package config

import (
	"fmt"
	"slices"
	"strings"
)

// InferenceAPI identifies an inference contract independently of its HTTP path.
// Chat and System One have separate routing namespaces; neither inherits the
// other API's aliases or fallback behavior.
type InferenceAPI string

const (
	ChatAPI      InferenceAPI = "chat"
	SystemOneAPI InferenceAPI = "systemone"
)

type EntrypointSource string

const (
	EntrypointExplicit EntrypointSource = "explicit"
	EntrypointBuiltin  EntrypointSource = "builtin"
)

// EffectiveEntrypoints returns the public routing table. Entrypoints on the
// config remain source declarations so exporting a document never freezes an
// inherited default name into a user override.
func (c *RouterConfig) EffectiveEntrypoints(api InferenceAPI) []EntrypointMapping {
	if api != ChatAPI && api != SystemOneAPI {
		return nil
	}
	var declared []EntrypointMapping
	if c != nil {
		declared = c.Entrypoints
	}
	result := make([]EntrypointMapping, 0, len(declared)+1)
	hasDefault := false
	for _, entrypoint := range declared {
		if entrypoint.API == "" {
			entrypoint.API = ChatAPI
		}
		if entrypoint.API != api {
			continue
		}
		entrypoint.Source = EntrypointExplicit
		entrypoint.ModelNames = append([]string(nil), entrypoint.ModelNames...)
		entrypoint.ClientAuth = entrypoint.ClientAuth.Clone()
		result = append(result, entrypoint)
		hasDefault = hasDefault || c.IsDefaultRecipe(entrypoint.Recipe)
	}
	if api == ChatAPI && !hasDefault {
		result = append([]EntrypointMapping{{
			API: ChatAPI, Source: EntrypointBuiltin, Default: true,
			ModelNames: []string{DefaultEntrypointModel}, Recipe: c.defaultRecipeName(),
		}}, result...)
	}
	return result
}

func (c *RouterConfig) ResolveEntrypoint(api InferenceAPI, model string) (EntrypointMapping, bool) {
	model = strings.TrimSpace(model)
	for _, entrypoint := range c.EffectiveEntrypoints(api) {
		for _, name := range entrypoint.ModelNames {
			if name == model {
				return entrypoint, true
			}
		}
	}
	return EntrypointMapping{}, false
}

// DefaultEntrypointNames returns the default recipe's effective public names,
// the default entrypoint's first. Consumers that need a default selection use
// the first name.
func (c *RouterConfig) DefaultEntrypointNames() []string {
	var marked, others []string
	for _, entrypoint := range c.EffectiveEntrypoints(ChatAPI) {
		switch {
		case entrypoint.Default:
			marked = append(marked, entrypoint.ModelNames...)
		case c.IsDefaultRecipe(entrypoint.Recipe):
			others = append(others, entrypoint.ModelNames...)
		}
	}
	return append(marked, others...)
}

// ValidateDelegateModels checks the approval-review and memory bindings in the
// public routing table. Their targets use the same inference namespace and
// cannot delegate again.
func (c *RouterConfig) ValidateDelegateModels() error {
	for index, entrypoint := range c.Entrypoints {
		api := entrypoint.API
		if api == "" {
			api = ChatAPI
		}
		for _, delegate := range []struct{ field, model string }{
			{"review_model", entrypoint.ReviewModel},
			{"memory_model", entrypoint.MemoryModel},
		} {
			field, model := delegate.field, strings.TrimSpace(delegate.model)
			if model == "" {
				continue
			}
			if slices.Contains(entrypoint.ModelNames, model) {
				return fmt.Errorf("entrypoints[%d].%s: cannot reference its own model name %q", index, field, model)
			}
			target, ok := c.ResolveEntrypoint(api, model)
			if !ok {
				return fmt.Errorf("entrypoints[%d].%s: unknown entrypoint model %q in %s API", index, field, model, api)
			}
			if strings.TrimSpace(target.ReviewModel) != "" || strings.TrimSpace(target.MemoryModel) != "" {
				return fmt.Errorf("entrypoints[%d].%s: target %q must not have a review_model or memory_model", index, field, model)
			}
		}
	}
	return nil
}

// IsReviewEntrypoint reports whether model names an entrypoint that another
// entrypoint uses as its review_model, under any of the target's names, whatever
// other role it serves.
func (c *RouterConfig) IsReviewEntrypoint(api InferenceAPI, model string) bool {
	return slices.Contains(c.DelegateRoles(api, model), DelegateReview)
}

// Roles an entrypoint serves for another one, as its review_model or
// memory_model.
const (
	DelegateReview = "review"
	DelegateMemory = "memory"
)

// DelegateRoles is what other entrypoints send to model, under any of its
// names: DelegateMemory, DelegateReview or both, in that order, and none when
// no entrypoint names it.
func (c *RouterConfig) DelegateRoles(api InferenceAPI, model string) []string {
	if c == nil {
		return nil
	}
	target, ok := c.ResolveEntrypoint(api, model)
	if !ok {
		return nil
	}
	var roles []string
	for _, entrypoint := range c.EffectiveEntrypoints(api) {
		for _, delegate := range [...]struct{ role, name string }{
			{DelegateReview, entrypoint.ReviewModel}, {DelegateMemory, entrypoint.MemoryModel},
		} {
			name := strings.TrimSpace(delegate.name)
			if name != "" && slices.Contains(target.ModelNames, name) && !slices.Contains(roles, delegate.role) {
				roles = append(roles, delegate.role)
			}
		}
	}
	slices.Sort(roles)
	return roles
}
