package config

import (
	"fmt"
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

// ValidateReviewModels checks approval-review bindings in the public routing table.
// Review targets use the same inference namespace and cannot delegate again.
func (c *RouterConfig) ValidateReviewModels() error {
	for index, entrypoint := range c.Entrypoints {
		review := strings.TrimSpace(entrypoint.ReviewModel)
		if review == "" {
			continue
		}
		api := entrypoint.API
		if api == "" {
			api = ChatAPI
		}
		for _, name := range entrypoint.ModelNames {
			if name == review {
				return fmt.Errorf("entrypoints[%d].review_model: cannot reference its own model name %q", index, review)
			}
		}
		target, ok := c.ResolveEntrypoint(api, review)
		if !ok {
			return fmt.Errorf("entrypoints[%d].review_model: unknown entrypoint model %q in %s API", index, review, api)
		}
		if strings.TrimSpace(target.ReviewModel) != "" {
			return fmt.Errorf("entrypoints[%d].review_model: target %q must not have a review_model", index, review)
		}
	}
	return nil
}
