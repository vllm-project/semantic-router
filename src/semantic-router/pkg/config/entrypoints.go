package config

import "strings"

// InferenceAPI identifies an inference contract independently of its HTTP path.
// Only ChatAPI has recipe routing today; SystemOneAPI is a separate native
// contract and must never inherit Chat aliases or fallback behavior.
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
	if api != ChatAPI {
		return nil
	}
	var declared []EntrypointMapping
	if c != nil {
		declared = c.Entrypoints
	}
	result := make([]EntrypointMapping, 0, len(declared)+1)
	hasDefault := false
	for _, entrypoint := range declared {
		entrypoint.API = ChatAPI
		entrypoint.Source = EntrypointExplicit
		entrypoint.ModelNames = append([]string(nil), entrypoint.ModelNames...)
		result = append(result, entrypoint)
		hasDefault = hasDefault || entrypoint.Recipe == DefaultRecipeName
	}
	if !hasDefault {
		result = append([]EntrypointMapping{{
			API: ChatAPI, Source: EntrypointBuiltin,
			ModelNames: []string{DefaultEntrypointModel}, Recipe: DefaultRecipeName,
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

// DefaultEntrypointNames returns the default recipe's effective public names.
// Consumers that need a default selection use the first declared name.
func (c *RouterConfig) DefaultEntrypointNames() []string {
	var names []string
	for _, entrypoint := range c.EffectiveEntrypoints(ChatAPI) {
		if entrypoint.Recipe == DefaultRecipeName {
			names = append(names, entrypoint.ModelNames...)
		}
	}
	return names
}
