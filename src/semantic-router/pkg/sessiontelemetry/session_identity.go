package sessiontelemetry

import (
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// RoutingSessionKey identifies router state by recipe and raw identity components.
// A session ID is one component; a session/conversation pair is two. Escaping
// each component preserves those boundaries and keeps client text from becoming
// a recipe namespace, including when the default recipe is unprefixed.
func RoutingSessionKey(recipe config.RecipeName, components ...string) string {
	if len(components) == 0 {
		return ""
	}
	encoded := make([]string, len(components))
	for i, component := range components {
		component = strings.TrimSpace(component)
		if component == "" {
			return ""
		}
		encoded[i] = url.QueryEscape(component)
	}
	return config.RoutingNamespaceKey(recipe, strings.Join(encoded, "/"))
}
