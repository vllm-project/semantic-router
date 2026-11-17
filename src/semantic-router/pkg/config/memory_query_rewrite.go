package config

import (
	"cmp"
	"fmt"
	"strings"
)

// MemoryRewriteDeployment is the openai_chat deployment that rewrites memory
// search queries: global.stores.memory.query_rewrite.deployment.
func (c *RouterConfig) MemoryRewriteDeployment() (string, ModelDeployment, bool) {
	if c == nil || c.Memory.QueryRewrite == nil {
		return "", ModelDeployment{}, false
	}
	name := strings.TrimSpace(c.Memory.QueryRewrite.Deployment)
	deployment, ok := c.ModelDeployments[name]
	return name, deployment.WithDefaults(), ok && name != ""
}

// validateMemoryQueryRewrite refuses a query rewrite that names no chat
// deployment.
func validateMemoryQueryRewrite(cfg *RouterConfig) error {
	if cfg.Memory.QueryRewrite == nil {
		return nil
	}
	name := strings.TrimSpace(cfg.Memory.QueryRewrite.Deployment)
	deployment, ok := cfg.ModelDeployments[name]
	switch {
	case name == "":
		return fmt.Errorf("global.stores.memory.query_rewrite.deployment must name an openai_chat deployment")
	case !ok:
		return fmt.Errorf("global.stores.memory.query_rewrite.deployment: unknown deployment %q; %s", name, declaredDeployments(cfg.ModelDeployments))
	case deployment.Provider != OpenAIChatProvider:
		return fmt.Errorf("global.stores.memory.query_rewrite.deployment: %q uses provider %s; a query rewrite asks a chat model, so name an openai_chat deployment", name, cmp.Or(deployment.Provider, ModelRuntimeProvider))
	}
	return nil
}
