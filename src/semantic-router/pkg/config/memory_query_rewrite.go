package config

import (
	"cmp"
	"fmt"
	"strings"
)

// MemoryRewriteDeployment is the openai_chat deployment that rewrites memory
// search queries: global.stores.memory.query_rewrite.deployment, or the
// external[] model the deprecated model_role memory_rewrite marked.
func (c *RouterConfig) MemoryRewriteDeployment() (string, ModelDeployment, bool) {
	resolved := c.withModelResources()
	if resolved == nil || resolved.Memory.QueryRewrite == nil {
		return "", ModelDeployment{}, false
	}
	name := strings.TrimSpace(resolved.Memory.QueryRewrite.Deployment)
	deployment, ok := resolved.ModelDeployments[name]
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

// translateMemoryRewrite reads the external[] model with model_role
// memory_rewrite as an openai_chat deployment the query rewrite names.
func (t *translationState) translateMemoryRewrite() {
	if t.cfg.Memory.QueryRewrite != nil {
		return
	}
	external := t.cfg.FindExternalModelByRole(ModelRoleMemoryRewrite)
	if external == nil || strings.TrimSpace(external.ModelEndpoint.Address) == "" {
		return
	}
	deployment := deploymentFromExternal(*external, OpenAIChatProvider, DurationFromSeconds(external.TimeoutSeconds))
	deployment.translated = true
	name := t.uniqueName(cmp.Or(external.Name, "memory-rewrite"))
	t.cfg.ModelDeployments[name] = deployment
	t.cfg.Memory.QueryRewrite = &MemoryQueryRewriteConfig{Deployment: name}
	t.warn(fmt.Sprintf("global.model_catalog.external[%q]", external.Name), fmt.Sprintf(
		"finding the memory rewrite model by model_role is deprecated; read as deployments.%s {provider: openai_chat, endpoint: %s, model: %q} and global.stores.memory.query_rewrite.deployment: %s",
		name, deployment.Endpoint, deployment.Model, name))
}
