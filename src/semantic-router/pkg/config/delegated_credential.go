package config

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

// An upstream whose auth is credential: delegated takes the requesting
// user's credential, which the authenticator in front issues for an
// authenticated request (global.services.authz.delegated_credential), in
// place of a key. Only a call made for a client request presents it; a call
// the Router makes on its own behalf has none, so a delegated upstream that
// such a call needs fails to load, and one it reaches at run time is refused.

const delegatedContractRequired = "auth.credential: delegated requires global.services.authz.delegated_credential.enabled: true"

// validateDelegatedUpstreams checks the delegated credential contract and
// every delegated deployment, and keeps an answer a delegated upstream gave
// one user out of a response cache other users read.
func validateDelegatedUpstreams(cfg *RouterConfig) error {
	contract := cfg.Authz.DelegatedCredential
	if err := validateDelegatedCredentialHeader(contract, cfg.Authz.Identity); err != nil {
		return err
	}
	routerOwn := cfg.routerOwnEmbeddingConsumers()
	inUse := DeploymentsInUse(cfg)
	for _, name := range sortedModelKeys(cfg.ModelDeployments) {
		deployment := cfg.ModelDeployments[name]
		if !deployment.Auth.Delegated() {
			continue
		}
		path := "global.model_catalog.deployments." + name
		if !contract.Enabled {
			return fmt.Errorf("%s: %s", path, delegatedContractRequired)
		}
		reason := ""
		switch _, used := inUse[name]; {
		case deployment.IsModelRuntime():
			reason = "the Router checks and prepares a model_runtime endpoint on its own behalf"
		case routerOwn[name] != "":
			reason = routerOwn[name] + " calls it as the Router"
		case !used:
			reason = "only Router-own calls reach it"
		}
		if reason != "" {
			return delegatedRouterOwnError(path, reason)
		}
		if deployment.Provider == OpenAIEmbeddingsProvider && (deployment.Capabilities == nil || deployment.Capabilities.Dimensions == 0) {
			return fmt.Errorf("%s: auth.credential: delegated: declare capabilities.dimensions; the Router otherwise reads a remote embedding model's width with a warmup call of its own", path)
		}
	}
	return cfg.validateDelegatedResponseCaches()
}

// delegatedRouterOwnError refuses a delegated upstream that a Router-own call
// needs; reason names the call.
func delegatedRouterOwnError(path, reason string) error {
	return fmt.Errorf("%s: auth.credential: delegated, but %s; a Router-own call never carries a user's credential, give it a static key", path, reason)
}

// validateDelegatedCredentialHeader keeps the delegated credential off a
// header that carries a client credential or an identity: the Router must
// never forward a client's raw credential, nor read one authentication
// output as another.
func validateDelegatedCredentialHeader(credential DelegatedCredentialConfig, identity IdentityConfig) error {
	header := credential.GetHeader()
	reserved := IsShadowCredentialHeader(header)
	for _, name := range []string{
		identity.GetUserIDHeader(), identity.GetUserGroupsHeader(),
		headers.AuthzUserID, headers.AuthzUserGroups, headers.AuthzTeamID, headers.AuthzTenantID,
	} {
		reserved = reserved || strings.EqualFold(header, name)
	}
	if reserved {
		return fmt.Errorf(
			"global.services.authz.delegated_credential.header %q carries a client credential or an identity; name a dedicated header the authenticator sets, such as %s",
			header, headers.VSRDelegatedCredential,
		)
	}
	return nil
}

// routerOwnEmbeddingConsumers names, per deployment, an embedding consumer
// whose calls run as the Router: vector store ingestion and memory storage
// build their stores on the Router's own behalf.
func (c *RouterConfig) routerOwnEmbeddingConsumers() map[string]string {
	consumers := map[string]string{}
	for _, requirement := range EmbeddingRequirements(c.ConfigForGlobalModelServices(), true) {
		switch requirement.Consumer {
		case EmbeddingConsumerVectorStore:
			consumers[requirement.Deployment] = "vector store ingestion"
		case EmbeddingConsumerMemory:
			if consumers[requirement.Deployment] == "" {
				consumers[requirement.Deployment] = "memory storage"
			}
		}
	}
	return consumers
}

// DelegatedBackendOf returns the path of the first delegated backend of a
// provider model, and whether it has one.
func (c *RouterConfig) DelegatedBackendOf(model string) (string, bool) {
	params, ok := c.ModelConfig[model]
	if !ok {
		return "", false
	}
	for index, endpoint := range params.PreferredEndpoints {
		if profile, err := c.GetProviderProfileForEndpoint(endpoint); err == nil && profile.Delegated() {
			return fmt.Sprintf("providers.models[%s].backend_refs[%d]", model, index), true
		}
	}
	return "", false
}

// firstDelegatedUpstream returns the path of a delegated upstream, if any.
func (c *RouterConfig) firstDelegatedUpstream() (string, bool) {
	for _, model := range sortedModelKeys(c.ModelConfig) {
		if path, ok := c.DelegatedBackendOf(model); ok {
			return path, true
		}
	}
	for _, name := range sortedModelKeys(c.ModelDeployments) {
		if c.ModelDeployments[name].Auth.Delegated() {
			return "global.model_catalog.deployments." + name, true
		}
	}
	return "", false
}

// validateDelegatedResponseCaches keeps every response cache at its default
// scope, user, once any upstream is delegated: an answer obtained with one
// user's credential, or routed or checked with it, is served back only to
// that user.
func (c *RouterConfig) validateDelegatedResponseCaches() error {
	upstream, delegated := c.firstDelegatedUpstream()
	if !delegated {
		return nil
	}
	check := func(path string, plugins []DecisionPlugin) error {
		plugin := ResponseCachePluginOf(plugins)
		if plugin == nil {
			return nil
		}
		if scope := strings.TrimSpace(plugin.Scope); scope != "" && scope != "user" {
			return fmt.Errorf("%s: response_cache scope %q shares answers across users, and %s is delegated; an answer obtained with one user's credential is served back only to that user, use scope: user", path, scope, upstream)
		}
		return nil
	}
	if err := check("global.plugins", c.GlobalPlugins); err != nil {
		return err
	}
	recipes := c.Recipes
	if len(recipes) == 0 {
		recipes = []RoutingRecipe{*c.DefaultRecipe()}
	}
	for _, recipe := range recipes {
		if err := check(fmt.Sprintf("recipes[%s].plugins", recipe.Name), recipe.Profile.DefaultPlugins); err != nil {
			return err
		}
		for _, decision := range recipe.Profile.Decisions {
			if err := check(fmt.Sprintf("recipes[%s].decisions[%s].plugins", recipe.Name, decision.Name), decision.DeclaredPlugins()); err != nil {
				return err
			}
		}
	}
	return nil
}
