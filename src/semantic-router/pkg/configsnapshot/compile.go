package configsnapshot

import (
	"fmt"
	"sort"
	"strconv"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Compile builds the typed resources of cfg and checks that every reference
// resolves and that names are unique within a kind. It reports every problem
// it finds as a *Rejection at StageCompile.
func Compile(cfg *config.RouterConfig) (*Resources, error) {
	if cfg == nil {
		return nil, RejectReasons(StageCompile, []Reason{{
			Stage: StageCompile, Code: CodeInvalidDocument, Message: "the configuration is empty",
		}})
	}
	c := &compiler{cfg: cfg, names: make(map[Ref]string)}
	c.runtimeModels()
	c.clusters()
	c.settings()
	c.programs()
	c.routes()
	c.listeners()
	c.resolve()
	if len(c.reasons) > 0 {
		return nil, RejectReasons(StageCompile, c.reasons)
	}
	return newResources(c.resources), nil
}

type compiler struct {
	cfg       *config.RouterConfig
	resources []*Resource
	// names maps every compiled resource to its path, to report duplicates.
	names   map[Ref]string
	reasons []Reason
}

func (c *compiler) add(resource *Resource) {
	if first, taken := c.names[resource.Ref]; taken {
		c.reasons = append(c.reasons, Reason{
			Stage: StageCompile, Code: CodeDuplicateName, Path: resource.Path,
			Message: fmt.Sprintf("%s is already defined at %s", resource.Ref, first),
		})
		return
	}
	c.names[resource.Ref] = resource.Path
	c.resources = append(c.resources, resource)
}

func (c *compiler) runtimeModels() {
	for _, name := range sortedKeys(c.cfg.ModelDeployments) {
		deployment := c.cfg.ModelDeployments[name]
		c.add(&Resource{
			Ref:  Ref{Kind: KindRuntimeModel, Name: name},
			Path: "global.model_catalog.deployments[" + name + "]",
			Hash: fingerprint(name, deployment),
			Spec: RuntimeModelSpec{
				Provider: deployment.Provider, Artifact: deployment.Artifact, Revision: deployment.Revision,
				Device: deployment.Device, Endpoint: deployment.Endpoint,
			},
		})
	}
}

// clusters compiles one cluster per provider model, its endpoints, and the
// secret that holds its credentials.
func (c *compiler) clusters() {
	byModel := make(map[string][]config.VLLMEndpoint)
	for _, endpoint := range c.cfg.VLLMEndpoints {
		byModel[endpoint.Model] = append(byModel[endpoint.Model], endpoint)
	}
	for order, alias := range c.clusterAliases(byModel) {
		params := c.cfg.ModelConfig[alias]
		path := "providers.models[" + alias + "]"
		cluster := &Resource{Ref: Ref{Kind: KindCluster, Name: alias}, Path: path}
		endpointNames := make([]string, 0, len(byModel[alias]))
		for _, endpoint := range byModel[alias] {
			ref := c.endpoint(alias, endpoint)
			cluster.Refs = append(cluster.Refs, ref)
			endpointNames = append(endpointNames, ref.Name)
		}
		if secret := c.clusterSecret(alias, params, byModel[alias]); secret != nil {
			cluster.Refs = append(cluster.Refs, secret.Ref)
		}
		cluster.Hash = fingerprint(alias, order, clusterContent(params), endpointNames)
		cluster.Spec = ClusterSpec{APIFormat: params.APIFormat, LBPolicy: params.Reliability.LBPolicy, Order: order}
		c.add(cluster)
	}
}

// settings compiles the settings resource: every field of the configuration
// that no other resource owns. Its hash is keyed because those fields include
// credentials, such as an external model's access key.
func (c *compiler) settings() {
	settings := &Resource{
		Ref:  Ref{Kind: KindSettings, Name: SettingsName},
		Path: "global",
		Hash: settingsFingerprint(c.cfg),
		Spec: SettingsSpec{DefaultModel: c.cfg.DefaultModel},
	}
	deployments := make(map[string]bool)
	for _, binding := range c.cfg.GlobalModelBindings {
		if binding.Deployment != "" {
			deployments[binding.Deployment] = true
		}
	}
	for _, deployment := range sortedKeys(deployments) {
		settings.Refs = append(settings.Refs, Ref{Kind: KindRuntimeModel, Name: deployment})
	}
	c.add(settings)
}

// clusterAliases lists the provider models in authored order, then any the
// configuration did not order, by name.
func (c *compiler) clusterAliases(byModel map[string][]config.VLLMEndpoint) []string {
	seen := make(map[string]bool)
	var aliases []string
	for _, alias := range c.cfg.ProviderModelOrder {
		if !seen[alias] {
			seen[alias] = true
			aliases = append(aliases, alias)
		}
	}
	var rest []string
	for alias := range c.cfg.ModelConfig {
		if !seen[alias] {
			seen[alias] = true
			rest = append(rest, alias)
		}
	}
	for alias := range byModel {
		if alias != "" && !seen[alias] {
			seen[alias] = true
			rest = append(rest, alias)
		}
	}
	sort.Strings(rest)
	return append(aliases, rest...)
}

// clusterContent is what a cluster's hash covers: the provider model without
// its credentials (the cluster's secret covers them) and without its backends
// (each endpoint is its own resource).
func clusterContent(params config.ModelParams) config.ModelParams {
	params.AccessKey = ""
	params.AccessKeys = nil
	params.PreferredEndpoints = nil
	if params.AuthoredModel != nil {
		authored := *params.AuthoredModel
		authored.BackendRefs = nil
		params.AuthoredModel = &authored
	}
	return params
}

// endpoint compiles one backend. Endpoint names come from the provider
// catalog and repeat across provider models, so a resource name is scoped by
// its cluster and made unique within it.
func (c *compiler) endpoint(alias string, endpoint config.VLLMEndpoint) Ref {
	base := alias + "/" + endpoint.Name
	ref := Ref{Kind: KindEndpoint, Name: base}
	for n := 2; ; n++ {
		if _, taken := c.names[ref]; !taken {
			break
		}
		ref.Name = base + "#" + strconv.Itoa(n)
	}
	profile := c.cfg.ProviderProfiles[endpoint.ProviderProfileName]
	content := endpoint
	content.APIKey = ""
	c.add(&Resource{
		Ref:  ref,
		Path: "providers.models[" + alias + "].backend_refs[" + endpoint.Name + "]",
		Hash: fingerprint(alias, content, profile),
		Spec: EndpointSpec{
			Cluster: alias, Address: endpoint.Address, Port: endpoint.Port, Weight: endpoint.Weight,
			Protocol: endpoint.Protocol,
		},
	})
	return ref
}

// clusterSecret compiles the credentials of a provider model, or returns nil
// when it has none.
func (c *compiler) clusterSecret(alias string, params config.ModelParams, endpoints []config.VLLMEndpoint) *Resource {
	values := make([]string, 0, len(endpoints)+1)
	if params.AccessKey != "" {
		values = append(values, params.AccessKey)
	}
	for _, provider := range sortedKeys(params.AccessKeys) {
		values = append(values, provider+"="+params.AccessKeys[provider])
	}
	for _, endpoint := range endpoints {
		if endpoint.APIKey != "" {
			values = append(values, endpoint.Name+"="+endpoint.APIKey)
		}
	}
	spec := SecretSpec{}
	if params.AuthoredModel != nil {
		for _, ref := range params.AuthoredModel.BackendRefs {
			switch {
			case ref.APIKeyEnv != "":
				spec.Env = appendUnique(spec.Env, ref.APIKeyEnv)
			case ref.APIKey != "":
				spec.Inline++
			}
		}
	}
	if len(values) == 0 && len(spec.Env) == 0 && spec.Inline == 0 {
		return nil
	}
	secret := &Resource{
		Ref:  Ref{Kind: KindSecret, Name: alias},
		Path: "providers.models[" + alias + "].backend_refs",
		Hash: secretFingerprint(alias, spec, values),
		Spec: spec,
	}
	c.add(secret)
	return secret
}

func (c *compiler) programs() {
	recipes := c.cfg.Recipes
	if len(recipes) == 0 {
		recipes = []config.RoutingRecipe{*c.cfg.DefaultRecipe()}
	}
	for _, recipe := range recipes {
		name := string(recipe.Name)
		path := "recipes[" + name + "]"
		if recipe.Name == config.DefaultRecipeName {
			path = "routing"
		}
		program := &Resource{Ref: Ref{Kind: KindProgram, Name: name}, Path: path, Hash: fingerprint(recipe)}
		spec := ProgramSpec{}
		clusters := make(map[string]bool)
		for _, decision := range recipe.Profile.Decisions {
			spec.Decisions = append(spec.Decisions, decision.Name)
			for _, model := range decision.ModelRefs {
				if _, isCluster := c.names[Ref{Kind: KindCluster, Name: model.Model}]; isCluster {
					clusters[model.Model] = true
				}
			}
		}
		for _, alias := range sortedKeys(clusters) {
			program.Refs = append(program.Refs, Ref{Kind: KindCluster, Name: alias})
		}
		deployments := make(map[string]bool)
		for _, binding := range recipe.Profile.ModelBindings {
			if binding.Deployment != "" {
				deployments[binding.Deployment] = true
			}
		}
		for _, deployment := range sortedKeys(deployments) {
			program.Refs = append(program.Refs, Ref{Kind: KindRuntimeModel, Name: deployment})
		}
		program.Spec = spec
		c.add(program)
	}
}

func (c *compiler) routes() {
	for index, entrypoint := range c.cfg.Entrypoints {
		for _, model := range entrypoint.ModelNames {
			program := Ref{Kind: KindProgram, Name: string(entrypoint.Recipe)}
			c.add(&Resource{
				Ref:  Ref{Kind: KindRoute, Name: model},
				Path: "entrypoints[" + strconv.Itoa(index) + "]",
				Hash: fingerprint(model, program.Name),
				Refs: []Ref{program},
				Spec: RouteSpec{Model: model, Program: program.Name},
			})
		}
	}
}

func (c *compiler) listeners() {
	for _, listener := range c.cfg.Listeners {
		path := "listeners[" + listener.Name + "]"
		if listener.Name == "" {
			c.reasons = append(c.reasons, Reason{
				Stage: StageCompile, Code: CodeInvalidResource, Path: "listeners",
				Message: "every listener needs a name",
			})
			continue
		}
		resource := &Resource{
			Ref:  Ref{Kind: KindListener, Name: listener.Name},
			Path: path,
			Spec: ListenerSpec{Address: listener.Address, Port: listener.Port, Timeout: listener.Timeout},
		}
		if len(listener.APIKeys) > 0 {
			secret := &Resource{
				Ref:  Ref{Kind: KindSecret, Name: "listener/" + listener.Name},
				Path: path + ".api_keys",
				Spec: SecretSpec{Inline: len(listener.APIKeys)},
			}
			secret.Hash = secretFingerprint(secret.Name, listener.APIKeys)
			c.add(secret)
			resource.Refs = []Ref{secret.Ref}
		}
		content := listener
		content.APIKeys = nil
		resource.Hash = fingerprint(content)
		c.add(resource)
	}
}

// resolve reports every reference that names no compiled resource.
func (c *compiler) resolve() {
	for _, resource := range c.resources {
		for _, ref := range resource.Refs {
			if _, ok := c.names[ref]; !ok {
				c.reasons = append(c.reasons, Reason{
					Stage: StageCompile, Code: CodeUnresolvedReference, Path: resource.Path,
					Message: fmt.Sprintf("%s references %s, which the configuration does not define", resource.Ref, ref),
				})
			}
		}
	}
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func appendUnique(values []string, value string) []string {
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}
