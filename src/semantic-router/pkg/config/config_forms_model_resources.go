package config

import (
	"cmp"
	"fmt"
	"net"
	"net/url"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// modelResourceForms are the ways to choose a model that the Router read as
// deployments and bindings before every model was a named deployment: the
// earlier spellings of deployment fields, deployments of the removed
// classifier providers, the external[] models, binding objects and bindings
// for one rule, and the request backends' connection fields, trusted gateway
// marker included. They run after the application and embedding forms, which
// rename or drop the bindings of removed modules first.
var modelResourceForms = []ConfigForm{
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.model_catalog.deployments.*.question_types"},
		What:    "question_types is removed",
		Becomes: "capabilities.question_types",
		Migrate: migrateDeploymentQuestionTypes,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.model_catalog.deployments.*.served_name", "global.model_catalog.deployments.*.replicas[].served_name"},
		What:    "served_name is removed",
		Becomes: "model",
		Migrate: func(m *Migration, at FormMatch) error {
			// model won over served_name when both were set.
			if mappingValue(at.Parent, "model") != nil {
				m.Delete(at.Parent, "served_name")
				return nil
			}
			return m.Rename(at.Parent, "served_name", "model")
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.model_catalog.deployments.*.operating_point"},
		What:    "a deployment's operating_point is removed: nothing reads it",
		Becomes: "nothing: migrate drops it; the model runtime applies the operating point its package declares",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			return nil
		},
	},
	{
		Code:  CodeInvalidValue,
		Paths: []string{"global.model_catalog.deployments.*"},
		Match: func(node *yaml.Node, _ string) bool {
			return slices.Contains(removedClassifierProviders, scalarAt(node, "provider"))
		},
		Describe: func(at FormMatch) (string, string) {
			return fmt.Sprintf("provider %s is removed: no signal asks a classifier service", scalarAt(at.Node, "provider")),
				"drop the deployment; the decision model answers the signals"
		},
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			deleteEmpty(m.Root(), "global", "model_catalog", "deployments")
			return dropClassifierReferences(m, at.Key, scalarAt(at.Node, "provider"))
		},
	},
	{
		Code:  CodeRemovedField,
		Paths: bindingPaths,
		MatchAt: func(at FormMatch) bool {
			return signalBinding(at) && slices.Contains(classifierContracts, scalarAt(at.Node, "contract"))
		},
		Describe: func(at FormMatch) (string, string) {
			return fmt.Sprintf("binding %s asks a classifier head (%s), which no signal asks", at.Key, scalarAt(at.Node, "contract")),
				"drop it; the decision model answers the signal"
		},
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			if strings.HasPrefix(at.Path, "global.") {
				deleteEmpty(m.Root(), "global", "model_catalog", "bindings")
			} else {
				deleteEmpty(scopeRouting(m.Root(), at.Path), "model_bindings")
			}
			return nil
		},
	},
	{
		Code:     CodeRemovedField,
		Paths:    bindingPaths,
		MatchAt:  func(at FormMatch) bool { return liveBinding(at) && len(legacyBindingFields(at.Node)) > 0 },
		Describe: describeLegacyBindingFields,
		Migrate:  migrateLegacyBindingFields,
	},
	{
		Code:  CodeRemovedField,
		Paths: bindingPaths,
		MatchAt: func(at FormMatch) bool {
			return liveBinding(at) && (strings.HasPrefix(at.Key, SignalTypeSafety+".") || strings.HasPrefix(at.Key, SignalTypeComplexity+"."))
		},
		Describe: func(at FormMatch) (string, string) {
			collection, rule, _ := strings.Cut(at.Key, ".")
			return "a binding for one rule is removed", fmt.Sprintf("the rule's deployment: recipes[].signals.%s[%s].deployment", collection, rule)
		},
		Migrate: migrateRuleBinding,
	},
	{
		Code:    CodeRemovedSection,
		Paths:   []string{"global.model_catalog.external[]"},
		What:    "external models are removed",
		Becomes: "drop it; the model_role: memory_rewrite model is an openai_chat deployment global.stores.memory.query_rewrite names",
		Migrate: migrateExternalModel,
	},
	{
		Code:    CodeRemovedSection,
		Paths:   []string{"global.model_catalog.external"},
		Match:   func(node *yaml.Node, _ string) bool { return node.Kind != yaml.ScalarNode && len(node.Content) == 0 },
		What:    "external models are removed",
		Becomes: "drop it",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			deleteEmpty(m.Root(), "global", "model_catalog")
			return nil
		},
	},
	{
		Code:     CodeRemovedField,
		Paths:    []string{"providers.models[].backend_refs[]"},
		Match:    func(node *yaml.Node, _ string) bool { return len(legacyConnectionFields(node)) > 0 },
		Describe: describeLegacyConnection,
		Migrate:  migrateLegacyConnection,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"providers.models[].backend_refs[].trusted_gateway"},
		What:    "trusted_gateway is removed",
		Becomes: "auth: {credential: delegated}",
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.services.authz.gateway_credential"},
		What:    "gateway_credential is removed",
		Becomes: "global.services.authz.delegated_credential",
	},
}

// removedClassifierProviders served the removed classifier modules: the
// classify API, and an external[] model asked over it (http).
var removedClassifierProviders = []string{"classify", "http"}

// dropClassifierReferences removes what named a dropped classifier
// deployment: a binding, the deployment's admission gate, and the deployment
// of a rule, selector, judge or check in a recipe, which then asks its
// binding or the decision model. A deployment named anywhere else has no such
// default, so migrate refuses with the place to change.
func dropClassifierReferences(m *Migration, name, provider string) error {
	removed := fmt.Sprintf("deployment %s (provider %s), which is removed", name, provider)
	for _, pattern := range bindingPaths {
		for _, at := range matchPattern(m.Root(), strings.Split(pattern, ".")) {
			if at.Node.Value != name && scalarAt(at.Node, "deployment") != name {
				continue
			}
			m.Delete(at.Parent, at.Key)
			m.Note(at.Path, "binding "+at.Key+" named "+removed+": the decision model answers the signal")
			if strings.HasPrefix(at.Path, "global.") {
				deleteEmpty(m.Root(), "global", "model_catalog", "bindings")
			} else {
				deleteEmpty(scopeRouting(m.Root(), at.Path), "model_bindings")
			}
		}
	}
	if admission := mappingAt(m.Root(), "global", "model_catalog", "admission"); mappingValue(admission, name) != nil {
		m.Delete(admission, name)
		deleteEmpty(m.Root(), "global", "model_catalog", "admission")
		m.Changed("global.model_catalog.admission."+name, "the admission gate of "+removed+" is dropped with it")
	}
	for _, at := range matchPattern(m.Root(), []string{"**", "deployment"}) {
		if at.Node.Value != name {
			continue
		}
		if !strings.HasPrefix(at.Path, "recipes[") {
			return fmt.Errorf("%s names %s; name a deployment of another provider there", at.Path, removed)
		}
		m.Delete(at.Parent, "deployment")
		m.Note(strings.TrimSuffix(at.Path, ".deployment"), "it asked "+removed+": its binding, else the decision model, answers it")
	}
	return nil
}

var bindingPaths = slices.Concat(RoutingScopes("model_bindings.*"), []string{"global.model_catalog.bindings.*"})

// classifierContracts are the contracts of classifier heads, which a binding
// asked over the removed classify path.
var classifierContracts = []string{"label_distribution.v1", "label_scores.v1", "label_decision.v1", "score.v1", "token_spans.v1"}

// signalBinding reports whether a binding is one of a signal the decision
// model answers, which the application and embedding forms leave in place.
func signalBinding(at FormMatch) bool {
	return !removedBindingKey(at.Key, at.Node) && at.Key != "embedding"
}

// liveBinding reports whether a signal binding names a deployment that may
// answer it: one that asked no classifier head.
func liveBinding(at FormMatch) bool {
	return signalBinding(at) && !slices.Contains(classifierContracts, scalarAt(at.Node, "contract"))
}

// legacyBindingObjectFields chose a classifier head, its label mapping and
// the protocol it was asked over.
var legacyBindingObjectFields = []string{"contract", "adapter", "head", "mapping_path", "operating_point"}

func legacyBindingFields(binding *yaml.Node) []string {
	var fields []string
	for _, field := range legacyBindingObjectFields {
		if mappingValue(binding, field) != nil {
			fields = append(fields, field)
		}
	}
	return fields
}

func describeLegacyBindingFields(at FormMatch) (string, string) {
	return removedFields(at.Path, legacyBindingFields(at.Node)),
		fmt.Sprintf("%s: %s; the signal decides what the deployment answers", at.Key, scalarAt(at.Node, "deployment"))
}

// migrateLegacyBindingFields writes the binding as its deployment's name.
func migrateLegacyBindingFields(m *Migration, at FormMatch) error {
	m.Set(at.Parent, at.Key, Scalar(scalarAt(at.Node, "deployment")))
	return nil
}

// migrateRuleBinding writes a binding for one safety or complexity rule as
// the deployment of that rule, which then wins as the binding did: in its
// recipe, or, for a global binding, in each recipe whose own bindings name
// neither the rule nor its type. A rule that names a deployment keeps it.
func migrateRuleBinding(m *Migration, at FormMatch) error {
	deployment := at.Node.Value
	if at.Node.Kind == yaml.MappingNode {
		deployment = scalarAt(at.Node, "deployment")
	}
	collection, rule, _ := strings.Cut(at.Key, ".")
	recipes := []*yaml.Node{scopeRouting(m.Root(), at.Path)}
	global := strings.HasPrefix(at.Path, "global.")
	if global {
		recipes = sequenceAt(m.Root(), "recipes")
	}
	for _, recipe := range recipes {
		recipe = resolveAlias(recipe)
		bindings := mappingAt(recipe, "model_bindings")
		if global && (mappingValue(bindings, at.Key) != nil || mappingValue(bindings, collection) != nil) {
			continue
		}
		for _, item := range sequenceAt(mappingAt(recipe, "signals"), collection) {
			if scalarAt(item, "name") == rule && scalarAt(item, "deployment") == "" {
				m.Set(item, "deployment", Scalar(deployment))
			}
		}
	}
	m.Delete(at.Parent, at.Key)
	if global {
		deleteEmpty(m.Root(), "global", "model_catalog", "bindings")
	} else {
		deleteEmpty(scopeRouting(m.Root(), at.Path), "model_bindings")
	}
	return nil
}

// migrateDeploymentQuestionTypes moves question_types under capabilities,
// which won when both were set.
func migrateDeploymentQuestionTypes(m *Migration, at FormMatch) error {
	m.Delete(at.Parent, "question_types")
	capabilities := ensureMapping(at.Parent, "capabilities")
	if mappingValue(capabilities, "question_types") == nil {
		m.Set(capabilities, "question_types", at.Node)
	}
	return nil
}

// legacyExternalModel is an external[] entry as the Router read it.
type legacyExternalModel struct {
	Name      string `yaml:"name"`
	ModelRole string `yaml:"model_role"`
	Endpoint  struct {
		Address  string `yaml:"address"`
		Port     int    `yaml:"port"`
		Protocol string `yaml:"protocol"`
	} `yaml:"llm_endpoint"`
	ModelName        string               `yaml:"llm_model_name"`
	TimeoutSeconds   float64              `yaml:"llm_timeout_seconds"`
	AccessKey        string               `yaml:"access_key"`
	MaxTokens        int                  `yaml:"max_tokens"`
	Temperature      float64              `yaml:"temperature"`
	Reasoning        *GenerationReasoning `yaml:"reasoning"`
	MaxRequestBytes  int64                `yaml:"max_request_bytes"`
	MaxResponseBytes int64                `yaml:"max_response_bytes"`
}

// memoryRewriteRole marked the external[] model memory search rewrote its
// queries with.
const memoryRewriteRole = "memory_rewrite"

// rewritesMemory reports whether an external[] entry is the memory rewrite
// model: the first entry of that role with an address, while no query
// rewrite names a deployment.
func rewritesMemory(root, entry *yaml.Node) bool {
	if mappingAt(root, "global", "stores", "memory", "query_rewrite") != nil {
		return false
	}
	for _, candidate := range sequenceAt(root, "global", "model_catalog", "external") {
		candidate = resolveAlias(candidate)
		if scalarAt(candidate, "model_role") == memoryRewriteRole && strings.TrimSpace(scalarAt(mappingValue(candidate, "llm_endpoint"), "address")) != "" {
			return candidate == entry
		}
	}
	return false
}

// migrateExternalModel drops an external[] entry; the memory rewrite model
// becomes the openai_chat deployment the query rewrite names, and a key the
// entry wrote moves to an environment variable the operator sets.
func migrateExternalModel(m *Migration, at FormMatch) error {
	if rewritesMemory(m.Root(), at.Node) {
		var external legacyExternalModel
		if err := at.Node.Decode(&external); err != nil {
			return err
		}
		deployments := ensureMapping(m.Root(), "global", "model_catalog", "deployments")
		name := uniqueDeploymentName(deployments, cmp.Or(external.Name, "memory-rewrite"))
		deployment := external.chatDeployment()
		if external.AccessKey != "" {
			environment := inlineKeyEnvironment(name)
			deployment.Auth = &DeploymentAuth{APIKeyEnv: environment}
			m.Note("global.model_catalog.deployments."+name+".auth.api_key_env", fmt.Sprintf(
				"the key external model %s wrote (access_key) is not written in the migrated file; set %s to it before the Router starts", cmp.Or(external.Name, name), environment))
		}
		if err := m.SetValue(deployments, name, deployment); err != nil {
			return err
		}
		m.Set(ensureMapping(m.Root(), "global", "stores", "memory"), "query_rewrite", flowMapping("deployment", name))
	}
	removeItem(at.Parent, at.Node)
	if len(at.Parent.Content) == 0 {
		m.Delete(at.Above(2), "external")
	}
	return nil
}

// chatDeployment is the openai_chat deployment an external[] model is asked
// as: its llm_endpoint as a base URL (with /v1 when it names no path), its
// timeout, limits and generation settings.
func (e legacyExternalModel) chatDeployment() ModelDeployment {
	endpoint := legacyEndpointURL(e.Endpoint.Address, e.Endpoint.Port, e.Endpoint.Protocol)
	if parsed, err := url.Parse(endpoint); err == nil && strings.Trim(parsed.Path, "/") == "" {
		endpoint = strings.TrimRight(endpoint, "/") + "/v1"
	}
	deployment := ModelDeployment{Provider: OpenAIChatProvider, Endpoint: endpoint, Model: e.ModelName}
	if e.TimeoutSeconds > 0 {
		deployment.TimeoutMs = int(e.TimeoutSeconds * 1000)
	}
	if e.MaxRequestBytes > 0 || e.MaxResponseBytes > 0 {
		deployment.Limits = &DeploymentLimits{MaxRequestBytes: e.MaxRequestBytes, MaxResponseBytes: e.MaxResponseBytes}
	}
	if e.MaxTokens > 0 || e.Temperature != 0 || e.Reasoning != nil {
		deployment.Generation = &DeploymentGeneration{MaxTokens: e.MaxTokens, Reasoning: e.Reasoning}
		if e.Temperature != 0 {
			temperature := e.Temperature
			deployment.Generation.Temperature = &temperature
		}
	}
	return deployment
}

// legacyEndpointURL is the URL of an llm_endpoint: an address that is
// already a URL is used as it is and its port is ignored; a host takes the
// port and protocol (http by default).
func legacyEndpointURL(address string, port int, protocol string) string {
	address = strings.TrimSpace(address)
	if strings.Contains(address, "://") {
		return strings.TrimRight(address, "/")
	}
	protocol = cmp.Or(strings.ToLower(strings.TrimSpace(protocol)), "http")
	if port > 0 {
		address = net.JoinHostPort(strings.Trim(address, "[]"), strconv.Itoa(port))
	}
	return protocol + "://" + address
}

// uniqueDeploymentName is base, with a numeric suffix when the document
// declares a deployment of that name already.
func uniqueDeploymentName(deployments *yaml.Node, base string) string {
	base = cmp.Or(strings.Trim(strings.ReplaceAll(strings.TrimSpace(base), "/", "-"), "-"), "model")
	name := base
	for i := 2; mappingValue(deployments, name) != nil; i++ {
		name = fmt.Sprintf("%s-%d", base, i)
	}
	return name
}

// inlineKeyEnvironment names the environment variable a key written in the
// configuration moves to.
func inlineKeyEnvironment(deployment string) string {
	name := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r - 'a' + 'A'
		case (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9'):
			return r
		}
		return '_'
	}, deployment)
	return "VLLM_SR_" + name + "_API_KEY"
}

// legacyConnectionKeys are a backend's connection fields from before it
// connected with a deployment's block: endpoint, auth and headers.
var legacyConnectionKeys = []string{"base_url", "api_key_env", "auth_header", "auth_prefix", "extra_headers"}

func legacyConnectionFields(backend *yaml.Node) []string {
	var fields []string
	for _, field := range legacyConnectionKeys {
		if mappingValue(backend, field) != nil {
			fields = append(fields, field)
		}
	}
	return fields
}

func describeLegacyConnection(at FormMatch) (string, string) {
	return removedFields(at.Path, legacyConnectionFields(at.Node)),
		"endpoint (a full URL), auth {header, prefix, api_key_env} and headers, as a deployment connects"
}

// migrateLegacyConnection writes a backend's base_url as its endpoint (which
// it won over), its credential fields as auth and extra_headers as headers.
// It rewrites a copy, so the match still describes what it replaced.
func migrateLegacyConnection(m *Migration, at FormMatch) error {
	backend := &yaml.Node{Kind: yaml.MappingNode, Tag: at.Node.Tag, Style: at.Node.Style, Content: slices.Clone(at.Node.Content)}
	at.Parent.Content[at.Index] = backend
	if mappingValue(backend, "base_url") != nil {
		m.Delete(backend, "protocol")
		m.Delete(backend, "endpoint")
		renameCopiedKey(backend, "base_url", "endpoint")
	}
	auth := mappingNode()
	for _, field := range [][2]string{{"auth_header", "header"}, {"auth_prefix", "prefix"}, {"api_key_env", "api_key_env"}} {
		if value := mappingValue(backend, field[0]); value != nil {
			m.Set(auth, field[1], value)
		}
	}
	if len(auth.Content) > 0 {
		if mappingValue(backend, "auth") != nil {
			return fmt.Errorf("auth is set as well as auth_header, auth_prefix or api_key_env; keep auth")
		}
		auth.Style = yaml.FlowStyle
		replaceKeys(backend, []string{"auth_header", "auth_prefix", "api_key_env"}, "auth", auth)
	}
	if mappingValue(backend, "extra_headers") != nil {
		if mappingValue(backend, "headers") != nil {
			return fmt.Errorf("headers is set as well as extra_headers; keep headers")
		}
		renameCopiedKey(backend, "extra_headers", "headers")
	}
	return nil
}

// renameCopiedKey renames a key of a copied mapping without changing the
// key node the original shares.
func renameCopiedKey(mapping *yaml.Node, from, to string) {
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == from {
			key := *mapping.Content[i]
			key.Value = to
			mapping.Content[i] = &key
		}
	}
}
