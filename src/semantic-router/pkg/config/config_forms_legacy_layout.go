package config

import (
	"fmt"
	"maps"
	"net"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// previousReleaseLayout is what a form of the layouts before config v0.3
// becomes. Migrate writes it in the v0.3 layout, which the structure forms
// rewrite in the same run.
const previousReleaseLayout = "its place in the v0.5 layout"

// preV03Versions are the schema versions a configuration wrote before v0.3;
// the Router's flat layout wrote none.
var preV03Versions = []string{"v0.1", "v0.2"}

// operatorRatingBenchmark holds a model's quality_score: its operator's own
// rating of the model.
const operatorRatingBenchmark = "vllm-sr/operator-rating@1.0.0"

// previousReleaseForms are the layouts before config v0.3: flat top-level
// signals, decisions, endpoints, model settings and global blocks, the global
// layout's runtime, models and modules, the provider catalog's earlier
// fields, and evaluations kept on models and model cards. Migrate writes them
// in the v0.3 layout, which every later form reads, so one migration takes
// such a document to v0.5. They run in this order: model_config's backend
// references read the endpoints it names, and a model's reasoning the
// families the defaults define.
var previousReleaseForms = []ConfigForm{
	{
		Code:  CodeInvalidValue,
		Paths: []string{"version"},
		Match: func(node *yaml.Node, _ string) bool { return slices.Contains(preV03Versions, node.Value) },
		Describe: func(at FormMatch) (string, string) {
			return "config schema " + at.Node.Value + " is removed", "version: " + CanonicalConfigVersion
		},
		Whole: true,
		Migrate: func(m *Migration, at FormMatch) error {
			m.Set(at.Parent, "version", Scalar(CanonicalConfigVersion))
			keepDefaultAutoModelNames(m, at.Root)
			return nil
		},
	},
	{
		Code:     CodeRemovedSection,
		Paths:    []string{"model_config"},
		Describe: describePreviousRelease("top-level %s belongs to the layout before config v0.3"),
		Migrate:  inPreviousRelease(migrateModelConfig),
	},
	{
		Code:     CodeRemovedSection,
		Paths:    []string{"*"},
		Match:    func(_ *yaml.Node, path string) bool { return preV03TopLevelKey(path) },
		Describe: describePreviousRelease("top-level %s belongs to the layout before config v0.3"),
		Migrate:  inPreviousRelease(migratePreV03TopLevel),
	},
	{
		Code:     CodeRemovedField,
		Paths:    []string{"global.runtime", "global.models", "global.modules"},
		Describe: describePreviousRelease("%s belongs to the global layout before config v0.3"),
		Migrate:  inPreviousRelease(migratePreV03Global),
	},
	providerCatalogForm([]string{"providers.model_targets"}, migrateModelTargets),
	providerCatalogForm([]string{"providers.backends", "providers.auth_profiles"}, func(m *Migration, at FormMatch) error {
		m.Delete(at.Parent, at.Key)
		m.Note(at.Path, at.Path+" is dropped: no providers.model_targets entry names it")
		return nil
	}),
	providerCatalogForm([]string{"providers.models[].access"}, migrateModelAccess),
	providerCatalogForm([]string{"providers.models[].endpoints"}, migrateModelEndpoints),
	providerCatalogForm([]string{"providers.models[].access_key"}, migrateModelAccessKey),
	providerCatalogForm([]string{"providers.models[].reasoning_family"}, migrateReasoningFamily),
	providerCatalogForm([]string{
		"providers.default_model", "providers.reasoning_families", "providers.default_reasoning_effort",
		"providers.defaults.default_model", "providers.defaults.reasoning_families", "providers.defaults.default_reasoning_effort",
	}, migrateProviderDefaults),
	providerCatalogForm([]string{"providers.models[].backend_refs[].type"}, func(m *Migration, at FormMatch) error {
		if scalarAt(at.Parent, "provider") != "" {
			m.Delete(at.Parent, "type")
			return nil
		}
		return m.Rename(at.Parent, "type", "provider")
	}),
	providerCatalogForm([]string{"routing.models"}, func(m *Migration, at FormMatch) error {
		return m.Rename(at.Parent, "models", "modelCards")
	}),
	providerCatalogForm([]string{
		"providers.models[].quality_score", "providers.models[].evaluations",
		"routing.modelCards[].quality_score", "routing.modelCards[].evaluations",
	}, migrateModelEvaluations),
	{
		Code:  CodeRemovedPluginType,
		Paths: []string{"routing.decisions[].plugins[]"},
		Match: func(node *yaml.Node, _ string) bool {
			return hasType("pii")(node, "") || hasType("jailbreak")(node, "")
		},
		Describe: func(at FormMatch) (string, string) {
			kind := scalarAt(at.Node, "type")
			return "plugin " + kind + " was removed", "nothing; write a " + kind + " rule, and a decision that refuses the requests it matches"
		},
		Migrate: inPreviousRelease(migrateGuardPlugin),
	},
}

// migrateGuardPlugin removes a decision's pii or jailbreak plugin. Enabled,
// the first release blocked a request its classifier flagged on the decision;
// the next read a decision's policy from the signals its rules name, and the
// v0.3 layout has no such plugin.
func migrateGuardPlugin(m *Migration, at FormMatch) error {
	kind, configuration := scalarAt(at.Node, "type"), mappingValue(at.Node, "configuration")
	removeItem(at.Parent, at.Node)
	if len(at.Parent.Content) == 0 {
		m.Delete(at.Above(2), "plugins")
	}
	switch {
	case !enabledScalar(configuration):
		m.Changed(at.Path, "the disabled "+kind+" plugin is removed")
	case kind == "pii":
		m.Note(at.Path, "the pii plugin is dropped, and nothing blocks personal data now: it blocked personal data, but its pii_types_allowed, in the requests the decision routed; "+
			"write a pii rule with those types as ignore_labels and a decision that refuses the requests it matches")
	default:
		m.Note(at.Path, "the jailbreak plugin is dropped, and nothing blocks jailbreaks now: it blocked the jailbreaks in the requests the decision routed; "+
			"write a jailbreak rule and a decision that refuses the requests it matches")
	}
	return nil
}

// preV03SignalCollections are the flat signal collections, by the collection
// of routing.signals each is.
var preV03SignalCollections = map[string]string{
	"keyword_rules": "keywords", "embedding_rules": "embeddings", "categories": "domains",
	"fact_check_rules": "fact_check", "user_feedback_rules": "user_feedbacks", "reask_rules": "reasks",
	"preference_rules": "preferences", "language_rules": "language", "context_rules": "context",
	"structure_rules": "structure", "complexity_rules": "complexity", "modality_rules": "modality",
	"role_bindings": "role_bindings", "jailbreak": "jailbreak", "hallucination": "hallucination", "pii": "pii",
	"kb": "kb", "conversation": "conversation", "events": "events", "metadata": "metadata",
	"classifiers": "classifiers", "input_modality": "input_modality",
}

// preV03ProviderPlaces are the provider settings written at the top level, by
// their place under providers; the reasoning families stay where the models'
// reasoning reads them.
var preV03ProviderPlaces = map[string]string{
	"default_model": "defaults.model", "default_reasoning_effort": "defaults.reasoning_effort",
	"reasoning_families": "defaults.reasoning_families",
}

// preV03GlobalPlaces are the global blocks written at the top level or in
// global.runtime and global.modules, by their place under global.
var preV03GlobalPlaces = map[string]string{
	"strategy": "router.strategy", "auto_model_name": "router.auto_model_name", "auto_model_names": "router.auto_model_names",
	"list_backend_models": "router.list_backend_models", "include_config_models_in_list": "router.include_config_models_in_list",
	"clear_route_cache": "router.clear_route_cache", "model_selection": "router.model_selection", "config_source": "router.config_source",
	"streamed_body_mode": "router.streamed_body.enabled", "max_streamed_body_bytes": "router.streamed_body.max_bytes",
	"streamed_body_timeout_sec": "router.streamed_body.timeout_sec", "response_api": "services.response_api",
	"router_replay": "services.router_replay", "api": "services.api", "observability": "services.observability",
	"authz": "services.authz", "ratelimit": "services.ratelimit",
	"semantic_cache": "stores.response_cache", "response_cache": "stores.response_cache", "memory": "stores.memory",
	"vector_store": "stores.vector_store", "tools": "integrations.tools", "looper": "integrations.looper",
	"system_models": "model_catalog.system", "external_models": "model_catalog.external",
	"embedding_models": "model_catalog.embeddings.semantic", "bert_model": "model_catalog.embeddings.semantic",
	"prompt_compression": "model_catalog.modules.prompt_compression", "modality_detector": "model_catalog.modules.modality_detector",
	"prompt_guard": "model_catalog.modules.prompt_guard", "feedback_detector": "model_catalog.modules.feedback_detector",
	"classifier": "model_catalog.modules.classifier", "hallucination_mitigation": "model_catalog.modules.hallucination_mitigation",
}

// preV03BlockFields are the global blocks whose fields the v0.3 layout
// renamed, by field.
var preV03BlockFields = map[string]map[string]string{
	"bert_model": {"model_id": "bert_model_path", "use_cpu": "use_cpu", "threshold": "embedding_config.min_score_threshold"},
	"classifier": {"category_model": "domain", "pii_model": "pii", "mcp_category_model": "mcp", "preference_model": "preference"},
	"hallucination_mitigation": {
		"enabled": "enabled", "fact_check_model": "fact_check", "hallucination_model": "detector", "nli_model": "explainer",
	},
}

// preV03TopLevelKey reports whether a top-level key belongs to the layouts
// before config v0.3, but model_config, which a form of its own reads first.
func preV03TopLevelKey(key string) bool {
	_, signal := preV03SignalCollections[key]
	_, provider := preV03ProviderPlaces[key]
	_, global := preV03GlobalPlaces[key]
	return signal || provider || global ||
		slices.Contains([]string{"signals", "decisions", "vllm_endpoints", "provider_profiles", "evaluation_catalog", "jailbreak_mapping_path", "mom_registry"}, key)
}

func describePreviousRelease(what string) func(FormMatch) (string, string) {
	return func(at FormMatch) (string, string) {
		return fmt.Sprintf(what, at.Path), previousReleaseLayout
	}
}

// providerCatalogForm is a field of the provider catalog before config v0.3.
func providerCatalogForm(paths []string, migrate func(*Migration, FormMatch) error) ConfigForm {
	return ConfigForm{
		Code:     CodeRemovedField,
		Paths:    paths,
		Describe: describePreviousRelease("%s belongs to the provider catalog before config v0.3"),
		Migrate:  inPreviousRelease(migrate),
	}
}

// inPreviousRelease runs a rewrite of a layout before config v0.3 and gives a
// document that names no schema version the version migrate writes.
func inPreviousRelease(migrate func(*Migration, FormMatch) error) func(*Migration, FormMatch) error {
	return func(m *Migration, at FormMatch) error {
		if mappingValue(at.Root, "version") == nil {
			insertSection(at.Root, "version", Scalar(CanonicalConfigVersion))
			keepDefaultAutoModelNames(m, at.Root)
		}
		return migrate(m, at)
	}
}

// migratePreV03TopLevel writes a top-level key where the v0.3 layout holds
// it: routing, providers or global.
func migratePreV03TopLevel(m *Migration, at FormMatch) error {
	key, value := at.Key, resolveAlias(at.Node)
	m.Delete(at.Root, key)
	collection, signal := preV03SignalCollections[key]
	switch {
	case key == "signals" || key == "decisions":
		return mergeValue(ensureSection(at.Root, "routing"), key, value)
	case signal:
		return mergeValue(ensureMapping(ensureSection(at.Root, "routing"), "signals"), collection, value)
	case key == "vllm_endpoints" || key == "provider_profiles":
		m.Note(key, key+" is dropped: no model_config model names it")
		return nil
	case key == "evaluation_catalog":
		if mappingValue(at.Root, "evaluation") != nil {
			return fmt.Errorf("evaluation is set as well; keep evaluation")
		}
		insertSection(at.Root, "evaluation", value)
		return nil
	case key == "jailbreak_mapping_path":
		m.Changed(key, "top-level jailbreak_mapping_path is dropped: the Router read the jailbreak classifier's mapping only in prompt_guard")
		return nil
	case key == "mom_registry":
		m.Note(key, "mom_registry is dropped: a model's repository is the artifact of its deployment in global.model_catalog.deployments")
		return nil
	}
	if place, ok := preV03ProviderPlaces[key]; ok {
		return placeAt(ensureSection(at.Root, "providers"), place, value)
	}
	return placePreV03Global(m, at.Root, key, key, value)
}

// placePreV03Global writes a global block at its place under global.
func placePreV03Global(m *Migration, root *yaml.Node, path, key string, value *yaml.Node) error {
	place, ok := preV03GlobalPlaces[key]
	if !ok {
		m.Note(path, path+" has no counterpart; it is dropped")
		return nil
	}
	global := ensureSection(root, "global")
	if fields, renamed := preV03BlockFields[key]; renamed {
		if value.Kind != yaml.MappingNode || len(value.Content) == 0 {
			return nil
		}
		return movePreV03Fields(m, path, value, ensureMapping(global, strings.Split(place, ".")...), fields)
	}
	if (key == "semantic_cache" || key == "response_cache") && value.Kind == yaml.MappingNode {
		for _, field := range preV03CacheFields {
			if mappingValue(value, field) != nil {
				m.Delete(value, field)
				m.Note(path+"."+field, path+"."+field+" has no counterpart; it is dropped")
			}
		}
		if mappingValue(value, "enabled") == nil {
			// The cache was off unless enabled; the v0.3 store is on unless disabled.
			value.Content = append([]*yaml.Node{Scalar("enabled"), boolNode(false)}, value.Content...)
		}
	}
	return placeAt(global, place, value)
}

// preV03CacheFields are the cache's in-memory index and hybrid backend
// settings, which the v0.3 store does not hold.
var preV03CacheFields = []string{"use_hnsw", "hnsw_m", "hnsw_ef_construction", "max_memory_entries", "backend_config_path"}

// movePreV03Fields writes each field of a block at its renamed place.
func movePreV03Fields(m *Migration, path string, block, to *yaml.Node, fields map[string]string) error {
	for i := 0; block.Kind == yaml.MappingNode && i+1 < len(block.Content); i += 2 {
		field := block.Content[i].Value
		place, ok := fields[field]
		if !ok {
			m.Note(path+"."+field, path+"."+field+" has no counterpart; it is dropped")
			continue
		}
		if err := placeAt(to, place, block.Content[i+1]); err != nil {
			return fmt.Errorf("%s.%s: %w", path, field, err)
		}
	}
	return nil
}

// migratePreV03Global writes the blocks of global.runtime (its router block's
// keys included), global.models and global.modules at their places under
// global.
func migratePreV03Global(m *Migration, at FormMatch) error {
	block := resolveAlias(at.Node)
	m.Delete(at.Parent, at.Key)
	global := ensureSection(at.Root, "global")
	for i := 0; block.Kind == yaml.MappingNode && i+1 < len(block.Content); i += 2 {
		key, value := block.Content[i].Value, resolveAlias(block.Content[i+1])
		path := at.Path + "." + key
		var err error
		switch {
		case at.Key == "runtime" && key == "router" && value.Kind == yaml.MappingNode:
			for j := 0; err == nil && j+1 < len(value.Content); j += 2 {
				err = placePreV03Global(m, at.Root, path+"."+value.Content[j].Value, value.Content[j].Value, resolveAlias(value.Content[j+1]))
			}
		case at.Key == "models" && key == "embeddings" && mappingValue(value, "semantic") != nil:
			err = placeAt(global, "model_catalog.embeddings.semantic", mappingValue(value, "semantic"))
		case at.Key == "models" && (key == "system" || key == "external"):
			err = placeAt(global, "model_catalog."+key, value)
		case at.Key == "modules" && preV03GlobalPlaces[key] == "":
			err = placeAt(global, "model_catalog.modules."+key, value)
		case at.Key == "models":
			m.Note(path, path+" has no counterpart; it is dropped")
		default:
			err = placePreV03Global(m, at.Root, path, key, value)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

// preV03ModelFields are the fields of a model_config entry that a
// providers.models[] entry states as written.
func preV03ModelFields() []string {
	return slices.Concat([]string{"pricing", "api_format", "external_model_ids", "reasoning_family", "quality_score"}, ModelFactFields(), laterCardFields)
}

// migrateModelConfig writes each model_config entry as the providers.models[]
// entry of its name, with a backend reference for each of its preferred
// endpoints: the vllm_endpoints entry of that name, with its provider
// profile's connection.
func migrateModelConfig(m *Migration, at FormMatch) error {
	backends := map[string]*yaml.Node{}
	profiles := resolveAlias(mappingValue(at.Root, "provider_profiles"))
	endpoints := sequenceAt(at.Root, "vllm_endpoints")
	for _, endpoint := range endpoints {
		endpoint = resolveAlias(endpoint)
		backends[scalarAt(endpoint, "name")] = preV03Backend(m, endpoint, profiles)
	}
	named := map[string]bool{}
	models := ensureSequence(ensureSection(at.Root, "providers"), "models")
	config := resolveAlias(at.Node)
	for i := 0; config.Kind == yaml.MappingNode && i+1 < len(config.Content); i += 2 {
		name, params := config.Content[i].Value, resolveAlias(config.Content[i+1])
		path := "model_config." + name
		model := providerModelNamed(models, name)
		if model == nil {
			model = mappingNode("name", name)
			models.Content = append(models.Content, model)
		}
		key := mappingValue(params, "access_key")
		preferred := stringItems(resolveAlias(mappingValue(params, "preferred_endpoints")))
		own := hasAnyKey("backend_refs", "endpoints")(model, "")
		for j := 0; params.Kind == yaml.MappingNode && j+1 < len(params.Content); j += 2 {
			field, value := params.Content[j].Value, params.Content[j+1]
			var err error
			switch {
			case field == "model_id":
				err = setModelField(model, "provider_model_id", value)
			case (field == "preferred_endpoints" || field == "access_key") && own:
				m.Note(path+"."+field, path+"."+field+" is dropped: providers.models["+name+"] names its own endpoints")
			case field == "preferred_endpoints":
			case field == "access_key":
				if len(preferred) == 0 {
					m.Note(path+".access_key", path+".access_key is dropped: the model names no preferred_endpoints")
				}
			case slices.Contains(preV03ModelFields(), field):
				err = setModelField(model, field, value)
			default:
				m.Note(path+"."+field, path+"."+field+" has no counterpart; it is dropped")
			}
			if err != nil {
				return fmt.Errorf("%s.%s: %w", path, field, err)
			}
		}
		for _, endpoint := range preferred {
			backend := backends[endpoint]
			if backend == nil && !own {
				m.Note(path+".preferred_endpoints", "preferred endpoint "+endpoint+" names no vllm_endpoints entry; it is dropped")
			}
			named[endpoint] = true
			if backend == nil || own {
				continue
			}
			backend = cloneNode(backend)
			if key != nil && !hasAnyKey("api_key", "api_key_env")(backend, "") {
				setNode(backend, "api_key", key)
			}
			refs := ensureSequence(model, "backend_refs")
			refs.Content = append(refs.Content, backend)
		}
		if CanonicalAPIFormat(scalarAt(model, "api_format")) == APIFormatMessages && mappingValue(model, "backend_refs") == nil && ownsTransport(at.Root) {
			// The v0.3 layout's one implicit endpoint: an Anthropic model the
			// Router itself sent to the Anthropic API.
			setNode(model, "backend_refs", sequenceOf(flowMapping("provider", "anthropic")))
		}
	}
	for _, endpoint := range endpoints {
		if name := scalarAt(resolveAlias(endpoint), "name"); !named[name] {
			m.Note("vllm_endpoints", "vllm_endpoints["+name+"] is dropped: no model_config model names it")
		}
	}
	for _, key := range []string{"model_config", "vllm_endpoints", "provider_profiles"} {
		m.Delete(at.Root, key)
	}
	return nil
}

// ownsTransport reports whether the Router sends the document's requests
// itself: an explicit empty listener list leaves that to an external gateway.
func ownsTransport(root *yaml.Node) bool {
	listeners := resolveAlias(mappingValue(root, "listeners"))
	return listeners == nil || len(listeners.Content) > 0
}

// preV03Backend is the backend reference of a vllm_endpoints entry: its
// address and port as the endpoint, its type as the provider, and the
// connection of its provider profile, whose type wins.
func preV03Backend(m *Migration, endpoint, profiles *yaml.Node) *yaml.Node {
	name := scalarAt(endpoint, "name")
	path := "vllm_endpoints[" + name + "]"
	backend := mappingNode("name", name)
	if address := strings.TrimSpace(scalarAt(endpoint, "address")); address != "" {
		if port := scalarAt(endpoint, "port"); port != "" && port != "0" {
			address = net.JoinHostPort(address, port)
		}
		setNode(backend, "endpoint", Scalar(address))
	}
	profileName := scalarAt(endpoint, "provider_profile")
	profile := resolveAlias(mappingValue(profiles, profileName))
	if profileName != "" && profile == nil {
		m.Note(path, "provider profile "+profileName+" names no provider_profiles entry; it is dropped")
	}
	provider := scalarAt(endpoint, "type")
	if scalarAt(profile, "type") != "" {
		provider = scalarAt(profile, "type")
	}
	if provider != "" {
		setNode(backend, "provider", Scalar(provider))
	}
	copyFields(m, path, endpoint, backend, []string{"protocol", "weight", "api_key", "api_key_env"}, "name", "address", "port", "type", "provider_profile")
	copyFields(m, "provider_profiles."+profileName, profile, backend, []string{"base_url", "auth_header", "auth_prefix", "extra_headers", "api_version", "chat_path"}, "type")
	return backend
}

// copyFields copies the fields of from to to, skipping those another rewrite
// reads; any other field has no counterpart.
func copyFields(m *Migration, path string, from, to *yaml.Node, fields []string, read ...string) {
	for i := 0; from != nil && from.Kind == yaml.MappingNode && i+1 < len(from.Content); i += 2 {
		field := from.Content[i].Value
		switch {
		case slices.Contains(read, field):
		case slices.Contains(fields, field):
			setNode(to, field, from.Content[i+1])
		default:
			m.Note(path+"."+field, path+"."+field+" has no counterpart; it is dropped")
		}
	}
}

// migrateModelTargets writes each model target, which named its backends and
// auth profile, as the providers.models[] entry of its name.
func migrateModelTargets(m *Migration, at FormMatch) error {
	providers := at.Parent
	backends := resolveAlias(mappingValue(providers, "backends"))
	profiles := resolveAlias(mappingValue(providers, "auth_profiles"))
	models := ensureSequence(providers, "models")
	targets := resolveAlias(at.Node)
	for i := 0; targets.Kind == yaml.MappingNode && i+1 < len(targets.Content); i += 2 {
		name, target := targets.Content[i].Value, resolveAlias(targets.Content[i+1])
		path := at.Path + "." + name
		if providerModelNamed(models, name) != nil {
			m.Note(path, path+" is dropped: providers.models["+name+"] describes the model")
			continue
		}
		model := mappingNode("name", name)
		profile := resolveAlias(mappingValue(profiles, scalarAt(target, "auth_ref")))
		copyFields(m, path, target, model, []string{"provider_model_id", "pricing", "api_format", "external_model_ids"}, "auth_ref", "backend_refs")
		for _, backendName := range stringItems(resolveAlias(mappingValue(target, "backend_refs"))) {
			backend := resolveAlias(mappingValue(backends, backendName))
			if backend == nil {
				m.Note(path, "backend "+backendName+" names no providers.backends entry; it is dropped")
				continue
			}
			ref := mappingNode("name", backendName)
			if scalarAt(backend, "provider") == "" && scalarAt(backend, "type") != "" {
				setNode(ref, "provider", mappingValue(backend, "type"))
			}
			copyFields(m, "providers.backends."+backendName, backend, ref, []string{
				"endpoint", "protocol", "weight", "provider", "base_url", "auth_header", "auth_prefix", "extra_headers", "api_version", "chat_path",
			}, "type")
			for _, key := range []string{"api_key", "api_key_env"} {
				if value := mappingValue(profile, key); value != nil {
					setNode(ref, key, value)
				}
			}
			refs := ensureSequence(model, "backend_refs")
			refs.Content = append(refs.Content, ref)
		}
		models.Content = append(models.Content, model)
	}
	for _, key := range []string{"model_targets", "backends", "auth_profiles"} {
		m.Delete(providers, key)
	}
	return nil
}

// migrateModelAccess writes the fields of a model's access block on the
// model.
func migrateModelAccess(m *Migration, at FormMatch) error {
	model, access := at.Parent, resolveAlias(at.Node)
	m.Delete(model, "access")
	for i := 0; access.Kind == yaml.MappingNode && i+1 < len(access.Content); i += 2 {
		field := access.Content[i].Value
		if !slices.Contains([]string{"provider_model_id", "pricing", "reliability", "api_format", "external_model_ids", "backend_refs"}, field) {
			m.Note(at.Path+"."+field, at.Path+"."+field+" has no counterpart; it is dropped")
			continue
		}
		if err := setModelField(model, field, access.Content[i+1]); err != nil {
			return fmt.Errorf("access.%s: %w", field, err)
		}
	}
	return nil
}

// migrateModelEndpoints writes a model's endpoints as its backend references,
// each with the model's access key.
func migrateModelEndpoints(m *Migration, at FormMatch) error {
	model := at.Parent
	if mappingValue(model, "backend_refs") != nil {
		return fmt.Errorf("backend_refs is set as well; keep backend_refs")
	}
	key := mappingValue(model, "access_key")
	refs := sequenceOf()
	for index, endpoint := range sequenceItems(resolveAlias(at.Node)) {
		endpoint = resolveAlias(endpoint)
		path := at.Path + "[" + itemLabel(endpoint, index) + "]"
		if scalarAt(endpoint, "endpoint") == "" {
			m.Note(path, path+" names no endpoint; it is dropped")
			continue
		}
		name := scalarAt(endpoint, "name")
		if name == "" {
			name = "primary"
		}
		backend := mappingNode("name", name)
		if provider := mappingValue(endpoint, "type"); provider != nil {
			setNode(backend, "provider", provider)
		}
		copyFields(m, path, endpoint, backend, []string{"endpoint", "protocol", "weight"}, "name", "type")
		if key != nil {
			setNode(backend, "api_key", key)
		}
		refs.Content = append(refs.Content, backend)
	}
	m.Delete(model, "access_key")
	m.Set(model, "endpoints", refs)
	return m.Rename(model, "endpoints", "backend_refs")
}

// migrateModelAccessKey writes a model's access key on each of its backend
// references that names no credential.
func migrateModelAccessKey(m *Migration, at FormMatch) error {
	model := at.Parent
	m.Delete(model, "access_key")
	refs := sequenceAt(model, "backend_refs")
	if len(refs) == 0 {
		m.Note(at.Path, at.Path+" is dropped: the model has no backend_refs")
	}
	for _, ref := range refs {
		if ref = resolveAlias(ref); !hasAnyKey("api_key", "api_key_env", "auth")(ref, "") {
			setNode(ref, "api_key", at.Node)
		}
	}
	return nil
}

// migrateReasoningFamily writes a model's reasoning family as its reasoning:
// the family's definition in the provider defaults, else the family by name.
// An effort-based definition names the efforts it sends; one written before
// config v0.3 sent the effort a request or the defaults chose.
func migrateReasoningFamily(m *Migration, at FormMatch) error {
	model, family := at.Parent, at.Node.Value
	m.Delete(model, "reasoning_family")
	if family == "" || mappingValue(model, "reasoning") != nil {
		return nil
	}
	providers := mappingAt(at.Root, "providers")
	for _, families := range []*yaml.Node{mappingValue(mappingAt(providers, "defaults"), "reasoning_families"), mappingValue(providers, "reasoning_families")} {
		definition := resolveAlias(mappingValue(resolveAlias(families), family))
		if definition == nil {
			continue
		}
		definition = cloneNode(definition)
		effort := scalarAt(definition, "type") == "reasoning_effort" || scalarAt(definition, "type") == "top_level_reasoning_effort"
		if effort && mappingValue(definition, "levels") == nil {
			levels := []string{"low", "medium", "high"}
			if chosen := preV03DefaultEffort(providers); chosen != "" && !slices.Contains(levels, chosen) {
				levels = append(levels, chosen)
			}
			setNode(definition, "levels", stringSequence(levels...))
			m.Changed(at.Path, "reasoning family "+family+" names no effort levels; it sends "+strings.Join(levels, ", "))
		}
		if mappingValue(definition, "modes") == nil && disablesReasoning(at.Root, scalarAt(model, "name")) {
			// Before v0.3 use_reasoning switched any family off; a family now
			// lists the modes it allows.
			setNode(definition, "modes", stringSequence(ReasoningModeEnabled, ReasoningModeDisabled))
			setNode(definition, "default_mode", Scalar(ReasoningModeEnabled))
			if effort {
				m.Note(at.Path, "with use_reasoning: false the Router kept the request's own reasoning effort for family "+family+
					"; disabled reasoning now sends the effort none, unless reasoning.disabled names the effort to send")
			} else {
				m.Changed(at.Path, "reasoning family "+family+" lists the modes enabled and disabled, which use_reasoning chooses")
			}
		}
		setNode(model, "reasoning", definition)
		return nil
	}
	setNode(model, "reasoning", flowMapping("family", family))
	return nil
}

// disablesReasoning reports whether a decision asks a model without
// reasoning: a model reference whose use_reasoning is not true.
func disablesReasoning(root *yaml.Node, model string) bool {
	var decisions []*yaml.Node
	decisions = append(decisions, sequenceAt(mappingAt(root, "routing"), "decisions")...)
	for _, recipe := range sequenceAt(root, "recipes") {
		decisions = append(decisions, sequenceAt(recipe, "decisions")...)
		decisions = append(decisions, sequenceAt(mappingAt(recipe, "routing"), "decisions")...)
	}
	for _, decision := range decisions {
		for _, ref := range sequenceAt(resolveAlias(decision), "modelRefs") {
			if ref = resolveAlias(ref); scalarAt(ref, "model") == model && scalarAt(ref, "use_reasoning") != "true" {
				return true
			}
		}
	}
	return false
}

// preV03DefaultEffort is the reasoning effort the provider defaults choose,
// wherever a layout before config v0.3 wrote it.
func preV03DefaultEffort(providers *yaml.Node) string {
	defaults := mappingAt(providers, "defaults")
	for _, effort := range []string{scalarAt(defaults, "reasoning_effort"), scalarAt(defaults, "default_reasoning_effort"), scalarAt(providers, "default_reasoning_effort")} {
		if effort != "" {
			return effort
		}
	}
	return ""
}

// migrateProviderDefaults writes the default model and reasoning effort as
// the provider defaults; the reasoning families are in the reasoning of the
// models that named them.
func migrateProviderDefaults(m *Migration, at FormMatch) error {
	m.Delete(at.Parent, at.Key)
	defaults := ensureMapping(mappingAt(at.Root, "providers"), "defaults")
	var err error
	switch at.Key {
	case "default_model":
		err = mergeValue(defaults, "model", at.Node)
	case "default_reasoning_effort":
		err = mergeValue(defaults, "reasoning_effort", at.Node)
	}
	deleteEmpty(at.Root, "providers", "defaults")
	return err
}

// migrateModelEvaluations writes a model's or model card's quality_score and
// evaluations as evaluation records of its catalog identity; a card left with
// only its name is dropped.
func migrateModelEvaluations(m *Migration, at FormMatch) error {
	holder := at.Parent
	model := scalarAt(holder, "name")
	if catalog := scalarAt(holder, "catalog"); catalog != "" {
		model = catalog
	}
	records := ensureSequence(ensureSection(at.Root, "evaluation"), "records")
	if at.Key == "quality_score" {
		records.Content = append(records.Content, mappingNode("model", model, "benchmark", operatorRatingBenchmark, "metrics", mappingNode("score", at.Node)))
	}
	for _, record := range sequenceItems(resolveAlias(at.Node)) {
		record = cloneNode(resolveAlias(record))
		deleteKey(record, "model")
		record.Content = append([]*yaml.Node{Scalar("model"), Scalar(model)}, record.Content...)
		records.Content = append(records.Content, record)
	}
	m.Delete(holder, at.Key)
	routing := mappingAt(at.Root, "routing")
	if cards := resolveAlias(mappingValue(routing, "modelCards")); cards != nil && strings.HasPrefix(at.Path, "routing.") && len(holder.Content) == 2 {
		removeItem(cards, holder)
		if len(cards.Content) == 0 {
			m.Delete(routing, "modelCards")
		}
	}
	return nil
}

// providerModelNamed is the providers.models[] entry of a name.
func providerModelNamed(models *yaml.Node, name string) *yaml.Node {
	for _, model := range models.Content {
		if model = resolveAlias(model); scalarAt(model, "name") == name {
			return model
		}
	}
	return nil
}

// setModelField writes a field a model does not state, or states the same.
func setModelField(model *yaml.Node, field string, value *yaml.Node) error {
	if written := resolveAlias(mappingValue(model, field)); written != nil {
		if written.Kind == yaml.ScalarNode && resolveAlias(value).Kind == yaml.ScalarNode && written.Value == resolveAlias(value).Value {
			return nil
		}
		return fmt.Errorf("providers.models[%s].%s is set as well; keep one", scalarAt(model, "name"), field)
	}
	setNode(model, field, value)
	return nil
}

// mergeValue writes value under key: a key the mapping lacks is set, two
// sequences are joined, two mappings merged key by key and an equal scalar
// kept; any other pair is refused.
func mergeValue(mapping *yaml.Node, key string, value *yaml.Node) error {
	written, value := resolveAlias(mappingValue(mapping, key)), resolveAlias(value)
	switch {
	case written == nil:
		setNode(mapping, key, value)
	case written.Kind == yaml.SequenceNode && value.Kind == yaml.SequenceNode:
		written.Content = append(written.Content, value.Content...)
	case written.Kind == yaml.MappingNode && value.Kind == yaml.MappingNode:
		for i := 0; i+1 < len(value.Content); i += 2 {
			if err := mergeValue(written, value.Content[i].Value, value.Content[i+1]); err != nil {
				return err
			}
		}
	case written.Kind != yaml.ScalarNode || value.Kind != yaml.ScalarNode || written.Value != value.Value:
		return fmt.Errorf("%s is set twice; keep one", key)
	}
	return nil
}

// placeAt merges value at a dotted place under mapping.
func placeAt(mapping *yaml.Node, place string, value *yaml.Node) error {
	keys := strings.Split(place, ".")
	return mergeValue(ensureMapping(mapping, keys[:len(keys)-1]...), keys[len(keys)-1], value)
}

// layoutOrder is the order of a document's top-level sections.
var layoutOrder = []string{"version", "listeners", "providers", "evaluation", "routing", "entrypoints", "recipes", "global", "setup"}

// ensureSection is the top-level mapping of a section, which a rewrite adds
// in its place among the document's sections.
func ensureSection(root *yaml.Node, key string) *yaml.Node {
	if section := mappingAt(root, key); section != nil {
		return section
	}
	section := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	deleteKey(root, key)
	insertSection(root, key, section)
	return section
}

// insertSection writes a top-level section before the first section that
// follows it in the layout.
func insertSection(root *yaml.Node, key string, value *yaml.Node) {
	rank, at := slices.Index(layoutOrder, key), len(root.Content)
	for i := 0; i+1 < len(root.Content); i += 2 {
		if slices.Index(layoutOrder, root.Content[i].Value) > rank {
			at = i
			break
		}
	}
	root.Content = slices.Insert(root.Content, at, Scalar(key), value)
}

func cloneNode(node *yaml.Node) *yaml.Node {
	copied := *node
	copied.Content = make([]*yaml.Node, len(node.Content))
	for i, child := range node.Content {
		copied.Content[i] = cloneNode(child)
	}
	return &copied
}

// legacyRouterForms are router settings the previous release read that
// entrypoints and in-process Loopers replaced: the auto model names, the
// backend model listing switch's earlier name and the Looper's loopback
// endpoint.
var legacyRouterForms = []ConfigForm{
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.router.auto_model_names", "global.router.auto_model_name"},
		What:    "the auto model names are removed",
		Becomes: "the model_names of the default recipe's entrypoint",
		Migrate: migrateAutoModelNames,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.router.include_config_models_in_list"},
		What:    "include_config_models_in_list is removed",
		Becomes: "global.router.list_backend_models",
		Migrate: func(m *Migration, at FormMatch) error {
			if mappingValue(at.Parent, "list_backend_models") != nil {
				m.Delete(at.Parent, "include_config_models_in_list")
				return nil
			}
			return m.Rename(at.Parent, "include_config_models_in_list", "list_backend_models")
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.integrations.looper.endpoint"},
		What:    "the Looper endpoint is removed",
		Becomes: "nothing; the Router makes a Looper's model calls in process, through each model's providers.models[].backend_refs",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, "endpoint")
			deleteEmpty(m.Root(), "global", "integrations", "looper")
			return nil
		},
	},
}

// previousAutoModelNames are the auto model names the previous release
// served: auto_model_names as written, else vllm-sr/auto, auto and
// auto_model_name (MoM unless set), trimmed and once each.
func previousAutoModelNames(router *yaml.Node) []string {
	var names []string
	if list := resolveAlias(mappingValue(router, "auto_model_names")); list != nil && list.Kind == yaml.SequenceNode {
		names = stringItems(list)
	} else {
		single := strings.TrimSpace(scalarAt(router, "auto_model_name"))
		if single == "" {
			single = "MoM"
		}
		names = []string{DefaultEntrypointModel, "auto", single}
	}
	var kept []string
	for _, name := range names {
		if name = strings.TrimSpace(name); name != "" && !slices.Contains(kept, name) {
			kept = append(kept, name)
		}
	}
	return kept
}

// keepDefaultAutoModelNames writes, for a document of an earlier layout that
// names no auto model name, the names the previous release served it under,
// vllm-sr/auto, auto and MoM (releases before v0.4 served auto and MoM), as
// its auto_model_names, which the auto model name form then rewrites.
func keepDefaultAutoModelNames(m *Migration, root *yaml.Node) {
	for _, key := range []string{"auto_model_names", "auto_model_name"} {
		if len(matchPattern(root, []string{"**", key})) > 0 {
			return
		}
	}
	names := previousAutoModelNames(nil)
	setNode(ensureMapping(ensureSection(root, "global"), "router"), "auto_model_names", stringSequence(names...))
	m.Changed("global.router.auto_model_names", "no auto model names are written: the default routing served "+strings.Join(names, ", ")+" (releases before v0.4 served auto and MoM), and migrate keeps them")
}

// looperModelNameForms are the model names a Looper algorithm served, which
// entrypoints replaced. Migrate rewrites them after every other form, so the
// recipe it writes for them copies the default recipe as migrated.
var looperModelNameForms = []ConfigForm{
	{
		Code: CodeRemovedField,
		Paths: []string{
			"global.integrations.looper.remom.model_names",
			"global.integrations.looper.fusion.model_names",
			"global.integrations.looper.flow.model_names",
		},
		Describe: func(at FormMatch) (string, string) {
			return "the model names of a Looper algorithm are removed",
				"an entrypoint whose model_names name a recipe with a decision that runs the " + looperAlgorithms[looperAlgorithm(at)].decisions + " algorithm"
		},
		Migrate: migrateLooperModelNames,
	},
}

// looperAlgorithms are, for each Looper algorithm, the algorithm type of the
// decisions its model names ran and the name it served when none was written.
var looperAlgorithms = map[string]struct{ decisions, builtin string }{
	"remom":  {DecisionAlgorithmReMoM, DefaultReMoMModelName},
	"fusion": {DecisionAlgorithmFusion, DefaultFusionModelName},
	"flow":   {DecisionAlgorithmWorkflows, DefaultFlowModelName},
}

func looperAlgorithm(at FormMatch) string {
	return strings.TrimSuffix(strings.TrimPrefix(at.Path, "global.integrations.looper."), ".model_names")
}

// keepLooperModelNames writes, for each Looper algorithm that a decision of a
// v0.3 document's default routing runs and that names no model names, the
// built-in name the previous release served it under as its model names,
// which the Looper model name form then rewrites.
func keepLooperModelNames(m *Migration, root *yaml.Node) {
	routing := resolveAlias(mappingValue(root, "routing"))
	if routing == nil {
		routing = resolveAlias(mappingValue(recipeNamed(root, string(DefaultRecipeName)), "routing"))
	}
	for _, algorithm := range slices.Sorted(maps.Keys(looperAlgorithms)) {
		looper := looperAlgorithms[algorithm]
		runs := slices.ContainsFunc(sequenceAt(routing, "decisions"), func(decision *yaml.Node) bool {
			return scalarAt(resolveAlias(mappingValue(resolveAlias(decision), "algorithm")), "type") == looper.decisions
		})
		if !runs || mappingValue(mappingAt(root, "global", "integrations", "looper", algorithm), "model_names") != nil {
			continue
		}
		setNode(ensureMapping(ensureSection(root, "global"), "integrations", "looper", algorithm), "model_names", stringSequence(looper.builtin))
		m.Changed("global.integrations.looper."+algorithm+".model_names", fmt.Sprintf("no %s model names are written: the previous release served %s, and migrate keeps it", algorithm, looper.builtin))
	}
}

// servedOnlyEntrypoints reports whether a v0.3 document was written for main
// between v0.4 and v0.5, which refused auto and Looper model names and served
// only its entrypoints: an entrypoint names a name v0.4 served by default,
// which v0.4 refused as an entrypoint name.
func servedOnlyEntrypoints(root *yaml.Node) bool {
	reserved := previousAutoModelNames(mappingAt(root, "global", "router"))
	for algorithm, looper := range looperAlgorithms {
		if mappingValue(mappingAt(root, "global", "integrations", "looper", algorithm), "model_names") == nil {
			reserved = append(reserved, looper.builtin)
		}
	}
	served := chatEntrypointNames(root)
	return slices.ContainsFunc(reserved, func(name string) bool { return served[name] })
}

// migrateLooperModelNames keeps what the model names of a Looper algorithm
// did: a request for one ran the default recipe with only its decisions of
// that algorithm. The names become an entrypoint of a recipe migrate writes,
// a copy of the default recipe with those decisions alone. Names an entrypoint
// serves already, and names no decision served, are dropped.
func migrateLooperModelNames(m *Migration, at FormMatch) error {
	algorithm := looperAlgorithm(at)
	names := looperModelNames(at.Node, looperAlgorithms[algorithm].builtin)
	m.Delete(at.Parent, "model_names")
	deleteEmpty(m.Root(), "global", "integrations", "looper", algorithm)
	if len(names) == 0 {
		m.Changed(at.Path, fmt.Sprintf("the %s model names are dropped: they name no model", algorithm))
		return nil
	}
	served := chatEntrypointNames(m.Root())
	if mapped := slices.DeleteFunc(slices.Clone(names), func(name string) bool { return !served[name] }); len(mapped) > 0 {
		m.Changed(at.Path, fmt.Sprintf("the %s model names %s are dropped: an entrypoint serves them already", algorithm, strings.Join(mapped, ", ")))
		names = slices.DeleteFunc(names, func(name string) bool { return served[name] })
	}
	if len(names) == 0 {
		return nil
	}
	recipe := defaultRecipeOf(m.Root())
	source, decisionType := scalarAt(recipe, "name"), looperAlgorithms[algorithm].decisions
	var decisions []*yaml.Node
	var decisionNames []string
	for _, decision := range sequenceAt(recipe, "decisions") {
		decision = resolveAlias(decision)
		if scalarAt(resolveAlias(mappingValue(decision, "algorithm")), "type") == decisionType {
			decisions = append(decisions, detachedCopy(decision))
			decisionNames = append(decisionNames, scalarAt(decision, "name"))
		}
	}
	if len(decisions) == 0 {
		m.Changed(at.Path, fmt.Sprintf("the %s model names %s are dropped: no decision of the default recipe runs the %s algorithm, so they served no request", algorithm, strings.Join(names, ", "), decisionType))
		return nil
	}
	name := unusedRecipeName(m.Root(), source+"-"+algorithm)
	copied := mappingNode("name", name, "description", fmt.Sprintf("The %s decisions of recipe %s", decisionType, source))
	for i := 0; i+1 < len(recipe.Content); i += 2 {
		switch key := recipe.Content[i].Value; key {
		case "name", "description":
		case "decisions":
			copied.Content = append(copied.Content, detachedCopy(recipe.Content[i]), sequenceOf(decisions...))
		default:
			copied.Content = append(copied.Content, detachedCopy(recipe.Content[i]), detachedCopy(recipe.Content[i+1]))
		}
	}
	recipes := ensureSequence(m.Root(), "recipes")
	recipes.Content = append(recipes.Content, copied)
	entrypoints := ensureSequence(m.Root(), "entrypoints")
	entrypoints.Content = append(entrypoints.Content, mappingNode("model_names", stringSequence(names...), "recipe", name))
	if !slices.ContainsFunc(entrypoints.Content, func(e *yaml.Node) bool { return scalarAt(resolveAlias(e), "default") == "true" }) {
		markRecipeEntrypointDefault(m.Root(), source)
	}
	m.Changed(at.Path, fmt.Sprintf("the %s model names %s are served by recipe %s, which migrate wrote: recipe %s with only its %s decisions (%s); a request none of them matches now goes to the default model instead of failing",
		algorithm, strings.Join(names, ", "), name, source, decisionType, strings.Join(decisionNames, ", ")))
	if slices.ContainsFunc(m.notes, func(note ConfigWarning) bool {
		return note.Code == migrationActionCode && notesCopiedPlace(note.Field, recipePath(source), decisionNames)
	}) {
		m.Note(recipePath(name), fmt.Sprintf("it copies recipe %s: what the notes on %s ask applies to it too", source, recipePath(source)))
	}
	return nil
}

// notesCopiedPlace reports whether a note's path is in the recipe at prefix
// outside its decisions, or in one of the decisions named.
func notesCopiedPlace(path, prefix string, decisions []string) bool {
	rest, ok := strings.CutPrefix(path, prefix)
	if !ok || (rest != "" && !strings.HasPrefix(rest, ".")) {
		return false
	}
	decision, ok := strings.CutPrefix(rest, ".decisions[")
	return !ok || slices.ContainsFunc(decisions, func(name string) bool { return strings.HasPrefix(decision, name+"]") })
}

// looperModelNames are the names a Looper algorithm served: those written,
// trimmed and once each, or its built-in name when none is written.
func looperModelNames(node *yaml.Node, builtin string) []string {
	if node.Tag == "!!null" || (node.Kind == yaml.SequenceNode && len(node.Content) == 0) {
		return []string{builtin}
	}
	var names []string
	for _, name := range stringItems(node) {
		if name = strings.TrimSpace(name); name != "" && !slices.Contains(names, name) {
			names = append(names, name)
		}
	}
	return names
}

// chatEntrypointNames are the model names the chat entrypoints serve.
func chatEntrypointNames(root *yaml.Node) map[string]bool {
	served := map[string]bool{}
	for _, entrypoint := range sequenceAt(root, "entrypoints") {
		entrypoint = resolveAlias(entrypoint)
		if api := scalarAt(entrypoint, "api"); api == "" || api == string(ChatAPI) {
			for _, name := range stringItems(resolveAlias(mappingValue(entrypoint, "model_names"))) {
				served[strings.TrimSpace(name)] = true
			}
		}
	}
	return served
}

// defaultRecipeOf is the recipe that serves the default routing: that of the
// entrypoint marked default, else the only recipe, else the one named default.
func defaultRecipeOf(root *yaml.Node) *yaml.Node {
	for _, entrypoint := range sequenceAt(root, "entrypoints") {
		if entrypoint = resolveAlias(entrypoint); scalarAt(entrypoint, "default") == "true" {
			return resolveAlias(recipeNamed(root, scalarAt(entrypoint, "recipe")))
		}
	}
	if recipes := sequenceAt(root, "recipes"); len(recipes) == 1 {
		return resolveAlias(recipes[0])
	}
	return resolveAlias(recipeNamed(root, string(DefaultRecipeName)))
}

// unusedRecipeName is name, or name with the first numeric suffix no recipe
// takes.
func unusedRecipeName(root *yaml.Node, name string) string {
	candidate := name
	for suffix := 2; recipeNamed(root, candidate) != nil; suffix++ {
		candidate = fmt.Sprintf("%s-%d", name, suffix)
	}
	return candidate
}

// detachedCopy copies a node with its aliases resolved and without anchors,
// so the copy shares no node, and declares no anchor, of the document.
func detachedCopy(node *yaml.Node) *yaml.Node {
	node = resolveAlias(node)
	copied := *node
	copied.Anchor = ""
	copied.Content = make([]*yaml.Node, len(node.Content))
	for i, child := range node.Content {
		copied.Content[i] = detachedCopy(child)
	}
	return &copied
}

// migrateAutoModelNames keeps the names that reached the default recipe as
// auto model names, as the previous release read them. They become the
// model_names of the entrypoint migrate added for the built-in name, or of a
// new entrypoint of the default recipe: beside one the document wrote, whose
// names stay its own, or in front of the others. A name another entrypoint
// serves already stays that entrypoint's.
func migrateAutoModelNames(m *Migration, at FormMatch) error {
	if mappingValue(at.Parent, "auto_model_names") == nil && mappingValue(at.Parent, "auto_model_name") == nil {
		return nil
	}
	names := dropConfiguredModelNames(m, at, previousAutoModelNames(at.Parent))
	m.Delete(at.Parent, "auto_model_names")
	m.Delete(at.Parent, "auto_model_name")
	deleteEmpty(m.Root(), "global", "router")
	if len(names) == 0 {
		return nil
	}
	served := chatEntrypointNames(m.Root())
	sequence := resolveAlias(mappingValue(m.Root(), "entrypoints"))
	for i, entrypoint := range sequenceItems(sequence) {
		entrypoint = resolveAlias(entrypoint)
		if api := scalarAt(entrypoint, "api"); scalarAt(entrypoint, "recipe") != string(DefaultRecipeName) || (api != "" && api != string(ChatAPI)) {
			continue
		}
		written := stringItems(mappingValue(entrypoint, "model_names"))
		if slices.Equal(written, []string{DefaultEntrypointModel}) {
			if kept := slices.DeleteFunc(names, func(name string) bool { return served[name] && name != DefaultEntrypointModel }); len(kept) > 0 {
				m.Set(entrypoint, "model_names", stringSequence(kept...))
			}
			return nil
		}
		names = slices.DeleteFunc(names, func(name string) bool { return served[name] })
		if len(names) > 0 {
			beside := mappingNode("model_names", stringSequence(names...), "recipe", string(DefaultRecipeName))
			sequence.Content = slices.Insert(sequence.Content, i+1, beside)
			m.Changed("entrypoints", fmt.Sprintf("the auto model names %s are an entrypoint of the default recipe beside the document's own, %s", strings.Join(names, ", "), strings.Join(written, ", ")))
		}
		return nil
	}
	if names = slices.DeleteFunc(names, func(name string) bool { return served[name] }); len(names) == 0 {
		return nil
	}
	if len(sequenceAt(m.Root(), "recipes")) == 0 {
		m.Set(m.Root(), "recipes", sequenceOf(mappingNode("name", string(DefaultRecipeName))))
		m.Changed("recipes["+string(DefaultRecipeName)+"]", "the document has no routing; an empty default recipe serves the auto model names, with the default model as before")
	}
	entrypoint := mappingNode("model_names", stringSequence(names...), "recipe", string(DefaultRecipeName))
	marked := slices.ContainsFunc(sequenceAt(m.Root(), "entrypoints"), func(e *yaml.Node) bool { return scalarAt(resolveAlias(e), "default") == "true" })
	if len(sequenceAt(m.Root(), "recipes")) > 1 && !marked {
		setNode(entrypoint, "default", boolNode(true))
	}
	sequence = ensureSequence(m.Root(), "entrypoints")
	sequence.Content = append([]*yaml.Node{entrypoint}, sequence.Content...)
	return nil
}

// dropConfiguredModelNames drops from the auto model names migrate wrote for
// a document that names none, which keepDefaultAutoModelNames notes at their
// path, each name a configured model takes: an entrypoint name must be new,
// and a request for the name reaches that model. A provider field it cannot
// read leaves the name to the load of the migrated document, which refuses it.
func dropConfiguredModelNames(m *Migration, at FormMatch, names []string) []string {
	if !slices.ContainsFunc(m.notes, func(note ConfigWarning) bool { return note.Field == at.Path }) {
		return names
	}
	var configured CanonicalConfig
	if providers := resolveAlias(mappingValue(m.Root(), "providers")); providers != nil {
		_ = providers.Decode(&configured.Providers)
	}
	return slices.DeleteFunc(names, func(name string) bool {
		meaning := configuredEntrypointNameConflict(&configured, name)
		if meaning != "" {
			m.Changed(at.Path, fmt.Sprintf("the auto model name %s is dropped: it is already %s, which a request for it reaches", name, meaning))
		}
		return meaning != ""
	})
}
