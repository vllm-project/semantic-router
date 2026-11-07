package config

import (
	"bytes"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// migrationActionCode marks a note of what an operator must do for the
// migrated file to work as the original did.
const migrationActionCode = "migration_action"

// MigrateModelResources rewrites the deprecated model-resource fields of a
// v0.3 document in the forms that replace them, as the Router reads them,
// and keeps every other field as written. The notes name the fields that have
// no replacement and stay (deprecated_model_field, which the Router still
// reads), and what an operator must do (migration_action), such as setting
// the environment variable a key written in the configuration moved to.
func MigrateModelResources(data []byte) ([]byte, []ConfigWarning, error) {
	cfg, err := ParseYAMLBytesWithoutEnvExpansion(data)
	if err != nil {
		return nil, nil, err
	}
	var fields []string
	for _, warning := range cfg.ModelResourceWarnings {
		if warning.Code == warningDeprecatedModelField {
			fields = append(fields, warning.Field)
		}
	}
	if len(fields) == 0 {
		return data, nil, nil
	}
	var document yaml.Node
	if err = yaml.Unmarshal(data, &document); err != nil {
		return nil, nil, err
	}
	if len(document.Content) == 0 || document.Content[0].Kind != yaml.MappingNode {
		return data, nil, nil
	}
	exported := CanonicalConfigFromRouterConfig(cfg)
	m := &modelResourceMigration{root: document.Content[0], cfg: cfg, exported: &exported, fields: fields}
	if err = m.apply(); err != nil {
		return nil, nil, err
	}
	var out bytes.Buffer
	encoder := yaml.NewEncoder(&out)
	encoder.SetIndent(2)
	if err = encoder.Encode(&document); err != nil {
		return nil, nil, err
	}
	if err = encoder.Close(); err != nil {
		return nil, nil, err
	}
	migrated, err := ParseYAMLBytesWithoutEnvExpansion(out.Bytes())
	if err != nil {
		return nil, nil, fmt.Errorf("the migrated configuration does not load: %w", err)
	}
	notes := m.actions
	for _, warning := range migrated.ModelResourceWarnings {
		if warning.Code == warningDeprecatedModelField {
			notes = append(notes, warning)
		}
	}
	return out.Bytes(), notes, nil
}

// answersJailbreak reports whether a binding of the document asks the
// deployment the jailbreak signal.
func (m *modelResourceMigration) answersJailbreak(name string) bool {
	key := BindingKey("prompt_guard")
	if m.exported.Global != nil && m.exported.Global.ModelCatalog.Bindings[key].Deployment == name {
		return true
	}
	if m.exported.Routing.ModelBindings[key].Deployment == name {
		return true
	}
	for _, recipe := range m.exported.Recipes {
		if recipe.Routing.ModelBindings[key].Deployment == name {
			return true
		}
	}
	return false
}

// unexpressedChatSettings are the settings of a deprecated external[] model
// that no deployment field expresses; the chat jailbreak guard reads them.
func unexpressedChatSettings(legacy *ExternalModelConfig) []string {
	var settings []string
	if legacy.ParserType != "" {
		settings = append(settings, "parser_type")
	}
	if legacy.Threshold > 0 {
		settings = append(settings, "threshold")
	}
	if legacy.ModelEndpoint.UseChatTemplate || legacy.ModelEndpoint.PromptTemplate != "" {
		settings = append(settings, "a prompt template")
	}
	return settings
}

// legacyGeneration is the generation block of a deprecated external[] chat
// model's max_tokens, temperature and reasoning.
func legacyGeneration(legacy *ExternalModelConfig) *DeploymentGeneration {
	if legacy.MaxTokens <= 0 && legacy.Temperature == 0 && legacy.Reasoning == nil {
		return nil
	}
	generation := &DeploymentGeneration{MaxTokens: legacy.MaxTokens, Reasoning: legacy.Reasoning}
	if legacy.Temperature != 0 {
		temperature := legacy.Temperature
		generation.Temperature = &temperature
	}
	return generation
}

// inlineKeyEnvironment names the environment variable a deployment's key,
// written in the configuration, moves to.
func inlineKeyEnvironment(deployment string) string {
	name := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' {
			return r - 'a' + 'A'
		}
		if (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') {
			return r
		}
		return '_'
	}, deployment)
	return "VLLM_SR_" + name + "_API_KEY"
}

type modelResourceMigration struct {
	root     *yaml.Node
	cfg      *RouterConfig
	exported *CanonicalConfig
	fields   []string
	actions  []ConfigWarning
}

// moduleMappingField is the module field that names a signal's label
// mapping file.
var moduleMappingField = map[string][]string{
	"domain_classifier": {"classifier", "domain", "category_mapping_path"},
	"pii_classifier":    {"classifier", "pii", "pii_mapping_path"},
	"prompt_guard":      {"prompt_guard", "jailbreak_mapping_path"},
	"feedback_detector": {"feedback_detector", "feedback_mapping_path"},
}

// migratedBindings writes each binding without the contract and adapter its
// signal and deployment imply, its label mapping in its module; a binding that
// asks for something else keeps those fields.
func (m *modelResourceMigration) migratedBindings(bindings map[string]ModelBinding) map[string]ModelBinding {
	if bindings == nil {
		return nil
	}
	modules := mappingAt(m.root, "global", "model_catalog", "modules")
	out := make(map[string]ModelBinding, len(bindings))
	for key, binding := range bindings {
		consumer := BindingConsumer(key)
		if deployment, ok := m.cfg.ModelDeployments[binding.Deployment]; ok {
			deployment = deployment.WithDefaults()
			implied := ImpliedBinding(consumer, ModelBinding{Deployment: binding.Deployment, Head: binding.Head, OperatingPoint: binding.OperatingPoint}, deployment)
			if binding.Contract == implied.Contract {
				binding.Contract = ""
			}
			// A model runtime serves a model with the adapter its card names.
			if deployment.IsModelRuntime() || binding.Adapter == implied.Adapter || binding.Adapter == "auto" {
				binding.Adapter = ""
			}
		}
		if field, ok := moduleMappingField[consumer]; ok && binding.MappingPath != "" && modules != nil {
			module := mappingAt(modules, field[:len(field)-1]...)
			current := scalarAt(module, field[len(field)-1])
			if module != nil && (current == "" || current == binding.MappingPath) {
				_ = setValue(module, field[len(field)-1], binding.MappingPath, false)
				binding.MappingPath = ""
			}
		}
		out[key] = binding
	}
	return out
}

// warned reports whether a deprecation warning names path or a field under it.
func (m *modelResourceMigration) warned(path string) bool {
	for _, field := range m.fields {
		if field == path || strings.HasPrefix(field, path+".") || strings.HasPrefix(field, path+"[") {
			return true
		}
	}
	return false
}

func (m *modelResourceMigration) apply() error {
	catalog := mappingAt(m.root, "global", "model_catalog")
	var exportedCatalog CanonicalModelCatalog
	if m.exported.Global != nil {
		exportedCatalog = m.exported.Global.ModelCatalog
	}
	if catalog != nil || len(exportedCatalog.Deployments) > 0 || len(exportedCatalog.Bindings) > 0 {
		if catalog == nil {
			catalog = ensureMapping(m.root, "global", "model_catalog")
		}
		if err := m.migrateCatalog(catalog, exportedCatalog); err != nil {
			return err
		}
	}
	if m.exported.Global != nil && m.exported.Global.Stores.Memory.QueryRewrite != nil && mappingAt(m.root, "global", "stores", "memory", "query_rewrite") == nil {
		if err := setValue(ensureMapping(m.root, "global", "stores", "memory"), "query_rewrite", m.exported.Global.Stores.Memory.QueryRewrite, false); err != nil {
			return err
		}
	}
	m.migrateBackendRefs()
	if err := m.migrateRouting("routing", mappingAt(m.root, "routing"), &m.exported.Routing); err != nil {
		return err
	}
	for i, recipe := range sequenceAt(m.root, "recipes") {
		name := scalarAt(recipe, "name")
		for j := range m.exported.Recipes {
			if m.exported.Recipes[j].Name == name {
				if err := m.migrateRouting(fmt.Sprintf("recipes[%s].routing", name), mappingAt(recipe, "routing"), &m.exported.Recipes[j].Routing); err != nil {
					return fmt.Errorf("recipes[%d]: %w", i, err)
				}
			}
		}
	}
	return nil
}

func (m *modelResourceMigration) migrateCatalog(catalog *yaml.Node, exported CanonicalModelCatalog) error {
	// Deployments: the authored ones in their current form, and every
	// deployment a deprecated field declared. The pre-declared ones are
	// declared already.
	authored := map[string]bool{}
	for _, name := range mappingKeys(mappingAt(catalog, "deployments")) {
		authored[name] = true
	}
	predeclared := DefaultModelDeployments()
	deployments := map[string]ModelDeployment{}
	for name, deployment := range exported.Deployments {
		if _, declared := predeclared[name]; declared && !authored[name] {
			continue
		}
		if legacy := deployment.LegacyExternal(); legacy != nil {
			if settings := unexpressedChatSettings(legacy); len(settings) > 0 && m.answersJailbreak(name) {
				return fmt.Errorf("deployments.%s: the external[] model %q sets %s, which no deployment field expresses; keep the earlier form", name, legacy.Name, strings.Join(settings, ", "))
			}
			if deployment.Provider == OpenAIChatProvider && deployment.Generation == nil {
				deployment.Generation = legacyGeneration(legacy)
			}
		}
		if deployment.Auth.HasInlineKey() {
			// A key is never written in the configuration: it moves to an
			// environment variable the operator sets.
			environment := inlineKeyEnvironment(name)
			deployment.Auth = &DeploymentAuth{Header: deployment.Auth.Header, Prefix: deployment.Auth.Prefix, APIKeyEnv: environment}
			m.actions = append(m.actions, ConfigWarning{
				Code: migrationActionCode, Field: "global.model_catalog.deployments." + name + ".auth.api_key_env",
				Message: fmt.Sprintf("the key the configuration wrote for %s (access_key) is not written in the migrated file; set %s to it before the Router starts", name, environment),
			})
		}
		if deployment.Input.MaxTokens == 0 && (deployment.Input.Overflow == "" || deployment.Input.Overflow == "reject") {
			// The default budget: the model's own limit, refusing more.
			deployment.Input = ModelInputBudget{}
		}
		deployments[name] = deployment
	}
	if err := setValue(catalog, "deployments", deployments, len(deployments) == 0); err != nil {
		return err
	}
	bindings := m.migratedBindings(exported.Bindings)
	if err := setValue(catalog, "bindings", bindings, len(bindings) == 0); err != nil {
		return err
	}
	if system := mappingAt(catalog, "system"); system != nil {
		for _, key := range mappingKeys(system) {
			if key != "decision_model" {
				deleteKey(system, key)
			}
		}
		if len(system.Content) == 0 {
			deleteKey(catalog, "system")
		}
	}
	if embeddings := mappingAt(catalog, "embeddings"); embeddings != nil || exported.Embeddings.Default != "" {
		if err := m.migrateEmbeddings(catalog, exported); err != nil {
			return err
		}
	}
	m.migrateModules(catalog)
	if external := mappingValue(catalog, "external"); external != nil && external.Kind == yaml.SequenceNode {
		kept := external.Content[:0]
		for _, entry := range external.Content {
			if !m.warned(fmt.Sprintf("global.model_catalog.external[%q]", scalarAt(entry, "name"))) {
				kept = append(kept, entry)
			}
		}
		external.Content = kept
		if len(kept) == 0 {
			deleteKey(catalog, "external")
		}
	}
	if m.warned("global.model_catalog.admission") {
		if err := setValue(catalog, "admission", exported.Admission, len(exported.Admission) == 0); err != nil {
			return err
		}
	}
	return nil
}

func (m *modelResourceMigration) migrateEmbeddings(catalog *yaml.Node, exported CanonicalModelCatalog) error {
	embeddings := ensureMapping(catalog, "embeddings")
	for key, name := range map[string]string{"default": exported.Embeddings.Default, "multimodal": exported.Embeddings.Multimodal} {
		predeclared := map[string]string{"default": EmbeddingDefaultDeployment, "multimodal": MultimodalEmbeddingDeployment}[key]
		if name != "" && name != predeclared && mappingValue(embeddings, key) == nil {
			if err := setValue(embeddings, key, name, false); err != nil {
				return err
			}
		}
	}
	semantic := mappingAt(embeddings, "semantic")
	if semantic != nil && m.warned("global.model_catalog.embeddings.semantic.endpoint") {
		deleteKey(semantic, "endpoint")
		if config := mappingAt(semantic, "embedding_config"); config != nil {
			deleteKey(config, "backend")
			deleteKey(config, "model_type")
		}
		deleteEmpty(catalog, "embeddings", "semantic", "embedding_config")
	}
	return nil
}

// migrateModules drops the module fields a deprecation warning names: a model
// the module ran, now a deployment and binding, and a backend. A module that
// ran no model is switched off.
func (m *modelResourceMigration) migrateModules(catalog *yaml.Node) {
	modules := mappingAt(catalog, "modules")
	if modules == nil {
		return
	}
	for _, field := range m.fields {
		rest, ok := strings.CutPrefix(field, "global.model_catalog.modules.")
		if !ok {
			continue
		}
		path := strings.Split(rest, ".")
		key := path[len(path)-1]
		module := mappingAt(modules, path[:len(path)-1]...)
		if module == nil {
			continue
		}
		noModel := key == "model_ref" && scalarAt(module, "model_ref") == "" && mappingValue(module, "model_ref") != nil && scalarAt(module, "model_id") == ""
		switch key {
		case "model_id", "model_ref", "backend", "model_path":
			deleteKey(module, key)
		}
		if key == "model_ref" {
			deleteKey(module, "model_id")
		}
		if noModel && moduleHasSwitch(path[:len(path)-1]) {
			_ = setValue(module, "enabled", false, false)
		}
		deleteEmpty(catalog, append([]string{"modules"}, path[:len(path)-1]...)...)
	}
}

// moduleHasSwitch reports whether a module path names a module that an
// enabled field turns off.
func moduleHasSwitch(path []string) bool {
	switch strings.Join(path, ".") {
	case "classifier.domain", "classifier.pii", "prompt_guard", "feedback_detector", "hallucination_mitigation.fact_check":
		return true
	}
	return false
}

// migrateBackendRefs writes each request backend that uses the deprecated
// connection fields in the connection block it is read as.
func (m *modelResourceMigration) migrateBackendRefs() {
	for _, model := range sequenceAt(m.root, "providers", "models") {
		name := scalarAt(model, "name")
		refs := mappingValue(model, "backend_refs")
		if refs == nil || refs.Kind != yaml.SequenceNode {
			continue
		}
		for _, exported := range m.exported.Providers.Models {
			if exported.Name != name || len(exported.BackendRefs) != len(refs.Content) {
				continue
			}
			for j := range refs.Content {
				if !m.warned(fmt.Sprintf("providers.models[%s].backend_refs[%d]", name, j)) {
					continue
				}
				var node yaml.Node
				if err := node.Encode(exported.BackendRefs[j]); err == nil {
					refs.Content[j] = &node
				}
			}
		}
	}
}

var deprecatedBindingKeyPath = regexp.MustCompile(`^(routing|recipes\[[^]]+\]\.routing)\.model_bindings\.`)

func (m *modelResourceMigration) migrateRouting(path string, routing *yaml.Node, exported *CanonicalRouting) error {
	if routing == nil {
		return nil
	}
	bindings := m.migratedBindings(exported.ModelBindings)
	// The rag plugins of the profile name the reranker its rag.reranker
	// binding chose.
	reranker, hasReranker := bindings[BindingKey(RAGRerankerConsumer)]
	delete(bindings, BindingKey(RAGRerankerConsumer))
	if mappingValue(routing, "model_bindings") != nil || len(bindings) > 0 {
		if err := setValue(routing, "model_bindings", bindings, len(bindings) == 0); err != nil {
			return err
		}
	}
	signals := mappingAt(routing, "signals")
	for _, rule := range sequenceAt(signals, "classifiers") {
		name := scalarAt(rule, "name")
		if !m.warned(fmt.Sprintf("%s.signals.classifiers[%q]", path, name)) && !m.bindingMoved(path, "classifier."+name) {
			continue
		}
		for _, key := range []string{"type", "model", "model_path", "use_cpu"} {
			deleteKey(rule, key)
		}
		for _, exportedRule := range exported.Signals.Classifiers {
			if exportedRule.Name == name && exportedRule.Deployment != "" {
				_ = setValue(rule, "deployment", exportedRule.Deployment, false)
			}
		}
	}
	for _, rule := range sequenceAt(signals, "safety") {
		name := scalarAt(rule, "name")
		for _, exportedRule := range exported.Signals.Safety {
			if exportedRule.Name != name {
				continue
			}
			if m.warned(fmt.Sprintf("%s.signals.safety[%q].model", path, name)) || m.bindingMoved(path, "safety."+name) {
				deleteKey(rule, "model")
				if exportedRule.Deployment != "" {
					_ = setValue(rule, "deployment", exportedRule.Deployment, false)
				}
			}
			hazard := mappingAt(rule, "hazard")
			if hazard != nil && exportedRule.Hazard != nil && (m.warned(fmt.Sprintf("%s.signals.safety[%q].hazard.model", path, name)) || m.bindingMoved(path, "safety."+name+".hazard")) {
				deleteKey(hazard, "model")
				if exportedRule.Hazard.Deployment != "" {
					_ = setValue(hazard, "deployment", exportedRule.Hazard.Deployment, false)
				}
			}
		}
	}
	for _, rule := range sequenceAt(signals, "complexity") {
		name := scalarAt(rule, "name")
		for _, exportedRule := range exported.Signals.Complexity {
			if exportedRule.Name == name && exportedRule.Deployment != "" && scalarAt(rule, "deployment") == "" {
				_ = setValue(rule, "deployment", exportedRule.Deployment, false)
			}
		}
	}
	var rerank *RAGRerankConfig
	if hasReranker {
		rerank = &RAGRerankConfig{Deployment: reranker.Deployment}
		if reranker.PairScorer != nil {
			rerank.Layer, rerank.Dimension = reranker.PairScorer.Layer, reranker.PairScorer.Dimension
		}
	}
	m.migrateRerank(routing, exported, rerank)
	return nil
}

// bindingMoved reports whether a deprecated binding for one rule was moved
// onto the rule.
func (m *modelResourceMigration) bindingMoved(path, consumer string) bool {
	for _, field := range m.fields {
		if deprecatedBindingKeyPath.MatchString(field) && strings.HasPrefix(field, path+".model_bindings.") && strings.TrimPrefix(field, path+".model_bindings.") == consumer {
			return true
		}
	}
	return false
}

// migrateRerank names the reranker in each rag plugin of a profile whose
// deprecated rag.reranker binding chose it: the plugin's rerank block, or the
// one the export read.
func (m *modelResourceMigration) migrateRerank(routing *yaml.Node, exported *CanonicalRouting, binding *RAGRerankConfig) {
	for _, decision := range sequenceAt(routing, "decisions") {
		chosen := binding
		if chosen == nil {
			for i := range exported.Decisions {
				if exported.Decisions[i].Name == scalarAt(decision, "name") {
					if rag := exported.Decisions[i].GetRAGConfig(); rag != nil && rag.Rerank != nil && rag.Rerank.Deployment != "" {
						chosen = rag.Rerank
					}
				}
			}
		}
		if chosen == nil {
			continue
		}
		for _, plugin := range sequenceAt(decision, "plugins") {
			configuration := mappingAt(plugin, "configuration")
			if scalarAt(plugin, "type") != "rag" || configuration == nil || mappingValue(configuration, "rerank") == nil {
				continue
			}
			rerank := mappingAt(configuration, "rerank")
			if rerank == nil || scalarAt(rerank, "deployment") != "" {
				continue
			}
			_ = setValue(rerank, "deployment", chosen.Deployment, false)
			if chosen.Layer != 0 {
				_ = setValue(rerank, "layer", chosen.Layer, false)
			}
			if chosen.Dimension != 0 {
				_ = setValue(rerank, "dimension", chosen.Dimension, false)
			}
		}
	}
}

// YAML node helpers. A mapping node holds its keys and values alternately.

func mappingValue(node *yaml.Node, key string) *yaml.Node {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value == key {
			return node.Content[i+1]
		}
	}
	return nil
}

func mappingAt(node *yaml.Node, path ...string) *yaml.Node {
	for _, key := range path {
		node = mappingValue(node, key)
	}
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	return node
}

func sequenceAt(node *yaml.Node, path ...string) []*yaml.Node {
	for _, key := range path {
		node = mappingValue(node, key)
	}
	if node == nil || node.Kind != yaml.SequenceNode {
		return nil
	}
	return node.Content
}

func scalarAt(node *yaml.Node, key string) string {
	if value := mappingValue(node, key); value != nil && value.Kind == yaml.ScalarNode {
		return value.Value
	}
	return ""
}

func mappingKeys(node *yaml.Node) []string {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	keys := make([]string, 0, len(node.Content)/2)
	for i := 0; i+1 < len(node.Content); i += 2 {
		keys = append(keys, node.Content[i].Value)
	}
	sort.Strings(keys)
	return keys
}

func ensureMapping(node *yaml.Node, path ...string) *yaml.Node {
	for _, key := range path {
		next := mappingValue(node, key)
		if next == nil || next.Kind != yaml.MappingNode {
			next = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			setNode(node, key, next)
		}
		node = next
	}
	return node
}

func deleteKey(node *yaml.Node, key string) {
	if node == nil || node.Kind != yaml.MappingNode {
		return
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value == key {
			node.Content = append(node.Content[:i], node.Content[i+2:]...)
			return
		}
	}
}

// deleteEmpty removes the mapping at path when the migration left it empty,
// and each parent that leaves empty in turn.
func deleteEmpty(node *yaml.Node, path ...string) {
	for ; len(path) > 0; path = path[:len(path)-1] {
		parent := mappingAt(node, path[:len(path)-1]...)
		child := mappingValue(parent, path[len(path)-1])
		if parent == nil || child == nil || child.Kind != yaml.MappingNode || len(child.Content) > 0 {
			return
		}
		deleteKey(parent, path[len(path)-1])
	}
}

func setNode(node *yaml.Node, key string, value *yaml.Node) {
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value == key {
			node.Content[i+1] = value
			return
		}
	}
	node.Content = append(node.Content, &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: key}, value)
}

// setValue writes value at key, or removes the key when remove is set.
func setValue(node *yaml.Node, key string, value interface{}, remove bool) error {
	if remove {
		deleteKey(node, key)
		return nil
	}
	var encoded yaml.Node
	if err := encoded.Encode(value); err != nil {
		return err
	}
	setNode(node, key, &encoded)
	return nil
}
