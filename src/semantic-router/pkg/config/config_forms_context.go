package config

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// The fields of the removed plugins that became match blocks, and where.
var (
	ragRecallFields    = []string{"top_k", "similarity_threshold", "rerank"}
	memoryRecallFields = []string{"retrieval_limit", "similarity_threshold"}
	recallFieldTargets = map[string]string{
		"top_k":                "match.recall.top_k",
		"retrieval_limit":      "match.recall.top_k",
		"similarity_threshold": "match.recall.min_similarity",
		"rerank":               "match.judge with the relevance preset on the decision model",
	}
	// contextKeyOrder is the order of a context plugin's configuration keys.
	contextKeyOrder = []string{"enabled", "sources", "compress", "budget", "injection"}
)

// contextForms are the forms of the request context before one context
// plugin assembled it. The removed plugins' recall and scoring fields are
// rewritten as match blocks, and RAG's min_confidence_threshold, a hidden
// fact-check condition, as a condition of the decision's rules, while the
// plugins still show where they were; then rag, memory and
// context_compression merge into the decision's context plugin, rag sources
// before memory sources, then compress. The reranker binding and the rerank
// provider go last: the decision model judges relevance.
var contextForms = []ConfigForm{
	{
		Code:    CodeInvalidValue,
		Paths:   slices.Concat(RoutingScopes("decisions[].plugins[]"), RoutingScopes("plugins[]"), []string{"global.plugins[]"}),
		Match:   func(node *yaml.Node, _ string) bool { return mcpRAGPlugin(node) || len(mcpRAGSources(node)) > 0 },
		What:    "an enabled RAG source on backend mcp is removed: the Router has no MCP tool invoker",
		Becomes: "nothing: migrate drops the source, or a hybrid's mcp fallback; serve the retrieval over HTTP and read it with backend: external_api",
		Migrate: migrateMCPRAG,
	},
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasPluginFields("rag", "min_confidence_threshold"),
		What:    "rag min_confidence_threshold is removed: it retrieved only when a fact_check rule matched",
		Becomes: "a fact_check condition in the decision's rules",
		Migrate: migrateRAGConfidenceThreshold,
	},
	{
		Code:     CodeRemovedField,
		Paths:    RoutingScopes("decisions[].plugins[]"),
		Match:    hasPluginFields("rag", ragRecallFields...),
		Describe: describeRecallFields("rag", ragRecallFields),
		Migrate:  migrateRAGRecallFields,
	},
	{
		Code:     CodeRemovedField,
		Paths:    RoutingScopes("decisions[].plugins[]"),
		Match:    hasPluginFields("memory", memoryRecallFields...),
		Describe: describeRecallFields("memory", memoryRecallFields),
		Migrate:  migrateMemoryRecallFields,
	},
	{
		Code:    CodeRemovedField,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasPluginFields(compressionPluginType, "scoring"),
		What:    "context_compression scoring is removed: segments are judged, not embedded",
		Becomes: "compress.match: the relevance judge by default, or recall: {method: bm25} for lexical order",
		Migrate: migrateCompressionScoring,
	},
	{
		Code:    CodeRemovedPluginType,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasType("rag"),
		What:    "plugin rag is removed",
		Becomes: RemovedPluginTypes["rag"],
		Migrate: migrateRAGPlugin,
	},
	{
		Code:    CodeRemovedPluginType,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasType("memory"),
		What:    "plugin memory is removed",
		Becomes: RemovedPluginTypes["memory"],
		Migrate: migrateMemoryPlugin,
	},
	{
		Code:    CodeRemovedPluginType,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasType(compressionPluginType),
		What:    "plugin context_compression is removed",
		Becomes: RemovedPluginTypes[compressionPluginType],
		Migrate: migrateCompressionPlugin,
	},
	{
		Code:    CodeRemovedField,
		Paths:   append([]string{"global.model_catalog.bindings.*"}, RoutingScopes("model_bindings.*")...),
		Match:   func(_ *yaml.Node, path string) bool { return strings.HasSuffix(path, ".rag.reranker") },
		What:    "the rag.reranker binding is removed: the decision model judges RAG relevance",
		Becomes: "the rag source's match.judge",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			return nil
		},
	},
	{
		Code:    CodeInvalidValue,
		Paths:   []string{"global.model_catalog.deployments.*"},
		Match:   func(node *yaml.Node, _ string) bool { return scalarAt(node, "provider") == "rerank" },
		What:    "provider rerank is removed: the decision model judges RAG relevance",
		Becomes: "the rag source's match.judge; the deployment is deleted",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, at.Key)
			return nil
		},
	},
}

// compressionPluginType is the removed context_compression plugin's type.
const compressionPluginType = "context_compression"

// hasPluginFields matches a plugin of a type that sets any of fields.
func hasPluginFields(pluginType string, fields ...string) func(*yaml.Node, string) bool {
	return func(node *yaml.Node, path string) bool {
		return hasType(pluginType)(node, path) && len(setFields(mappingValue(node, "configuration"), fields)) > 0
	}
}

// setFields are the fields a mapping sets, in the order given.
func setFields(mapping *yaml.Node, fields []string) []string {
	var set []string
	for _, field := range fields {
		if mappingValue(mapping, field) != nil {
			set = append(set, field)
		}
	}
	return set
}

func describeRecallFields(pluginType string, fields []string) func(FormMatch) (string, string) {
	return func(at FormMatch) (string, string) {
		set := setFields(mappingValue(at.Node, "configuration"), fields)
		targets := make([]string, len(set))
		for index, field := range set {
			targets[index] = recallFieldTargets[field]
		}
		if len(set) == 1 {
			return pluginType + " " + set[0] + " is removed", targets[0]
		}
		return pluginType + " " + strings.Join(set, ", ") + " are removed", strings.Join(targets, ", ")
	}
}

// migrateRAGConfidenceThreshold puts the condition the threshold hid in the
// decision's rules. Every matched fact_check rule set the confidence to 1, so
// a threshold in (0, 1] retrieved when any of the scope's fact_check rules
// matched; at most 0 it always did, and above 1 never.
func migrateRAGConfidenceThreshold(m *Migration, at FormMatch) error {
	configuration := mappingValue(at.Node, "configuration")
	threshold, _ := strconv.ParseFloat(scalarAt(configuration, "min_confidence_threshold"), 64)
	m.Delete(configuration, "min_confidence_threshold")
	path := at.Path + ".configuration.min_confidence_threshold"
	if threshold <= 0 || !yamlTrue(mappingValue(configuration, "enabled")) {
		return nil
	}
	decision, scope := scopeDecision(at.Root, at.Parent)
	if decision == nil {
		return fmt.Errorf("the plugin belongs to no decision")
	}
	if threshold > 1 {
		m.Note(path, "the threshold exceeded every fact-check confidence, so this retrieval never ran; remove the rag source to keep it so")
		return nil
	}
	names := scopeRuleNames(at.Root, scope, SignalTypeFactCheck)
	if len(names) == 0 {
		m.Note(path, "the scope has no fact_check rule, so this retrieval never ran and now runs on every request the decision matches; add a fact_check rule and a condition on it to the decision's rules to keep it conditional")
		return nil
	}
	conditions := make([]*yaml.Node, len(names))
	for index, name := range names {
		conditions[index] = flowMapping("type", SignalTypeFactCheck, "name", name)
	}
	condition := conditions[0]
	if len(conditions) > 1 {
		condition = ruleNode("OR", conditions...)
	}
	andIntoRules(m, decision, condition)
	m.Note(scope+".decisions["+scalarAt(decision, "name")+"].rules", fmt.Sprintf(
		"the decision now matches only when fact_check %s matches, as its retrieval ran; the requests it matched otherwise reach the next decision, which may need its own route",
		strings.Join(names, " or ")))
	return nil
}

// migrateRAGRecallFields writes a rag plugin's retrieval bounds and reranker
// as its match block: recall by the bounds it set, else the RAG defaults,
// then the relevance judge, keeping the reranker's top_k.
func migrateRAGRecallFields(m *Migration, at FormMatch) error {
	configuration := mappingValue(at.Node, "configuration")
	defaults := MatchRAG.Default()
	rerank := mappingValue(configuration, "rerank")
	recall := mappingNode("method", Scalar(MatchRecallEmbedding),
		"top_k", valueOr(mappingValue(configuration, "top_k"), intNode(defaults.Recall.TopK)),
		"min_similarity", valueOr(mappingValue(configuration, "similarity_threshold"), floatNode(*defaults.Recall.MinSimilarity)))
	judge := mappingNode("preset", Scalar(defaults.Judge.Preset),
		"top_k", valueOr(mappingValue(rerank, "top_k"), intNode(defaults.Judge.TopK)),
		"min_score", floatNode(*defaults.Judge.MinScore))
	path := at.Path + ".configuration"
	if threshold := scalarAt(configuration, "similarity_threshold"); threshold != "" {
		m.Changed(path+".similarity_threshold", "match.recall.min_similarity "+threshold+" bounds the cosine in the store's embedding space; the judge's min_score is a relevance probability the decision model answers")
	}
	if deployment := scalarAt(rerank, "deployment"); deployment != "" {
		m.Changed(path+".rerank", "the reranker deployment "+deployment+" no longer scores the documents: the decision model judges their relevance")
	}
	replaceKeys(configuration, ragRecallFields, "match", mappingNode("recall", recall, "judge", judge))
	return nil
}

// migrateMemoryRecallFields writes a memory plugin's retrieval bounds as its
// match block. An unset bound was the memory store's default, and a
// threshold of at most 0 the built-in default.
func migrateMemoryRecallFields(m *Migration, at FormMatch) error {
	configuration := mappingValue(at.Node, "configuration")
	store := mappingAt(at.Root, "global", "stores", "memory")
	limit := positive(mappingValue(configuration, "retrieval_limit"), mappingValue(store, "default_retrieval_limit"), intNode(MatchMemory.Default().Recall.TopK))
	threshold := mappingValue(configuration, "similarity_threshold")
	if threshold == nil {
		threshold = positive(mappingValue(store, "default_similarity_threshold"), floatNode(float64(DefaultMemorySimilarityThreshold)))
	} else {
		threshold = positive(threshold, floatNode(float64(DefaultMemorySimilarityThreshold)))
	}
	replaceKeys(configuration, memoryRecallFields, "match", mappingNode("recall", mappingNode("method", Scalar(MatchRecallEmbedding), "top_k", limit, "min_similarity", threshold)))
	return nil
}

// migrateCompressionScoring writes context compression's scoring as its
// match block: lexical scoring as BM25 recall over the segments, embedding
// and hybrid scoring as the default relevance judge.
func migrateCompressionScoring(m *Migration, at FormMatch) error {
	configuration := mappingValue(at.Node, "configuration")
	method := scalarAt(mappingValue(configuration, "scoring"), "method")
	if method == "" || method == MatchRecallBM25 {
		replaceKeys(configuration, []string{"scoring"}, "match", mappingNode("recall", mappingNode("method", Scalar(MatchRecallBM25))))
		m.Changed(at.Path+".configuration.match", "segments keep their lexical order; add judge: {preset: relevance} to have the decision model judge them")
		return nil
	}
	m.Delete(configuration, "scoring")
	m.Changed(at.Path+".configuration", method+" scoring embedded the segments; the decision model now judges them with the relevance preset")
	return nil
}

// migrateRAGPlugin makes an enabled rag plugin the rag source of the
// decision's context plugin; a disabled one did nothing.
func migrateRAGPlugin(m *Migration, at FormMatch) error {
	old := mappingValue(at.Node, "configuration")
	if !yamlTrue(mappingValue(old, "enabled")) {
		removeItem(at.Parent, at.Node)
		m.Changed(at.Path, "the disabled rag plugin retrieved nothing and is dropped")
		return nil
	}
	source := sourceNode(ContextSourceRAG, old, "enabled", "injection_mode")
	configuration, err := contextConfiguration(m, at)
	if err != nil {
		return err
	}
	if err := addSource(configuration, source); err != nil {
		return err
	}
	if mode := scalarAt(old, "injection_mode"); mode != "" && mode != ContextInjectionToolRole {
		m.Set(configuration, "injection", mappingNode("mode", Scalar(mode)))
	}
	orderKeys(configuration, contextKeyOrder)
	return nil
}

// ragSettings are the settings of a rag plugin, under its configuration, or of
// a context plugin's rag source.
func ragSettings(node *yaml.Node) *yaml.Node {
	if configuration := mappingValue(node, "configuration"); configuration != nil {
		return resolveAlias(configuration)
	}
	return node
}

// ragReadsMCP reports whether a rag plugin or source retrieves through the mcp
// backend, itself or as a hybrid's primary or fallback.
func ragReadsMCP(node *yaml.Node) bool {
	settings := ragSettings(node)
	switch scalarAt(settings, "backend") {
	case "mcp":
		return true
	case "hybrid":
		hybrid := mappingAt(settings, "backend_config")
		return scalarAt(hybrid, "primary") == "mcp" || scalarAt(hybrid, "fallback") == "mcp"
	}
	return false
}

// mcpRAGPlugin reports whether a plugin is an enabled rag plugin that
// retrieves through mcp. A disabled one retrieves nothing and loads as written.
func mcpRAGPlugin(plugin *yaml.Node) bool {
	return hasType("rag")(plugin, "") && enabledScalar(resolveAlias(mappingValue(plugin, "configuration"))) && ragReadsMCP(plugin)
}

// mcpRAGSources are the rag sources of an enabled context plugin that
// retrieve through mcp.
func mcpRAGSources(plugin *yaml.Node) []*yaml.Node {
	configuration := resolveAlias(mappingValue(plugin, "configuration"))
	if !hasType(DecisionPluginContext)(plugin, "") || !enabledScalar(configuration) {
		return nil
	}
	var sources []*yaml.Node
	for _, source := range sequenceItems(resolveAlias(mappingValue(configuration, "sources"))) {
		if source = resolveAlias(source); hasType(ContextSourceRAG)(source, "") && ragReadsMCP(source) {
			sources = append(sources, source)
		}
	}
	return sources
}

// migrateMCPRAG drops what retrieves through mcp, which has no invoker: an
// enabled rag plugin on it, or a context plugin's source on it, or the mcp
// fallback of a hybrid source, which then retrieves from its primary alone.
func migrateMCPRAG(m *Migration, at FormMatch) error {
	if mcpRAGPlugin(at.Node) {
		removeItem(at.Parent, at.Node)
		if holder := at.Ancestors[len(at.Ancestors)-1]; len(at.Parent.Content) == 0 {
			for i := 0; i+1 < len(holder.Content); i += 2 {
				if resolveAlias(holder.Content[i+1]) == at.Parent {
					m.Delete(holder, holder.Content[i].Value)
					break
				}
			}
		}
		noteMCPRAGDropped(m, at.Path, ragSettings(at.Node))
		return nil
	}
	configuration := resolveAlias(mappingValue(at.Node, "configuration"))
	sources := resolveAlias(mappingValue(configuration, "sources"))
	for _, source := range mcpRAGSources(at.Node) {
		path := fmt.Sprintf("%s.configuration.sources[%s]", at.Path, itemLabel(source, slices.Index(sources.Content, source)))
		config := mappingAt(source, "backend_config")
		if scalarAt(source, "backend") == "hybrid" && scalarAt(config, "primary") != "mcp" {
			m.Delete(config, "fallback")
			m.Delete(config, "fallback_config")
			m.Note(path, "the hybrid source's mcp fallback is dropped: the Router has no MCP tool invoker; the source retrieves from its primary "+scalarAt(config, "primary")+" alone")
			continue
		}
		removeItem(sources, source)
		noteMCPRAGDropped(m, path, source)
	}
	if len(sources.Content) == 0 {
		m.Delete(configuration, "sources")
		keepGlobalMemoryOff(m, at, configuration)
	}
	return nil
}

// noteMCPRAGDropped notes the RAG source on mcp that migrate dropped.
func noteMCPRAGDropped(m *Migration, path string, settings *yaml.Node) {
	config := mappingAt(settings, "backend_config")
	if scalarAt(settings, "backend") == "hybrid" {
		config = mappingAt(config, "primary_config")
	}
	m.Note(path, fmt.Sprintf("the RAG source on the mcp backend is dropped: the Router has no MCP tool invoker, so these requests retrieve nothing from server %s (tool %s); "+
		"to keep that retrieval, serve it over HTTP and add a source {type: rag, backend: external_api, backend_config: {endpoint: <its URL>, request_format: custom, "+
		"request_template: <its JSON body, with {{.Query}} and {{.TopK}}>}}", scalarAt(config, "server_name"), scalarAt(config, "tool_name")))
}

// migrateMemoryPlugin makes an enabled memory plugin the memory source of
// the decision's context plugin. A disabled one turned memory off for the
// decision: its context plugin lists no memory source, or, without one, the
// decision gets a disabled context plugin.
func migrateMemoryPlugin(m *Migration, at FormMatch) error {
	old := mappingValue(at.Node, "configuration")
	if !yamlTrue(mappingValue(old, "enabled")) {
		if existing := pluginOfType(at.Parent, DecisionPluginContext); existing != nil {
			removeItem(at.Parent, at.Node)
			keepGlobalMemoryOff(m, at, mappingAt(existing, "configuration"))
			return nil
		}
		m.Set(at.Node, "type", Scalar(DecisionPluginContext))
		m.Set(at.Node, "configuration", mappingNode("enabled", boolNode(false)))
		return nil
	}
	source := sourceNode(ContextSourceMemory, old, "enabled", "auto_store")
	if auto := mappingValue(old, "auto_store"); auto != nil {
		insertAfter(source, "match", "store", mappingNode("auto", auto))
	}
	configuration, err := contextConfiguration(m, at)
	if err != nil {
		return err
	}
	if err := addSource(configuration, source); err != nil {
		return err
	}
	orderKeys(configuration, contextKeyOrder)
	m.Changed(at.Path, "memories are injected as the context plugin's injection.mode says (a tool result by default), no longer as a user message")
	return nil
}

// migrateCompressionPlugin makes an enabled context_compression plugin the
// compress block and budget of the decision's context plugin. Compression
// now starts when the request exceeds the budget, so a fixed trigger is
// dropped.
func migrateCompressionPlugin(m *Migration, at FormMatch) error {
	old := mappingValue(at.Node, "configuration")
	if !yamlTrue(mappingValue(old, "enabled")) {
		removeItem(at.Parent, at.Node)
		m.Changed(at.Path, "the disabled context_compression plugin compressed nothing and is dropped")
		return nil
	}
	compress := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	for i := 0; i+1 < len(old.Content); i += 2 {
		if key := old.Content[i].Value; key != "enabled" && key != "budget" {
			compress.Content = append(compress.Content, old.Content[i], old.Content[i+1])
		}
	}
	budget := compressionBudget(m, at.Path+".configuration.budget", mappingValue(old, "budget"))
	retrieved := decisionContext(at).Source(ContextSourceMemory) != nil
	configuration, err := contextConfiguration(m, at)
	if err != nil {
		return err
	}
	if mappingValue(configuration, "compress") != nil {
		return fmt.Errorf("the decision's context plugin already compresses; keep one")
	}
	m.Set(configuration, "enabled", boolNode(true))
	m.Set(configuration, "compress", compress)
	if budget != nil {
		m.Set(configuration, "budget", budget)
	}
	orderKeys(configuration, contextKeyOrder)
	if !retrieved {
		keepGlobalMemoryOff(m, at, configuration)
	}
	return nil
}

// compressionBudget writes the old budget as the context budget: its target
// is max_tokens, which now also starts compression in place of the trigger.
func compressionBudget(m *Migration, path string, old *yaml.Node) *yaml.Node {
	if old == nil {
		return nil
	}
	budget := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	if target := mappingValue(old, "target_tokens"); target != nil {
		m.Set(budget, "max_tokens", target)
	}
	if reserve := mappingValue(old, "reserve_output_tokens"); reserve != nil {
		m.Set(budget, "reserve_output_tokens", reserve)
	}
	trigger, target := scalarAt(old, "trigger_tokens"), scalarAt(old, "target_tokens")
	switch {
	case trigger != "" && !strings.EqualFold(trigger, "auto"):
		m.Changed(path+".trigger_tokens", "compression started above "+trigger+" tokens; it now starts when the request exceeds budget.max_tokens, or, for auto, nears the model's window or has a segment over its target's min_tokens")
	case target != "" && !strings.EqualFold(target, "auto"):
		m.Changed(path+".target_tokens", "budget.max_tokens "+target+" also starts compression: a request over it is compressed down to it")
	}
	if len(budget.Content) == 0 {
		return nil
	}
	return budget
}

// contextConfiguration returns the configuration of the decision's context
// plugin, which the plugin at at merges into: the decision's own, or a new
// one in place of the plugin. A new one also reads the global memory a
// decision without a memory plugin retrieved.
func contextConfiguration(m *Migration, at FormMatch) (*yaml.Node, error) {
	if existing := pluginOfType(at.Parent, DecisionPluginContext); existing != nil {
		removeItem(at.Parent, at.Node)
		configuration := mappingValue(existing, "configuration")
		if configuration == nil || configuration.Kind != yaml.MappingNode {
			return nil, fmt.Errorf("the decision's context plugin has no configuration")
		}
		return configuration, nil
	}
	configuration := mappingNode("enabled", boolNode(true))
	if decisionContext(at).Source(ContextSourceMemory) != nil && pluginOfType(at.Parent, "memory") == nil {
		m.Set(configuration, "sources", &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq", Content: []*yaml.Node{mappingNode("type", Scalar(ContextSourceMemory))}})
		m.Changed(at.Path, "the global memory now reaches this decision through its context plugin's memory source")
	}
	m.Set(at.Node, "type", Scalar(DecisionPluginContext))
	m.Set(at.Node, "configuration", configuration)
	return configuration, nil
}

// decisionContext is the enabled context plugin the decision at runs, nil
// without one: its own context plugin layered over its recipe's and
// global.plugins', as the loader layers them, where the memory store's
// switch is the context plugin with a memory source it becomes.
func decisionContext(at FormMatch) *ContextPluginConfig {
	global := mappingAt(at.Root, "global")
	var memorySwitch *yaml.Node
	if yamlTrue(mappingValue(mappingAt(global, "stores", "memory"), "enabled")) {
		memorySwitch = mappingNode("type", DecisionPluginContext, "configuration", mappingNode(
			"enabled", boolNode(true),
			"sources", sequenceOf(mappingNode("type", ContextSourceMemory)),
		))
	}
	var plugins []DecisionPlugin
	for _, layer := range []*yaml.Node{
		memorySwitch,
		pluginOfType(mappingValue(global, "plugins"), DecisionPluginContext),
		pluginOfType(mappingValue(at.Above(4), "plugins"), DecisionPluginContext),
		pluginOfType(at.Parent, DecisionPluginContext),
	} {
		var plugin DecisionPlugin
		if layer != nil && layer.Decode(&plugin) == nil {
			plugins, _ = layerPlugins(plugins, []DecisionPlugin{plugin})
		}
	}
	return ContextPluginOf(plugins)
}

// keepGlobalMemoryOff writes sources: [] in the configuration of a decision's
// context plugin that lists none, for a decision that did not retrieve the
// memory it would inherit: written sources replace the inherited ones, and
// its injection, which places what sources retrieve, is dropped. A plugin
// that would then neither retrieve nor compress writes enabled: false.
func keepGlobalMemoryOff(m *Migration, at FormMatch, configuration *yaml.Node) {
	context := decisionContext(at)
	if configuration == nil || mappingValue(configuration, "sources") != nil || context.Source(ContextSourceMemory) == nil {
		return
	}
	if context.Compress == nil {
		m.Set(configuration, "enabled", boolNode(false))
		m.Changed(at.Path, "the decision did not retrieve memory and its context plugin lists no sources and compresses nothing, so it writes enabled: false; one that lists no sources inherits the memory source of its recipe or global.plugins")
		return
	}
	m.Set(configuration, "sources", sequenceOf())
	m.Delete(configuration, "injection")
	orderKeys(configuration, contextKeyOrder)
	m.Changed(at.Path, "the decision did not retrieve memory, so its context plugin writes sources: []; one that lists no sources inherits the memory source of its recipe or global.plugins")
}

// sourceNode is a source of a type with an old plugin's fields but those
// dropped.
func sourceNode(sourceType string, old *yaml.Node, dropped ...string) *yaml.Node {
	source := mappingNode("type", Scalar(sourceType))
	for i := 0; old != nil && i+1 < len(old.Content); i += 2 {
		if !slices.Contains(dropped, old.Content[i].Value) {
			source.Content = append(source.Content, old.Content[i], old.Content[i+1])
		}
	}
	return source
}

// addSource adds a source to a context configuration, rag sources before
// memory sources.
func addSource(configuration, source *yaml.Node) error {
	sources := mappingValue(configuration, "sources")
	if sources == nil {
		sources = &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq"}
		setNode(configuration, "sources", sources)
	}
	sourceType := scalarAt(source, "type")
	at := len(sources.Content)
	for index, existing := range sources.Content {
		switch scalarAt(existing, "type") {
		case sourceType:
			return fmt.Errorf("the decision's context plugin already has a %s source; keep one", sourceType)
		case ContextSourceMemory:
			if sourceType == ContextSourceRAG && at == len(sources.Content) {
				at = index
			}
		}
	}
	sources.Content = slices.Insert(sources.Content, at, source)
	return nil
}

// scopeDecision returns the decision whose plugins are plugins, and the
// document path of its recipe.
func scopeDecision(root, plugins *yaml.Node) (*yaml.Node, string) {
	for _, pattern := range RoutingScopes("decisions[]") {
		for _, decision := range matchPattern(root, strings.Split(pattern, ".")) {
			if mappingValue(decision.Node, "plugins") == plugins {
				return decision.Node, decision.Path[:strings.LastIndex(decision.Path, ".decisions[")]
			}
		}
	}
	return nil, ""
}

// scopeRuleNames are the names of a recipe's rules of a signal type.
func scopeRuleNames(root *yaml.Node, scope, signalType string) []string {
	var names []string
	for _, pattern := range RoutingScopes("signals." + signalType + "[]") {
		for _, rule := range matchPattern(root, strings.Split(pattern, ".")) {
			if strings.HasPrefix(rule.Path, scope+".signals.") {
				names = append(names, scalarAt(rule.Node, "name"))
			}
		}
	}
	return names
}

// andIntoRules adds a condition the decision's rules must also meet.
func andIntoRules(m *Migration, decision, condition *yaml.Node) {
	rules := mappingValue(decision, "rules")
	switch {
	case rules == nil || rules.Kind != yaml.MappingNode || len(rules.Content) == 0:
		m.Set(decision, "rules", condition)
	case strings.EqualFold(scalarAt(rules, "operator"), "AND") && mappingValue(rules, "conditions") != nil:
		conditions := mappingValue(rules, "conditions")
		conditions.Content = append(conditions.Content, condition)
	default:
		m.Set(decision, "rules", ruleNode("AND", rules, condition))
	}
}

func ruleNode(operator string, conditions ...*yaml.Node) *yaml.Node {
	return mappingNode("operator", Scalar(operator), "conditions", &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq", Content: conditions})
}

func pluginOfType(plugins *yaml.Node, pluginType string) *yaml.Node {
	if plugins == nil || plugins.Kind != yaml.SequenceNode {
		return nil
	}
	for _, item := range plugins.Content {
		if scalarAt(item, "type") == pluginType {
			return item
		}
	}
	return nil
}

// orderKeys puts a mapping's keys in order, the keys order does not name
// after them as they were.
func orderKeys(mapping *yaml.Node, order []string) {
	var sorted, rest []*yaml.Node
	for _, key := range order {
		if value := mappingValue(mapping, key); value != nil {
			sorted = append(sorted, Scalar(key), value)
		}
	}
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if !slices.Contains(order, mapping.Content[i].Value) {
			rest = append(rest, mapping.Content[i], mapping.Content[i+1])
		}
	}
	sorted = append(sorted, rest...)
	mapping.Content = sorted
}

// replaceKeys writes key in place of the first of fields a mapping sets, and
// deletes the others.
func replaceKeys(mapping *yaml.Node, fields []string, key string, value *yaml.Node) {
	at := -1
	for i := 0; i+1 < len(mapping.Content); {
		if !slices.Contains(fields, mapping.Content[i].Value) {
			i += 2
			continue
		}
		if at < 0 {
			at = i
		}
		mapping.Content = slices.Delete(mapping.Content, i, i+2)
	}
	if at < 0 {
		at = len(mapping.Content)
	}
	mapping.Content = slices.Insert(mapping.Content, at, Scalar(key), value)
}

// insertAfter writes key after the key after, or first without it.
func insertAfter(mapping *yaml.Node, after, key string, value *yaml.Node) {
	at := 2
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == after {
			at = i + 2
		}
	}
	mapping.Content = slices.Insert(mapping.Content, at, Scalar(key), value)
}

// mappingNode builds a mapping of alternating keys and values, each value a
// string or a node.
func mappingNode(pairs ...any) *yaml.Node {
	mapping := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	for i := 0; i+1 < len(pairs); i += 2 {
		value, ok := pairs[i+1].(*yaml.Node)
		if !ok {
			value = Scalar(fmt.Sprint(pairs[i+1]))
		}
		mapping.Content = append(mapping.Content, Scalar(pairs[i].(string)), value)
	}
	return mapping
}

func flowMapping(pairs ...any) *yaml.Node {
	mapping := mappingNode(pairs...)
	mapping.Style = yaml.FlowStyle
	return mapping
}

func intNode(value int) *yaml.Node {
	return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!int", Value: strconv.Itoa(value)}
}

func floatNode(value float64) *yaml.Node {
	return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!float", Value: strconv.FormatFloat(value, 'f', -1, 32)}
}

func boolNode(value bool) *yaml.Node {
	return &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!bool", Value: strconv.FormatBool(value)}
}

// valueOr is the node written, else the default.
func valueOr(written, fallback *yaml.Node) *yaml.Node {
	if written != nil {
		return written
	}
	return fallback
}

// positive is the first node written that is not a number at most 0 (an
// unset bound), else the last.
func positive(candidates ...*yaml.Node) *yaml.Node {
	for _, candidate := range candidates {
		if candidate == nil {
			continue
		}
		if value, err := strconv.ParseFloat(candidate.Value, 64); err != nil || value > 0 {
			return candidate
		}
	}
	return candidates[len(candidates)-1]
}
