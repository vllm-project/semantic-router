package config

import (
	"fmt"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// previousConfigVersion is the schema version v0.5 replaces.
const previousConfigVersion = "v0.3"

// structureForms are the v0.3 layout of a configuration, which migrate
// rewrites before any other form so that every later form matches v0.5 paths
// only: the schema version, the model cards, the top-level routing block and
// each recipe's routing block. The Router refuses a v0.3 document by its
// version alone.
var structureForms = []ConfigForm{
	{
		Code:    CodeInvalidValue,
		Paths:   []string{"version"},
		Match:   func(node *yaml.Node, _ string) bool { return node.Value == previousConfigVersion },
		What:    "config schema " + previousConfigVersion + " is removed",
		Becomes: "version: " + CanonicalConfigVersion,
		Whole:   true,
		Migrate: func(m *Migration, at FormMatch) error {
			m.Set(at.Parent, "version", Scalar(CanonicalConfigVersion))
			noteDecisionlessCache(m, at.Root)
			return nil
		},
	},
	{
		Code:    CodeRemovedSection,
		Paths:   []string{"routing.modelCards"},
		What:    "routing.modelCards is removed",
		Becomes: "each card's fields on the providers.models[] entry of the same name",
		Migrate: migrateModelCards,
	},
	{
		Code:     CodeRemovedSection,
		Paths:    []string{"routing"},
		Describe: describeTopLevelRouting,
		Migrate:  migrateTopLevelRouting,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"recipes[].routing"},
		What:    "recipes[].routing is removed",
		Becomes: "its fields on the recipe: recipes[].signals, recipes[].decisions, ...",
		Migrate: migrateRecipeRouting,
	},
}

// migrateModelCards merges each card into the providers.models[] entry whose
// catalog identity it is (its catalog, else its name; one of the same name
// first). A card no entry names becomes a metadata-only entry.
func migrateModelCards(m *Migration, at FormMatch) error {
	models := ensureSequence(ensureMapping(at.Root, "providers"), "models")
	loras := map[string]bool{}
	for _, card := range at.Node.Content {
		for _, lora := range sequenceAt(resolveAlias(card), "loras") {
			loras[scalarAt(lora, "name")] = true
		}
	}
	for _, card := range at.Node.Content {
		card = resolveAlias(card)
		name := scalarAt(card, "name")
		if name == "" {
			return fmt.Errorf("a model card has no name")
		}
		entry := providerModelForCard(models, name)
		if entry == nil && loras[name] {
			// A card of a LoRA adapter's name described the adapter, which is
			// now its loras[] entry: a name and a description.
			if len(card.Content) > 2 {
				m.Note("routing.modelCards["+name+"]", "the card of LoRA adapter "+name+" is dropped; its loras[] entry holds its name and description")
			}
			continue
		}
		if entry == nil {
			entry = mappingNode("name", name)
			models.Content = append(models.Content, entry)
			m.Note("providers.models["+name+"]", "model card "+name+" named no providers.models[] entry; it is a metadata-only entry, so add its backend_refs or drop it")
		}
		for i := 0; i+1 < len(card.Content); i += 2 {
			field := card.Content[i].Value
			if field == "name" {
				continue
			}
			if !slices.Contains(ModelFactFields(), field) {
				return fmt.Errorf("model card %s: unknown field %s", name, field)
			}
			if mappingValue(entry, field) != nil {
				m.Note("providers.models["+scalarAt(entry, "name")+"]."+field, "model card "+name+" also set "+field+"; the entry's own value is kept")
				continue
			}
			entry.Content = append(entry.Content, card.Content[i], card.Content[i+1])
		}
	}
	m.Delete(at.Parent, "modelCards")
	return nil
}

// providerModelForCard is the providers.models[] entry a card describes: the
// entry of the card's name, else the first whose catalog is the card.
func providerModelForCard(models *yaml.Node, card string) *yaml.Node {
	var byCatalog *yaml.Node
	for _, item := range models.Content {
		item = resolveAlias(item)
		switch {
		case scalarAt(item, "name") == card && (scalarAt(item, "catalog") == "" || scalarAt(item, "catalog") == card):
			return item
		case byCatalog == nil && scalarAt(item, "catalog") == card:
			byCatalog = item
		}
	}
	return byCatalog
}

func describeTopLevelRouting(at FormMatch) (string, string) {
	becomes := "recipes: [{name: " + string(DefaultRecipeName) + ", ...}]"
	if len(sequenceAt(at.Root, "recipes")) > 0 {
		becomes += ", with an entrypoint marked default: true"
	}
	return "the top-level routing block is removed", becomes
}

// migrateTopLevelRouting writes the top-level routing block as the recipe
// named default, the first of the recipes.
func migrateTopLevelRouting(m *Migration, at FormMatch) error {
	if recipeNamed(at.Root, string(DefaultRecipeName)) != nil {
		return fmt.Errorf("a recipe is already named %s; rename it first", DefaultRecipeName)
	}
	recipe := mappingNode("name", string(DefaultRecipeName))
	if routing := resolveAlias(at.Node); routing.Kind == yaml.MappingNode {
		recipe.Content = append(recipe.Content, routing.Content...)
	}
	if recipes := resolveAlias(mappingValue(at.Root, "recipes")); recipes != nil && recipes.Kind == yaml.SequenceNode {
		recipes.Content = append([]*yaml.Node{recipe}, recipes.Content...)
		m.Delete(at.Root, "routing")
	} else {
		index := slices.IndexFunc(at.Root.Content, func(key *yaml.Node) bool { return key.Value == "routing" })
		at.Root.Content[index].Value = "recipes"
		at.Root.Content[index+1] = sequenceOf(recipe)
	}
	return markDefaultEntrypoint(m, at.Root)
}

// migrateRecipeRouting moves the fields of a recipe's routing block onto the
// recipe.
func migrateRecipeRouting(m *Migration, at FormMatch) error {
	recipe, routing := at.Parent, resolveAlias(at.Node)
	index := slices.IndexFunc(recipe.Content, func(key *yaml.Node) bool { return key.Value == "routing" })
	var fields []*yaml.Node
	for i := 0; routing.Kind == yaml.MappingNode && i+1 < len(routing.Content); i += 2 {
		if mappingValue(recipe, routing.Content[i].Value) != nil {
			return fmt.Errorf("the recipe also sets %s; keep one", routing.Content[i].Value)
		}
		fields = append(fields, routing.Content[i], routing.Content[i+1])
	}
	recipe.Content = slices.Replace(recipe.Content, index, index+2, fields...)
	return markDefaultEntrypoint(m, at.Root)
}

// markDefaultEntrypoint keeps which recipe serves the default routing once
// the layout has no top-level routing block. A v0.3 default profile was the
// recipe named default, or else an empty profile, which becomes an empty
// recipe named default when other recipes would take its place. With several
// recipes, the first chat entrypoint of the default recipe is marked
// default: true, or one serving vllm-sr/auto is added.
func markDefaultEntrypoint(m *Migration, root *yaml.Node) error {
	recipes := resolveAlias(mappingValue(root, "recipes"))
	entrypoints := sequenceAt(root, "entrypoints")
	for _, entrypoint := range entrypoints {
		if scalarAt(entrypoint, "default") == "true" {
			return nil
		}
	}
	if recipeNamed(root, string(DefaultRecipeName)) == nil {
		recipes.Content = append([]*yaml.Node{mappingNode("name", string(DefaultRecipeName))}, recipes.Content...)
		m.Note("recipes["+string(DefaultRecipeName)+"]", "the default routing profile was empty; it stays an empty recipe, so "+
			DefaultEntrypointModel+" requests go to the default model as before")
	}
	if len(recipes.Content) < 2 {
		return nil
	}
	for _, entrypoint := range entrypoints {
		api := scalarAt(entrypoint, "api")
		if scalarAt(entrypoint, "recipe") == string(DefaultRecipeName) && (api == "" || api == string(ChatAPI)) {
			setNode(entrypoint, "default", boolNode(true))
			return nil
		}
	}
	// The built-in default entrypoint led the routing table.
	entrypoint := mappingNode("model_names", stringSequence(DefaultEntrypointModel), "recipe", string(DefaultRecipeName))
	setNode(entrypoint, "default", boolNode(true))
	sequence := ensureSequence(root, "entrypoints")
	sequence.Content = append([]*yaml.Node{entrypoint}, sequence.Content...)
	return nil
}

func recipeNamed(root *yaml.Node, name string) *yaml.Node {
	for _, recipe := range sequenceAt(root, "recipes") {
		if scalarAt(recipe, "name") == name {
			return recipe
		}
	}
	return nil
}

// switchForms are the feature switches global.stores and global.services held,
// now global.plugins entries. Migrate rewrites them last, after the stores'
// own forms renamed them and the plugin forms gave each plugin its canonical
// type.
var switchForms = []ConfigForm{
	{
		Code:     CodeRemovedField,
		Paths:    []string{"global.stores.response_cache"},
		Match:    func(node *yaml.Node, _ string) bool { return len(responseCacheSwitches(node)) > 0 },
		Describe: describeResponseCacheSwitches,
		Migrate:  migrateResponseCacheSwitches,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.stores.memory.enabled"},
		What:    "global.stores.memory.enabled is removed",
		Becomes: "a context plugin with a memory source in global.plugins",
		Migrate: migrateMemorySwitch,
	},
	{
		Code:     CodeRemovedField,
		Paths:    []string{"global.services.router_replay"},
		Match:    func(node *yaml.Node, _ string) bool { return len(routerReplayPluginFields(node)) > 0 },
		Describe: describeRouterReplaySwitches,
		Migrate:  migrateRouterReplaySwitches,
	},
}

func responseCacheSwitches(store *yaml.Node) []string {
	var fields []string
	for _, field := range []string{"enabled", "similarity_threshold"} {
		if mappingValue(store, field) != nil {
			fields = append(fields, field)
		}
	}
	return fields
}

func describeResponseCacheSwitches(at FormMatch) (string, string) {
	return removedFields("global.stores.response_cache", responseCacheSwitches(at.Node)),
		"a response_cache plugin turns caching on: in global.plugins for every request, in a decision's plugins for its own"
}

// removedFields words the removal of fields of the mapping at path:
// "path.a is removed", "path.a and b are removed", "path.a, b and c are removed".
// Migrate describes a match again once rewritten, when the fields are gone.
func removedFields(path string, fields []string) string {
	if len(fields) == 0 {
		return path + " feature switches are removed"
	}
	listed := fields[0]
	if len(fields) > 1 {
		listed = strings.Join(fields[:len(fields)-1], ", ") + " and " + fields[len(fields)-1]
	}
	return path + "." + listed + pluralIs(len(fields)) + " removed"
}

// migrateResponseCacheSwitches keeps what the store's switch did. Enabled, it
// cached every request of a configuration without decisions, bounded by the
// store's threshold, and otherwise only the decisions whose plugin enabled
// it; disabled, it turned those plugins off.
func migrateResponseCacheSwitches(m *Migration, at FormMatch) error {
	enabled := scalarAt(at.Node, "enabled") != "false"
	threshold := mappingValue(at.Node, "similarity_threshold")
	m.Delete(at.Node, "enabled")
	m.Delete(at.Node, "similarity_threshold")
	dropEmpty(m, at.Parent, "response_cache")
	dropEmpty(m, at.Above(2), "stores")
	plugins := decisionPluginsOfType(at.Root, DecisionPluginResponseCache)
	switch {
	case !enabled:
		for _, plugin := range plugins {
			if configuration := mappingValue(plugin, "configuration"); scalarAt(configuration, "enabled") == "true" {
				setNode(configuration, "enabled", boolNode(false))
				m.Note(at.Path, "the store was disabled, so a decision's response_cache plugin is disabled as well; enable it to cache its requests")
			}
		}
	case decisionCount(at.Root) > 0:
		if threshold != nil {
			m.Note(at.Path+".similarity_threshold", "it bounded requests that match no decision, which this configuration does not cache; it is dropped")
		}
	default:
		configuration := mappingNode("enabled", boolNode(true), "scope", "global")
		if threshold != nil {
			setNode(configuration, "semantic", mappingNode("match", mappingNode("recall", mappingNode("top_k", intNode(1), "min_similarity", threshold))))
		}
		return addGlobalPlugin(at.Root, DecisionPluginResponseCache, configuration)
	}
	return nil
}

func migrateMemorySwitch(m *Migration, at FormMatch) error {
	enabled := at.Node.Value == "true"
	m.Delete(at.Parent, "enabled")
	dropEmpty(m, at.Above(2), "memory")
	dropEmpty(m, at.Above(3), "stores")
	if !enabled {
		return nil
	}
	return addGlobalPlugin(at.Root, DecisionPluginContext, mappingNode(
		"enabled", boolNode(true),
		"sources", sequenceOf(mappingNode("type", ContextSourceMemory)),
	))
}

// routerReplayPluginFields are the fields of the replay service that its
// plugin now holds: the switch and the capture defaults.
func routerReplayPluginFields(service *yaml.Node) []string {
	var fields []string
	for _, field := range []string{
		"enabled", "capture_request_body", "capture_response_body", "capture_personal_data", "capture_tool_results",
		"max_records", "max_body_bytes", "max_tool_trace_bytes", "max_tool_trace_steps",
	} {
		if mappingValue(service, field) != nil {
			fields = append(fields, field)
		}
	}
	return fields
}

func describeRouterReplaySwitches(at FormMatch) (string, string) {
	return removedFields("global.services.router_replay", routerReplayPluginFields(at.Node)),
		"the configuration of a router_replay plugin in global.plugins"
}

func migrateRouterReplaySwitches(m *Migration, at FormMatch) error {
	configuration := mappingNode()
	for _, field := range routerReplayPluginFields(at.Node) {
		moveField(at.Node, configuration, field)
	}
	dropEmpty(m, at.Parent, "router_replay")
	dropEmpty(m, at.Above(2), "services")
	if len(configuration.Content) == 2 && scalarAt(configuration, "enabled") == "false" {
		return nil
	}
	return addGlobalPlugin(at.Root, DecisionPluginRouterReplay, configuration)
}

// moveField moves key, its comments with it, from one mapping to the end of
// another.
func moveField(from, to *yaml.Node, key string) {
	for i := 0; i+1 < len(from.Content); i += 2 {
		if from.Content[i].Value == key {
			to.Content = append(to.Content, from.Content[i], from.Content[i+1])
			from.Content = slices.Delete(from.Content, i, i+2)
			return
		}
	}
}

// dropEmpty deletes the key of holder whose mapping a removed switch left
// empty.
func dropEmpty(m *Migration, holder *yaml.Node, key string) {
	if value := mappingValue(holder, key); value != nil && value.Kind == yaml.MappingNode && len(value.Content) == 0 {
		m.Delete(holder, key)
	}
}

// addGlobalPlugin adds a plugin to global.plugins, which a document written
// before them cannot already hold one of.
func addGlobalPlugin(root *yaml.Node, pluginType string, configuration *yaml.Node) error {
	plugins := ensureSequence(ensureMapping(root, "global"), "plugins")
	for _, plugin := range plugins.Content {
		if scalarAt(resolveAlias(plugin), "type") == pluginType {
			return fmt.Errorf("global.plugins already has a %s plugin; keep one", pluginType)
		}
	}
	plugins.Content = append(plugins.Content, mappingNode("type", pluginType, "configuration", configuration))
	return nil
}

// decisionPluginsOfType are the plugins of a type in every decision of every
// recipe.
func decisionPluginsOfType(root *yaml.Node, pluginType string) []*yaml.Node {
	var plugins []*yaml.Node
	for _, recipe := range sequenceAt(root, "recipes") {
		for _, decision := range sequenceAt(recipe, "decisions") {
			for _, plugin := range sequenceAt(decision, "plugins") {
				if scalarAt(plugin, "type") == pluginType {
					plugins = append(plugins, plugin)
				}
			}
		}
	}
	return plugins
}

func decisionCount(root *yaml.Node) int {
	count := 0
	for _, recipe := range sequenceAt(root, "recipes") {
		count += len(sequenceAt(recipe, "decisions"))
	}
	return count
}

func pluralIs(count int) string {
	if count == 1 {
		return " is"
	}
	return " are"
}

// noteDecisionlessCache tells a v0.3 configuration without decisions, whose
// response cache store was on by default and cached every response, how to
// keep caching them.
func noteDecisionlessCache(m *Migration, root *yaml.Node) {
	decisions := len(sequenceAt(mappingValue(root, "routing"), "decisions"))
	for _, recipe := range sequenceAt(root, "recipes") {
		decisions += len(sequenceAt(mappingValue(recipe, "routing"), "decisions"))
	}
	if decisions > 0 || mappingValue(mappingAt(root, "global", "stores", "response_cache"), "enabled") != nil {
		return
	}
	m.Note("global.plugins", "without decisions, the response cache store cached every response while it was on, as it was by default; "+
		"add a response_cache plugin with scope: global to global.plugins to keep caching them")
}
