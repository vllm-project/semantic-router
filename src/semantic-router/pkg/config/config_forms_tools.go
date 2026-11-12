package config

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// The tools plugin fields that moved under its policy, and those of the
// removed embedding retrieval, which select replaces.
var (
	toolsPolicyFields    = []string{"mode", "allow_tools", "block_tools", "strip_tool_history", "trusted_facts"}
	toolsRetrievalFields = []string{"semantic_selection", "strategy", "dynamic_retrieval"}
)

// The catalog and top_k the removed retrieval read when nothing named them.
const (
	legacyToolsDBPath = "config/tools_db.json"
	legacyToolsTopK   = 3
)

// toolsForms are the forms of tool handling before the decision model chose
// the tools: the global tools block, whose catalog and top_k each decision
// that retrieved from it now names in its selection; the tool_selection
// plugin, now the tools plugin's select block; and the tools plugin's
// top-level fields, now its policy. The global block is rewritten first,
// while each decision still shows whether it retrieved from it.
var toolsForms = []ConfigForm{
	{
		Code:    CodeRemovedSection,
		Paths:   []string{"global.integrations.tools"},
		What:    "global.integrations.tools is removed",
		Becomes: "tools.select.tools_db_path and match.judge.top_k on the decisions that retrieved tools from it",
		Migrate: migrateGlobalTools,
	},
	{
		Code:    CodeRemovedPluginType,
		Paths:   RoutingScopes("decisions[].plugins[]"),
		Match:   hasType("tool_selection"),
		What:    "plugin tool_selection is removed",
		Becomes: RemovedPluginTypes["tool_selection"],
		Migrate: migrateToolSelectionPlugin,
	},
	{
		Code:     CodeRemovedField,
		Paths:    RoutingScopes("decisions[].plugins[]"),
		Match:    hasFlatToolsFields,
		Describe: describeFlatToolsFields,
		Migrate:  migrateFlatToolsFields,
	},
}

// migrateGlobalTools names the global catalog and top_k where decisions read
// them: an enabled tool_selection plugin in add mode that names neither, and
// an enabled tools plugin whose semantic selection retrieved from the
// enabled global catalog, which now selects from it in add mode.
func migrateGlobalTools(m *Migration, at FormMatch) error {
	global := at.Node
	catalog := scalarAt(global, "tools_db_path")
	if catalog == "" {
		catalog = legacyToolsDBPath
	}
	topK, _ := strconv.Atoi(scalarAt(global, "top_k"))
	if topK <= 0 {
		topK = legacyToolsTopK
	}
	fallback := mappingValue(global, "fallback_to_empty")
	for _, pattern := range RoutingScopes("decisions[]") {
		for _, decision := range matchPattern(at.Root, strings.Split(pattern, ".")) {
			plugins := mappingValue(decision.Node, "plugins")
			if selection := enabledPlugin(plugins, "tool_selection"); selection != nil {
				configuration := mappingValue(selection, "configuration")
				if toolSelectionMode(configuration) == ToolsSelectModeAdd {
					if scalarAt(configuration, "tools_db_path") == "" {
						m.Set(configuration, "tools_db_path", Scalar(catalog))
					}
					if own, _ := strconv.Atoi(scalarAt(configuration, "top_k")); own <= 0 {
						if err := m.SetValue(configuration, "top_k", topK); err != nil {
							return err
						}
					}
				}
				if fallback != nil && mappingValue(configuration, "fallback_to_empty") == nil {
					m.Set(configuration, "fallback_to_empty", fallback)
				}
				continue
			}
			tools := enabledPlugin(plugins, DecisionPluginTools)
			if tools == nil || !yamlTrue(mappingValue(global, "enabled")) || !retrievedTools(mappingValue(tools, "configuration")) {
				continue
			}
			spec := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			m.Set(spec, "mode", Scalar(ToolsSelectModeAdd))
			m.Set(spec, "tools_db_path", Scalar(catalog))
			if err := setToolRankJudge(m, spec, ToolsSelectModeAdd, topK); err != nil {
				return err
			}
			m.Set(mappingValue(tools, "configuration"), "select", spec)
		}
	}
	if mappingValue(global, "similarity_threshold") != nil {
		m.Note(at.Path+".similarity_threshold", "an embedding similarity cut has no judge equivalent; fit match.judge.min_score or min_ratio to the decision model if the selections need one")
	}
	if mappingValue(global, "advanced_filtering") != nil {
		m.Note(at.Path+".advanced_filtering", "lexical, tag, category and history scoring are removed; to use another catalog per domain, split the decisions on a domain condition")
	}
	m.Delete(at.Parent, "tools")
	deleteEmpty(at.Root, "global", "integrations")
	return nil
}

// retrievedTools reports whether a tools plugin written before select
// retrieved from the global catalog: semantic selection, on by default,
// with a mode that keeps tools.
func retrievedTools(configuration *yaml.Node) bool {
	if mappingValue(configuration, "policy") != nil || mappingValue(configuration, "select") != nil {
		return false
	}
	if scalarAt(configuration, "mode") == ToolsPolicyModeNone {
		return false
	}
	semantic := mappingValue(configuration, "semantic_selection")
	return semantic == nil || yamlTrue(semantic)
}

// migrateToolSelectionPlugin moves an enabled tool_selection plugin into the
// select block of the decision's tools plugin, whose policy it ran under, or
// makes it the decision's tools plugin. A disabled one did nothing.
func migrateToolSelectionPlugin(m *Migration, at FormMatch) error {
	plugins := at.Parent
	selection := mappingValue(at.Node, "configuration")
	if !yamlTrue(mappingValue(selection, "enabled")) {
		removeItem(plugins, at.Node)
		return nil
	}
	spec, err := toolsSelectFromToolSelection(m, at.Path+".configuration", selection)
	if err != nil {
		return err
	}
	if tools := enabledPlugin(plugins, DecisionPluginTools); tools != nil {
		configuration := mappingValue(tools, "configuration")
		if mappingValue(configuration, "select") != nil {
			return fmt.Errorf("the decision's tools plugin already has a select block")
		}
		m.Set(configuration, "select", spec)
		removeItem(plugins, at.Node)
		return nil
	}
	// A disabled tools plugin applied no policy to the selection.
	for _, item := range slices.Clone(plugins.Content) {
		if scalarAt(item, "type") == DecisionPluginTools {
			removeItem(plugins, item)
		}
	}
	configuration := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	if err := m.SetValue(configuration, "enabled", true); err != nil {
		return err
	}
	m.Set(configuration, "select", spec)
	m.Set(at.Node, "type", Scalar(DecisionPluginTools))
	m.Set(at.Node, "configuration", configuration)
	return nil
}

// toolsSelectFromToolSelection writes a tool_selection plugin's settings as a
// select block. Its embedding thresholds, retrieval strategy and advanced
// filtering have no equivalent and leave a note.
func toolsSelectFromToolSelection(m *Migration, path string, selection *yaml.Node) (*yaml.Node, error) {
	mode := toolSelectionMode(selection)
	spec := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	m.Set(spec, "mode", Scalar(mode))
	topK, _ := strconv.Atoi(scalarAt(selection, "top_k"))
	if mode == ToolsSelectModeAdd {
		catalog := scalarAt(selection, "tools_db_path")
		if catalog == "" {
			catalog = legacyToolsDBPath
		}
		m.Set(spec, "tools_db_path", Scalar(catalog))
	}
	if err := setToolRankJudge(m, spec, mode, topK); err != nil {
		return nil, err
	}
	if preserve := mappingValue(selection, "preserve_count"); preserve != nil && mode == ToolsSelectModeFilter {
		m.Set(spec, "preserve_count", preserve)
	}
	// fallback_to_empty, true unless written, stripped the request's tools
	// when retrieval failed; add mode now keeps them.
	if fallback := mappingValue(selection, "fallback_to_empty"); mode == ToolsSelectModeFilter && (fallback == nil || yamlTrue(fallback)) {
		m.Set(spec, "on_error", Scalar(ToolsSelectOnErrorStrip))
	}
	if sticky := mappingValue(selection, "sticky"); sticky != nil {
		m.Set(spec, "sticky", sticky)
	}
	for _, field := range []string{"similarity_threshold", "relevance_threshold"} {
		if mappingValue(selection, field) != nil {
			m.Note(path+"."+field, "an embedding similarity cut has no judge equivalent; fit match.judge.min_score or min_ratio to the decision model if the selections need one")
		}
	}
	if mappingValue(selection, "advanced_filtering") != nil {
		m.Note(path+".advanced_filtering", "lexical, tag, category and history scoring are removed; to narrow the catalog, write allow_tools or block_tools in the tools plugin's policy, which also applies to the request's tools")
	}
	return spec, nil
}

// setToolRankJudge writes the judge a selection needs beside the default
// (tool_rank, the 3 best at 0.1 of the best or more): another top_k in add
// mode; in filter mode no count, as the removed filter kept every tool
// relevant enough.
func setToolRankJudge(m *Migration, spec *yaml.Node, mode string, topK int) error {
	judge := MatchJudge{Preset: presets.ToolRank, MinRatio: matchBound(0.1)}
	switch {
	case mode == ToolsSelectModeAdd && topK > 0 && topK != legacyToolsTopK:
		judge.TopK = topK
	case mode == ToolsSelectModeAdd:
		return nil
	}
	return m.SetValue(spec, "match", MatchConfig{Judge: &judge})
}

func hasFlatToolsFields(node *yaml.Node, _ string) bool {
	return len(flatToolsFields(node)) > 0
}

// flatToolsFields are the top-level fields a tools plugin writes that it now
// reads under its policy, or no longer reads.
func flatToolsFields(node *yaml.Node) []string {
	if scalarAt(node, "type") != DecisionPluginTools {
		return nil
	}
	configuration := mappingValue(node, "configuration")
	var fields []string
	for _, field := range slices.Concat(toolsPolicyFields, toolsRetrievalFields) {
		if mappingValue(configuration, field) != nil {
			fields = append(fields, field)
		}
	}
	return fields
}

func describeFlatToolsFields(at FormMatch) (string, string) {
	return "tools plugin fields " + strings.Join(flatToolsFields(at.Node), ", ") + " are not read at the top of its configuration",
		"mode, allow_tools, block_tools, strip_tool_history and trusted_facts go under policy; select replaces semantic_selection, strategy and dynamic_retrieval"
}

// migrateFlatToolsFields moves a tools plugin's policy fields under policy,
// ahead of its select block, and drops the fields of the removed retrieval.
func migrateFlatToolsFields(m *Migration, at FormMatch) error {
	configuration := mappingValue(at.Node, "configuration")
	policy := mappingValue(configuration, "policy")
	if policy == nil {
		policy = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	}
	for _, field := range toolsPolicyFields {
		value := mappingValue(configuration, field)
		if value == nil {
			continue
		}
		if mappingValue(policy, field) != nil {
			return fmt.Errorf("both %s and policy.%s are set; keep one", field, field)
		}
		m.Set(policy, field, value)
		m.Delete(configuration, field)
	}
	if dynamic := mappingValue(configuration, "dynamic_retrieval"); yamlTrue(mappingValue(dynamic, "enabled")) {
		m.Note(at.Path+".configuration.dynamic_retrieval", "history-aware retrieval is removed; write select.input.history: true for the selection to read the earlier user turns")
	}
	for _, field := range toolsRetrievalFields {
		m.Delete(configuration, field)
	}
	selection := mappingValue(configuration, "select")
	m.Delete(configuration, "select")
	if len(policy.Content) > 0 {
		m.Set(configuration, "policy", policy)
	}
	if selection != nil {
		m.Set(configuration, "select", selection)
	}
	return nil
}

func toolSelectionMode(configuration *yaml.Node) string {
	if mode := scalarAt(configuration, "mode"); mode != "" {
		return mode
	}
	return ToolsSelectModeAdd
}

// enabledPlugin returns a decision's enabled plugin of a type, nil without one.
func enabledPlugin(plugins *yaml.Node, pluginType string) *yaml.Node {
	if plugins == nil || plugins.Kind != yaml.SequenceNode {
		return nil
	}
	for _, item := range plugins.Content {
		if scalarAt(item, "type") == pluginType && yamlTrue(mappingValue(mappingValue(item, "configuration"), "enabled")) {
			return item
		}
	}
	return nil
}

func yamlTrue(node *yaml.Node) bool {
	return node != nil && node.Kind == yaml.ScalarNode && node.ShortTag() == "!!bool" && strings.EqualFold(node.Value, "true")
}
