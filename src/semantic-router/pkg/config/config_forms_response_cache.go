package config

import (
	"fmt"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// responseCacheAliases are the spellings the response_cache plugin type had.
var responseCacheAliases = []string{"semantic-cache", "semantic_cache", "response-cache"}

// The response_cache fields that were read at the top of its configuration,
// and the threshold its semantic block held, before semantic.match.
const (
	responseCacheFlatThreshold     = "similarity_threshold"
	responseCacheAllowControls     = "allow_request_controls"
	responseCacheControlHeader     = "control_header"
	responseCacheSemanticThreshold = "semantic.similarity_threshold"
)

// responseCacheForms are the forms of the response cache before semantic
// reuse went through a match block: the plugin type's aliases and the store's
// old name, rewritten first; then the flat threshold and request-control
// fields, now semantic.match.recall and request_controls.
var responseCacheForms = []ConfigForm{
	{
		Code:  CodeRemovedPluginType,
		Paths: RoutingScopes("decisions[].plugins[]"),
		Match: func(node *yaml.Node, _ string) bool {
			return slices.Contains(responseCacheAliases, scalarAt(node, "type"))
		},
		Describe: func(at FormMatch) (string, string) {
			return "plugin " + scalarAt(at.Node, "type") + " is removed", "type: " + DecisionPluginResponseCache
		},
		Migrate: migrateResponseCacheAlias,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.stores.semantic_cache"},
		What:    "global.stores.semantic_cache is removed",
		Becomes: "global.stores.response_cache",
		Migrate: func(m *Migration, at FormMatch) error {
			return m.Rename(at.Parent, "semantic_cache", "response_cache")
		},
	},
	{
		Code:     CodeRemovedField,
		Paths:    RoutingScopes("decisions[].plugins[]"),
		Match:    hasFlatResponseCacheFields,
		Describe: describeFlatResponseCacheFields,
		Migrate:  migrateFlatResponseCacheFields,
	},
}

func migrateResponseCacheAlias(m *Migration, at FormMatch) error {
	for _, item := range at.Parent.Content {
		if item != at.Node && scalarAt(item, "type") == DecisionPluginResponseCache {
			return fmt.Errorf("the decision also has a %s plugin; keep one", DecisionPluginResponseCache)
		}
	}
	m.Set(at.Node, "type", Scalar(DecisionPluginResponseCache))
	return nil
}

func hasFlatResponseCacheFields(node *yaml.Node, _ string) bool {
	return len(flatResponseCacheFields(node)) > 0
}

// flatResponseCacheFields are the fields a response cache plugin, under any
// of its spellings, writes that it no longer reads.
func flatResponseCacheFields(node *yaml.Node) []string {
	pluginType := scalarAt(node, "type")
	if pluginType != DecisionPluginResponseCache && !slices.Contains(responseCacheAliases, pluginType) {
		return nil
	}
	configuration := mappingValue(node, "configuration")
	var fields []string
	for _, field := range []string{responseCacheFlatThreshold, responseCacheAllowControls, responseCacheControlHeader} {
		if mappingValue(configuration, field) != nil {
			fields = append(fields, field)
		}
	}
	if mappingValue(mappingValue(configuration, "semantic"), "similarity_threshold") != nil {
		fields = append(fields, responseCacheSemanticThreshold)
	}
	return fields
}

func describeFlatResponseCacheFields(at FormMatch) (string, string) {
	fields := flatResponseCacheFields(at.Node)
	if len(fields) == 0 { // MigrateConfig describes a match again after rewriting it
		fields = []string{responseCacheFlatThreshold, responseCacheAllowControls, responseCacheControlHeader, responseCacheSemanticThreshold}
	}
	what := "response_cache field " + fields[0] + " is removed"
	if len(fields) > 1 {
		what = "response_cache fields " + strings.Join(fields, ", ") + " are removed"
	}
	return what, "the similarity threshold is semantic.match.recall.min_similarity; allow_request_controls and control_header are request_controls.enabled and request_controls.header"
}

// migrateFlatResponseCacheFields writes the threshold as the recall bound of
// the semantic tier's match block, which keeps the default recall depth, and
// the request-control fields under request_controls.
func migrateFlatResponseCacheFields(m *Migration, at FormMatch) error {
	configuration := mappingValue(at.Node, "configuration")
	semantic := mappingValue(configuration, "semantic")
	threshold := mappingValue(configuration, responseCacheFlatThreshold)
	if nested := mappingValue(semantic, "similarity_threshold"); nested != nil {
		if threshold != nil {
			return fmt.Errorf("both %s and %s are set; keep one", responseCacheFlatThreshold, responseCacheSemanticThreshold)
		}
		threshold = nested
	}
	if threshold != nil {
		if err := setResponseCacheRecall(m, configuration, threshold); err != nil {
			return err
		}
	}
	return moveResponseCacheControls(m, configuration)
}

func setResponseCacheRecall(m *Migration, configuration, threshold *yaml.Node) error {
	semantic := mappingValue(configuration, "semantic")
	if mappingValue(semantic, "match") != nil {
		return fmt.Errorf("both a similarity threshold and semantic.match are set; keep semantic.match")
	}
	recall := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	if err := m.SetValue(recall, "top_k", MatchResponseCache.Default().Recall.TopK); err != nil {
		return err
	}
	m.Set(recall, "min_similarity", threshold)
	match := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	m.Set(match, "recall", recall)
	if semantic == nil {
		semantic = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
		m.Set(configuration, "semantic", semantic)
	}
	m.Delete(semantic, "similarity_threshold")
	m.Set(semantic, "match", match)
	m.Delete(configuration, responseCacheFlatThreshold)
	return nil
}

func moveResponseCacheControls(m *Migration, configuration *yaml.Node) error {
	allow := mappingValue(configuration, responseCacheAllowControls)
	header := mappingValue(configuration, responseCacheControlHeader)
	if allow == nil && header == nil {
		return nil
	}
	if mappingValue(configuration, "request_controls") != nil {
		return fmt.Errorf("both request_controls and %s or %s are set; keep request_controls", responseCacheAllowControls, responseCacheControlHeader)
	}
	controls := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	if yamlTrue(allow) {
		m.Set(controls, "enabled", allow)
	}
	if header != nil && strings.TrimSpace(header.Value) != "" {
		m.Set(controls, "header", header)
	}
	m.Delete(configuration, responseCacheAllowControls)
	m.Delete(configuration, responseCacheControlHeader)
	if len(controls.Content) > 0 {
		m.Set(configuration, "request_controls", controls)
	}
	return nil
}
