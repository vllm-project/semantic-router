package handlers

import (
	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl"
)

// preserveBaseDecisionFields copies each base decision's dsl.BaseDecisionFields
// into the fragment decision with the same name, matching recipes by name
// first. The DSL cannot express them, so compiled fragments never carry them;
// a value already in the fragment wins. dsl.MergeRoutingIntoBase applies the
// same rule for CLI compiles with a base config.
func preserveBaseDecisionFields(baseRoot, fragmentRoot *yaml.Node) {
	baseRecipes := sequenceItemsByName(mappingValueNode(baseRoot, "recipes"))
	for _, recipe := range sequenceItems(mappingValueNode(fragmentRoot, "recipes")) {
		preserveDecisionFields(baseRecipes[itemName(recipe)], recipe)
	}
}

func preserveDecisionFields(baseRecipe, fragmentRecipe *yaml.Node) {
	baseDecisions := sequenceItemsByName(mappingValueNode(baseRecipe, "decisions"))
	for _, decision := range sequenceItems(mappingValueNode(fragmentRecipe, "decisions")) {
		for _, field := range dsl.BaseDecisionFields() {
			if mappingValueNode(decision, field) != nil {
				continue
			}
			if value := mappingValueNode(baseDecisions[itemName(decision)], field); value != nil {
				setMappingValueNode(decision, field, value)
			}
		}
	}
}

func sequenceItems(node *yaml.Node) []*yaml.Node {
	if node == nil || node.Kind != yaml.SequenceNode {
		return nil
	}
	return node.Content
}

func sequenceItemsByName(node *yaml.Node) map[string]*yaml.Node {
	items := make(map[string]*yaml.Node)
	for _, item := range sequenceItems(node) {
		if name := itemName(item); name != "" {
			items[name] = item
		}
	}
	return items
}

func itemName(item *yaml.Node) string {
	if name := mappingValueNode(item, "name"); name != nil && name.Kind == yaml.ScalarNode {
		return name.Value
	}
	return ""
}
