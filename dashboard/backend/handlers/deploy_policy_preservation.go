package handlers

import "gopkg.in/yaml.v3"

// preserveBaseDecisionAdaptations copies each base decision's adaptations into
// the fragment decision with the same name, matching recipes by name first.
// The DSL cannot express adaptations, so compiled fragments never carry them;
// a value already in the fragment wins. dsl.MergeRoutingIntoBase applies the
// same rule for CLI compiles with a base config.
func preserveBaseDecisionAdaptations(baseRoot, fragmentRoot *yaml.Node) {
	preserveDecisionAdaptations(mappingValueNode(baseRoot, "routing"), mappingValueNode(fragmentRoot, "routing"))
	baseRecipes := sequenceItemsByName(mappingValueNode(baseRoot, "recipes"))
	for _, recipe := range sequenceItems(mappingValueNode(fragmentRoot, "recipes")) {
		baseRecipe := baseRecipes[itemName(recipe)]
		preserveDecisionAdaptations(mappingValueNode(baseRecipe, "routing"), mappingValueNode(recipe, "routing"))
	}
}

func preserveDecisionAdaptations(baseRouting, fragmentRouting *yaml.Node) {
	baseDecisions := sequenceItemsByName(mappingValueNode(baseRouting, "decisions"))
	for _, decision := range sequenceItems(mappingValueNode(fragmentRouting, "decisions")) {
		if mappingValueNode(decision, "adaptations") != nil {
			continue
		}
		if adaptations := mappingValueNode(baseDecisions[itemName(decision)], "adaptations"); adaptations != nil {
			setMappingValueNode(decision, "adaptations", adaptations)
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
