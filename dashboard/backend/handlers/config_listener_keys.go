package handlers

import "gopkg.in/yaml.v3"

func containsAliasNode(node *yaml.Node) bool {
	if node == nil {
		return false
	}
	if node.Kind == yaml.AliasNode {
		return true
	}
	for _, child := range node.Content {
		if containsAliasNode(child) {
			return true
		}
	}
	return false
}
