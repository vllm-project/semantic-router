package handlers

import (
	"bytes"
	"errors"
	"strings"

	"gopkg.in/yaml.v3"
)

var errConfigSecretReuse = errors.New("redacted credentials must match an existing field; replace or clear credentials when their destination changes")

// Restore only exact placeholders from the same stored credential and target.
// The caller holds the ordinary config mutation lock while reading and merging.
func restoreConfigSecrets(existing, updated []byte) ([]byte, error) {
	if !bytes.Contains(updated, []byte(redactedConfigSecret)) {
		return updated, nil
	}
	oldNode, err := configSecretDocument(existing)
	if err != nil {
		return nil, err
	}
	newNode, err := configSecretDocument(updated)
	if err != nil {
		return nil, err
	}
	if err := restoreConfigSecretNode(oldNode, newNode, false, configSecretTargetsChanged(oldNode, newNode), ""); err != nil {
		return nil, err
	}
	return yaml.Marshal(newNode)
}

func restoreConfigSecretNode(old, updated *yaml.Node, sensitive, targetChanged bool, path string) error {
	if updated.Kind == yaml.ScalarNode && strings.Contains(updated.Value, redactedConfigSecret) {
		if old == nil || old.Kind != yaml.ScalarNode || targetChanged {
			return errConfigSecretReuse
		}
		masked := *old
		if !redactConfigNodeAt(&masked, sensitive, path) || masked.Value != updated.Value {
			return errConfigSecretReuse
		}
		*updated = *cloneYAMLNode(old)
		return nil
	}
	switch updated.Kind {
	case yaml.MappingNode:
		targetChanged = targetChanged || configSecretTargetChanged(old, updated)
		for i := 0; i+1 < len(updated.Content); i += 2 {
			key := updated.Content[i].Value
			if err := restoreConfigSecretNode(mappingValueNode(old, key), updated.Content[i+1], sensitive || configSecretChild(key, path+"."+key), targetChanged, path+"."+key); err != nil {
				return err
			}
		}
	case yaml.SequenceNode:
		for i, child := range updated.Content {
			previous := matchingConfigSequenceItem(old, child, i)
			if err := restoreConfigSecretNode(previous, child, sensitive, targetChanged, path); err != nil {
				return err
			}
		}
	}
	return nil
}

func matchingConfigSequenceItem(old, updated *yaml.Node, index int) *yaml.Node {
	if old == nil || old.Kind != yaml.SequenceNode {
		return nil
	}
	for _, key := range []string{"name", "id"} {
		identity := mappingValueNode(updated, key)
		if identity == nil {
			continue
		}
		var match *yaml.Node
		for _, item := range old.Content {
			if previous := mappingValueNode(item, key); previous != nil && sameConfigNode(previous, identity) {
				if match != nil {
					return nil
				}
				match = item
			}
		}
		return match
	}
	if index < len(old.Content) {
		return old.Content[index]
	}
	return nil
}

var configSecretTargetFields = [...]string{"name", "id", "address", "addresses", "cluster_addresses", "cluster_mode", "host", "hosts", "hostname", "port", "endpoint", "endpoints", "url", "base_url", "baseURL", "provider", "provider_profile", "type", "protocol", "scheme", "command", "transport", "api_key_env", "api_key_file"}

// Config editors submit a whole document, and some schemas associate a secret
// with a target in another branch. Without a schema-level association for every
// supported extension, fail closed: preserve no placeholder if any target changed.
func configSecretTargetsChanged(old, updated *yaml.Node) bool {
	if updated == nil {
		return configSecretContainsTarget(old)
	}
	if old == nil || old.Kind != updated.Kind {
		return configSecretContainsTarget(old)
	}
	if configSecretTargetChanged(old, updated) {
		return true
	}
	switch updated.Kind {
	case yaml.MappingNode:
		for i := 0; i+1 < len(updated.Content); i += 2 {
			key := updated.Content[i].Value
			if configSecretTargetsChanged(mappingValueNode(old, key), updated.Content[i+1]) {
				return true
			}
		}
		for i := 0; i+1 < len(old.Content); i += 2 {
			key := old.Content[i].Value
			if mappingValueNode(updated, key) == nil && configSecretContainsTarget(old.Content[i+1]) {
				return true
			}
		}
	case yaml.SequenceNode:
		for i, child := range updated.Content {
			if configSecretTargetsChanged(matchingConfigSequenceItem(old, child, i), child) {
				return true
			}
		}
		for i, previous := range old.Content {
			if matchingConfigSequenceItem(updated, previous, i) == nil && configSecretContainsTarget(previous) {
				return true
			}
		}
	}
	return false
}

func configSecretContainsTarget(node *yaml.Node) bool {
	if node == nil {
		return false
	}
	switch node.Kind {
	case yaml.MappingNode:
		for i := 0; i+1 < len(node.Content); i += 2 {
			key := node.Content[i].Value
			if isConfigSecretTargetField(key) || configSecretContainsTarget(node.Content[i+1]) {
				return true
			}
		}
	case yaml.SequenceNode:
		for _, child := range node.Content {
			if configSecretContainsTarget(child) {
				return true
			}
		}
	}
	return false
}

func isConfigSecretTargetField(candidate string) bool {
	for _, key := range configSecretTargetFields {
		if key == candidate {
			return true
		}
	}
	return false
}

func configSecretTargetChanged(old, updated *yaml.Node) bool {
	for _, key := range configSecretTargetFields {
		before, after := mappingValueNode(old, key), mappingValueNode(updated, key)
		if key == "cluster_addresses" || key == "cluster_mode" {
			// A newly introduced cluster can redirect an existing password too.
			// Explicit empty/false defaults do not change the selected target.
			if defaultConfigSecretClusterTarget(before, key) && defaultConfigSecretClusterTarget(after, key) {
				continue
			}
		} else if before == nil {
			continue
		}
		if before == nil || after == nil {
			return true
		}
		masked := cloneYAMLNode(before)
		redactConfigNode(masked, false)
		if !sameConfigNode(before, after) && !sameConfigNode(masked, after) {
			return true
		}
	}
	return false
}

func defaultConfigSecretClusterTarget(node *yaml.Node, key string) bool {
	if node == nil || node.Tag == "!!null" {
		return true
	}
	if key == "cluster_mode" {
		return node.Tag == "!!bool" && node.Value == "false"
	}
	return node.Kind == yaml.SequenceNode && len(node.Content) == 0
}
