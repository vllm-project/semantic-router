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
	if err := restoreConfigSecretNode(oldNode, newNode, false, false, ""); err != nil {
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
		if !redactConfigNode(&masked, sensitive) || masked.Value != updated.Value {
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

func configSecretTargetChanged(old, updated *yaml.Node) bool {
	for _, key := range []string{"name", "id", "address", "addresses", "host", "hosts", "hostname", "port", "endpoint", "endpoints", "url", "base_url", "baseURL", "provider", "provider_profile", "type", "protocol", "scheme", "command", "transport", "api_key_env", "api_key_file"} {
		before, after := mappingValueNode(old, key), mappingValueNode(updated, key)
		if before == nil && after == nil {
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
