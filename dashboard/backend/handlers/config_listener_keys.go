package handlers

import (
	"errors"
	"net/http"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

var errListenerAPIKeysNotRedacted = errors.New("listener API keys could not be redacted")

func callerCanWriteConfig(r *http.Request, readonlyMode bool) bool {
	if readonlyMode {
		return false
	}
	ac, ok := auth.AuthFromContext(r)
	return ok && (ac.Perms[auth.PermConfigWrite] || ac.Perms[auth.PermConfigDeploy])
}

func withoutListenerAPIKeys(data []byte) ([]byte, error) {
	doc, err := parseYAMLDocument(data)
	if err != nil {
		return nil, err
	}
	root, err := documentMappingNode(doc)
	if err != nil {
		return nil, err
	}
	redacted := data
	if listeners := mappingValueNode(root, "listeners"); listeners != nil && listeners.Kind == yaml.SequenceNode {
		removed := false
		for _, listener := range listeners.Content {
			for value := mappingValueNode(listener, "api_keys"); value != nil; value = mappingValueNode(listener, "api_keys") {
				if containsAliasNode(value) {
					return nil, errListenerAPIKeysNotRedacted
				}
				deleteMappingValueNode(listener, "api_keys")
				removed = true
			}
		}
		if removed {
			if redacted, err = marshalYAMLDocument(doc); err != nil {
				return nil, err
			}
		}
	}
	var check struct {
		Listeners []struct {
			APIKeys []string `yaml:"api_keys"`
		} `yaml:"listeners"`
	}
	if err := yaml.Unmarshal(redacted, &check); err != nil {
		return nil, errListenerAPIKeysNotRedacted
	}
	for _, listener := range check.Listeners {
		if len(listener.APIKeys) > 0 {
			return nil, errListenerAPIKeysNotRedacted
		}
	}
	return redacted, nil
}

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
