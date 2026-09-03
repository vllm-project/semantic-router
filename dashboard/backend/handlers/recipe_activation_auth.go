package handlers

import (
	"errors"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/recipe"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// bindActivationManagementCredential adds a least-privilege service identity
// only to the realized runtime document. Package bytes remain immutable and no
// token value is serialized into YAML.
func bindActivationManagementCredential(data []byte) ([]byte, bool, error) {
	mode, _, err := managementAPIState(data)
	if err != nil {
		return nil, false, err
	}
	if mode != routerconfig.ManagementAuthModeBearer {
		return append([]byte(nil), data...), false, nil
	}
	document, auth, err := activationManagementAuthDocument(data)
	if err != nil {
		return nil, false, err
	}
	if roleErr := bindManagementCredentialRole(auth); roleErr != nil {
		return nil, false, roleErr
	}
	if tokenErr := bindManagementCredentialToken(auth); tokenErr != nil {
		return nil, false, tokenErr
	}
	encoded, err := marshalYAMLDocument(document)
	if err != nil {
		return nil, false, err
	}
	return encoded, true, nil
}

func activationManagementAuthDocument(data []byte) (*yaml.Node, *yaml.Node, error) {
	document, err := parseYAMLDocument(data)
	if err != nil {
		return nil, nil, err
	}
	root, err := documentMappingNode(document)
	if err != nil {
		return nil, nil, err
	}
	global, err := ensureMappingNode(root, "global")
	if err != nil {
		return nil, nil, err
	}
	services, err := ensureMappingNode(global, "services")
	if err != nil {
		return nil, nil, err
	}
	management, err := ensureMappingNode(services, "management_api")
	if err != nil {
		return nil, nil, err
	}
	auth, err := ensureMappingNode(management, "auth")
	if err != nil {
		return nil, nil, err
	}
	return document, auth, nil
}

func bindManagementCredentialRole(auth *yaml.Node) error {
	roles, err := ensureMappingNode(auth, "roles")
	if err != nil {
		return err
	}
	permissionNodes := make([]*yaml.Node, 0, len(recipe.ManagementCredentialPermissions()))
	for _, permission := range recipe.ManagementCredentialPermissions() {
		permissionNodes = append(permissionNodes, &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: permission})
	}
	setMappingValueNode(roles, recipe.ManagementCredentialRole, &yaml.Node{
		Kind:    yaml.SequenceNode,
		Tag:     "!!seq",
		Content: permissionNodes,
	})
	return nil
}

func bindManagementCredentialToken(auth *yaml.Node) error {
	tokens := mappingValueNode(auth, "tokens")
	if tokens == nil {
		tokens = &yaml.Node{Kind: yaml.SequenceNode, Tag: "!!seq"}
		setMappingValueNode(auth, "tokens", tokens)
	}
	if tokens.Kind != yaml.SequenceNode {
		return errors.New("management API auth.tokens must be a sequence")
	}
	found := false
	for _, token := range tokens.Content {
		if token.Kind != yaml.MappingNode {
			return errors.New("management API auth token must be a mapping")
		}
		if env := mappingValueNode(token, "env"); env != nil && env.Value == recipe.ManagementCredentialEnv {
			setMappingValueNode(token, "role", &yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: recipe.ManagementCredentialRole})
			found = true
		}
	}
	if !found {
		tokens.Content = append(tokens.Content, &yaml.Node{
			Kind: yaml.MappingNode,
			Tag:  "!!map",
			Content: []*yaml.Node{
				{Kind: yaml.ScalarNode, Tag: "!!str", Value: "env"},
				{Kind: yaml.ScalarNode, Tag: "!!str", Value: recipe.ManagementCredentialEnv},
				{Kind: yaml.ScalarNode, Tag: "!!str", Value: "role"},
				{Kind: yaml.ScalarNode, Tag: "!!str", Value: recipe.ManagementCredentialRole},
			},
		})
	}
	return nil
}
