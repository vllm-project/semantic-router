package handlers

import (
	"net/http"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
	sigsyaml "sigs.k8s.io/yaml"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const keyReferenceDestinationRefused = "Forbidden: an environment reference (${NAME} or $NAME) or a saved key's file or variable (api_key_file, api_key_env, key_env, password_env, hmac_key_env) can stay only where the saved config uses it, on the same destination; enter the value itself to send it elsewhere"

// rejectKeyReferenceRedirect refuses a save, by a caller who may not read
// configuration secrets, that names a reference where the stored config does
// not. The Router expands every variable of its environment across the whole
// document, and reads a saved key from the file or key Secret variable an
// auth block names, while the editor shows the reference, not the value. So a
// writer could otherwise send a value they never saw, a provider key such as
// ${OPENAI_API_KEY} or any variable like $HOME, to a host of theirs, in a
// backend, a plugin prompt, a path, a header or a store password. Each block
// that names a reference must match, in the destination fields a redacted key
// is checked on (configSecretTargetsChanged), a block that names it now. A
// value the caller types is theirs to send anywhere.
func rejectKeyReferenceRedirect(w http.ResponseWriter, r *http.Request, existing, updated []byte) bool {
	if callerCanViewConfigSecrets(r) || keyReferencesKept(existing, updated) {
		return false
	}
	http.Error(w, keyReferenceDestinationRefused, http.StatusForbidden)
	return true
}

// keyReferencesKept reports whether each block of updated that names a key
// reference matches a block of existing that names it. A document that
// cannot be read keeps none.
func keyReferencesKept(existing, updated []byte) bool {
	stored, err := keyReferenceUsers(existing)
	if err != nil {
		return false
	}
	saved, err := keyReferenceUsers(updated)
	if err != nil {
		return false
	}
	for reference, users := range saved {
		for _, user := range users {
			if !slices.ContainsFunc(stored[reference], func(previous *yaml.Node) bool {
				return !configSecretTargetsChanged(previous, user, "")
			}) {
				return false
			}
		}
	}
	return true
}

// keyReferenceUsers maps each reference a document names to the blocks, such
// as a backend, a deployment, a plugin or a store, that send what it reads:
// every environment variable the Router expands in a string value, and the
// file or key Secret variable an auth block names. The document is read as
// the Router's YAML decoder reads it: aliases and merge keys resolved.
func keyReferenceUsers(data []byte) (map[string][]*yaml.Node, error) {
	resolved, err := sigsyaml.YAMLToJSON(data)
	if err != nil {
		return nil, err
	}
	doc, err := parseYAMLDocument(resolved)
	if err != nil {
		return nil, err
	}
	users := map[string][]*yaml.Node{}
	addKeyReferenceUsers(users, doc, doc, "")
	return users, nil
}

// addKeyReferenceUsers adds every reference node names, each with the block
// that sends what it reads: the mapping that holds the field, or for a field
// of an auth block, the mapping that holds that block, whose endpoint the
// value goes to. A listener's client keys (api_keys, api_key_files) are not
// followed, as they check what clients send, not a value the Router sends on.
// path is node's path as the editor's redaction reads it.
func addKeyReferenceUsers(users map[string][]*yaml.Node, node, block *yaml.Node, path string) {
	switch node.Kind {
	case yaml.MappingNode:
		if !strings.HasSuffix(path, ".auth") {
			block = node
		}
		for i := 0; i+1 < len(node.Content); i += 2 {
			key, value := node.Content[i].Value, node.Content[i+1]
			field := path + "." + key
			if key == "api_key_file" || routerconfig.IsSecretEnvReferenceField(key) {
				if value.Value != "" {
					prefix := "env:"
					if key == "api_key_file" {
						prefix = "file:"
					}
					users[prefix+value.Value] = append(users[prefix+value.Value], block)
				}
				continue
			}
			if field == ".listeners.api_keys" || field == ".listeners.api_key_files" {
				continue
			}
			addKeyReferenceUsers(users, value, block, field)
		}
	case yaml.ScalarNode:
		for _, name := range routerconfig.EnvironmentReferenceNames(node.Value) {
			users["env:"+name] = append(users["env:"+name], block)
		}
	default:
		for _, child := range node.Content {
			addKeyReferenceUsers(users, child, block, path)
		}
	}
}
