package handlers

import (
	"net/http"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
	sigsyaml "sigs.k8s.io/yaml"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

const keyReferenceDestinationRefused = "Forbidden: a saved key's reference (auth.api_key_file, or the key Secret's variable) can stay only where the saved config uses it, on the same destination; enter the key itself to send it elsewhere"

// rejectKeyReferenceRedirect refuses a save, by a caller who may not read
// configuration secrets, that names a saved key's reference where the stored
// config does not. A save moves a typed key into a key file, or on Kubernetes
// into the key Secret's VSR_KEY_ variables, and the reference is not redacted
// the way the key was, so otherwise a writer could move a backend that reads
// a saved key to a host of theirs, or name any file the Router can read as a
// key. Each block that names a reference must match, in the destination
// fields a redacted key is checked on (configSecretTargetsChanged), a block
// that names it now. A key the caller types is theirs to send anywhere, and a
// variable the operator wrote stays a reference the editor shows as written.
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

// keyReferenceUsers maps each saved key's reference a document names to the
// blocks, such as a backend or a deployment, that send the key it reads: an
// auth block's api_key_file, or its api_key_env of the key Secret, names the
// block's parent, and an api_key's reference to the key Secret names the block
// that holds it. Listener keys check what clients send, so they are not
// followed. The document is read as the Router's YAML decoder reads it:
// aliases and merge keys resolved.
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
	var visit func(node, block *yaml.Node)
	visit = func(node, block *yaml.Node) {
		if file := mappingValueNode(node, "api_key_file"); file != nil && file.Value != "" {
			users["file:"+file.Value] = append(users["file:"+file.Value], block)
		}
		if env := mappingValueNode(node, "api_key_env"); env != nil && strings.HasPrefix(env.Value, configwriter.KeySecretEntryPrefix) {
			users["env:"+env.Value] = append(users["env:"+env.Value], block)
		}
		if key := mappingValueNode(node, "api_key"); key != nil && key.Kind == yaml.ScalarNode {
			for _, name := range routerconfig.EnvironmentReferenceNames(key.Value) {
				if strings.HasPrefix(name, configwriter.KeySecretEntryPrefix) {
					users["env:"+name] = append(users["env:"+name], node)
				}
			}
		}
		if node.Kind == yaml.MappingNode {
			block = node
		}
		for _, child := range node.Content {
			visit(child, block)
		}
	}
	visit(doc, doc)
	return users, nil
}
