package handlers

import (
	"bytes"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"gopkg.in/yaml.v3"
	sigsyaml "sigs.k8s.io/yaml"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

//nolint:gosec // G101: this is a fixed redaction marker, not a credential.
const redactedConfigSecret = "__VLLM_SR_REDACTED__"

var configEnvReference = regexp.MustCompile(`^\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)$`)

func callerCanViewConfigSecrets(r *http.Request) bool {
	if r == nil {
		return false
	}
	ac, ok := auth.AuthFromContext(r)
	return ok && ac.Perms[auth.PermConfigSecretsRead]
}

func compactConfigKey(key string) string {
	return strings.NewReplacer("_", "", "-", "", " ", "").Replace(strings.ToLower(key))
}

func sensitiveConfigKey(key string) bool {
	key = compactConfigKey(key)
	if strings.HasSuffix(key, "env") || strings.HasSuffix(key, "file") || strings.HasSuffix(key, "envset") {
		return false
	}
	for _, suffix := range []string{"apikey", "apikeys", "accesskey", "accesskeys", "password", "clientsecret", "privatekey", "authorization", "credential", "secret", "token"} {
		if strings.HasSuffix(key, suffix) {
			return true
		}
	}
	return key == "extraheaders"
}

// Decode and re-encode before redaction so aliases, merge keys and comments
// cannot retain a second, unredacted copy of a credential.
func configSecretDocument(data []byte) (*yaml.Node, error) {
	var value any
	if err := yaml.Unmarshal(data, &value); err != nil {
		return nil, err
	}
	var node yaml.Node
	if err := node.Encode(value); err != nil {
		return nil, err
	}
	return &node, nil
}

func redactConfigYAML(data []byte) ([]byte, error) {
	doc, err := parseYAMLDocument(data)
	if err != nil {
		return nil, err
	}
	if containsAliasNode(doc) {
		return nil, errConfigSecretReuse
	}
	node, err := configSecretDocument(data)
	if err != nil {
		return nil, err
	}
	if !redactConfigNode(node, false) {
		return data, nil
	}
	return yaml.Marshal(node)
}

func redactConfigNode(node *yaml.Node, sensitive bool) bool {
	return redactConfigNodeAt(node, sensitive, "")
}

func configSecretChild(key, path string) bool {
	if strings.HasSuffix(path, ".authz.gateway_credential") {
		return false
	}
	if compactConfigKey(key) != "headers" {
		return sensitiveConfigKey(key)
	}
	// These typed sections describe header names/policy, not values sent to
	// an upstream. Masking their enum fields would break an editor round trip.
	return !strings.HasSuffix(path, ".router.headers") && !strings.HasSuffix(path, ".identity.headers")
}

func redactConfigNodeAt(node *yaml.Node, sensitive bool, path string) bool {
	changed := false
	switch node.Kind {
	case yaml.MappingNode:
		for i := 0; i+1 < len(node.Content); i += 2 {
			key := node.Content[i].Value
			if redactConfigNodeAt(node.Content[i+1], sensitive || configSecretChild(key, path+"."+key), path+"."+key) {
				changed = true
			}
		}
	case yaml.SequenceNode, yaml.DocumentNode:
		for _, child := range node.Content {
			if redactConfigNodeAt(child, sensitive, path) {
				changed = true
			}
		}
	case yaml.ScalarNode:
		if node.Tag == "!!null" || node.Value == "" || configEnvReference.MatchString(node.Value) {
			return false
		}
		if sensitive {
			node.SetString(redactedConfigSecret)
			return true
		}
		// URL userinfo and query strings may carry credentials regardless of
		// the surrounding field name. Preserve only the non-secret origin/path.
		parsed, err := url.Parse(node.Value)
		if err != nil && strings.Contains(node.Value, "://") {
			node.SetString(redactedConfigSecret)
			return true
		}
		if err == nil && parsed.Host != "" && (parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "") {
			if parsed.User != nil {
				parsed.User = url.UserPassword(redactedConfigSecret, redactedConfigSecret)
			}
			if parsed.RawQuery != "" {
				parsed.RawQuery = redactedConfigSecret
			}
			if parsed.Fragment != "" {
				parsed.Fragment = redactedConfigSecret
				parsed.RawFragment = ""
			}
			node.SetString(parsed.String())
			return true
		}
	}
	return changed
}

func writeConfigViewJSON(w http.ResponseWriter, r *http.Request, value any) error {
	data, err := marshalYAMLBytes(value)
	if err != nil {
		return err
	}
	return writeConfigViewPayloadJSON(w, r, data)
}

func writeConfigViewPayloadJSON(w http.ResponseWriter, r *http.Request, data []byte) error {
	var err error
	if !callerCanViewConfigSecrets(r) {
		data, err = redactConfigYAML(data)
		if err != nil {
			return err
		}
	}
	data, err = sigsyaml.YAMLToJSON(data)
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "private, no-store")
	_, err = w.Write(data)
	return err
}

func sameConfigNode(a, b *yaml.Node) bool {
	left, err := yaml.Marshal(a)
	if err != nil {
		return false
	}
	right, err := yaml.Marshal(b)
	return err == nil && bytes.Equal(left, right)
}
