package config

import (
	"fmt"
	"regexp"
	"strings"
)

// ClientAuthType names how an entrypoint authenticates its clients.
type ClientAuthType string

// ClientAuthBearerAPIKey accepts `Authorization: Bearer <key>` carrying one of
// the keys the entrypoint's environment variables hold, as `vllm serve
// --api-key` does.
const ClientAuthBearerAPIKey ClientAuthType = "bearer_api_key" //nolint:gosec // G101: names a policy type, not a credential

// EntrypointClientAuth requires a client key for every model name of one
// entrypoint. It names the environment variables that hold the accepted keys;
// the keys never appear in configuration. Several variables let a key rotate.
type EntrypointClientAuth struct {
	Type       ClientAuthType `yaml:"type" jsonschema:"enum=bearer_api_key"`
	APIKeyEnvs []string       `yaml:"api_key_envs"`
}

// clientAuthAvailable admits entrypoints[].client_auth. It stays off until
// every request path enforces the policy, so that no configuration declares a
// key the Router would not check.
var clientAuthAvailable = false

var environmentVariableName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// Clone returns a copy that shares no slice with a.
func (a *EntrypointClientAuth) Clone() *EntrypointClientAuth {
	if a == nil {
		return nil
	}
	clone := *a
	clone.APIKeyEnvs = append([]string(nil), a.APIKeyEnvs...)
	return &clone
}

// normalizeEntrypointClientAuth validates the policy of entrypoints[index].
// Messages never quote an api_key_envs item: one written by mistake may be the
// key itself.
func normalizeEntrypointClientAuth(index int, auth *EntrypointClientAuth) (*EntrypointClientAuth, error) {
	if auth == nil {
		return nil, nil
	}
	path := fmt.Sprintf("entrypoints[%d].client_auth", index)
	if auth.Type != ClientAuthBearerAPIKey {
		return nil, fmt.Errorf("%s.type must be %q, got %q", path, ClientAuthBearerAPIKey, auth.Type)
	}
	if len(auth.APIKeyEnvs) == 0 {
		return nil, fmt.Errorf("%s.api_key_envs must name at least one environment variable", path)
	}
	first := make(map[string]int, len(auth.APIKeyEnvs))
	for i, name := range auth.APIKeyEnvs {
		if !environmentVariableName.MatchString(name) {
			return nil, fmt.Errorf("%s.api_key_envs[%d] must be an environment variable name: letters, digits and underscores, not starting with a digit", path, i)
		}
		if earlier, repeated := first[name]; repeated {
			return nil, fmt.Errorf("%s.api_key_envs[%d] repeats api_key_envs[%d]", path, i, earlier)
		}
		first[name] = i
	}
	return auth.Clone(), nil
}

// checkRawClientAuth runs before environment expansion, on every client_auth
// key whatever its value. A policy names variables the Router reads itself, so
// a $ reference anywhere in it would expand to a key that a later error could
// quote. Until the Router enforces policies, the key is rejected outright,
// null included.
func checkRawClientAuth(raw map[string]interface{}) error {
	entrypoints, _ := raw["entrypoints"].([]interface{})
	for index, item := range entrypoints {
		auth, present := nestedStringMap(item)["client_auth"]
		if !present {
			continue
		}
		path := fmt.Sprintf("entrypoints[%d].client_auth", index)
		if holdsEnvReference(auth) {
			return fmt.Errorf("%s must not contain $ references: name the variables in api_key_envs and the Router reads them itself", path)
		}
		if !clientAuthAvailable {
			return fmt.Errorf("%s: entrypoint client authentication is not available yet; remove client_auth", path)
		}
	}
	return nil
}

// holdsEnvReference reports whether any key or scalar of a raw YAML value
// contains a $.
func holdsEnvReference(value interface{}) bool {
	switch typed := value.(type) {
	case string:
		return strings.Contains(typed, "$")
	case []interface{}:
		for _, item := range typed {
			if holdsEnvReference(item) {
				return true
			}
		}
	case map[string]interface{}:
		for key, item := range typed {
			if strings.Contains(key, "$") || holdsEnvReference(item) {
				return true
			}
		}
	case map[interface{}]interface{}:
		for key, item := range typed {
			if holdsEnvReference(key) || holdsEnvReference(item) {
				return true
			}
		}
	}
	return false
}
