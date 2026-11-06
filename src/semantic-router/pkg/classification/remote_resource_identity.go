package classification

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/url"
	"path"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// remoteOperationIdentity follows the connector's actual request target and
// model selector. Catalog aliases and unused model names cannot bypass capacity.
func remoteOperationIdentity(adapter string, deployment config.ModelDeployment) (string, string, error) {
	base, _, err := config.SplitEndpoint(deployment.Endpoint)
	if err != nil {
		return "", "", fmt.Errorf("remote operation endpoint is invalid")
	}
	endpoint, err := url.Parse(base)
	if err != nil || endpoint.Hostname() == "" || endpoint.User != nil || endpoint.Fragment != "" {
		return "", "", fmt.Errorf("remote operation endpoint is invalid")
	}
	var operation, model string
	switch adapter {
	case config.RemoteClassifierProtocolHTTPClassify:
		operation = "/classify" // The body has inputs only; the model is never sent.
	case config.RemoteClassifierProtocolHTTPChat:
		operation, model = "/chat/completions", deployment.Model
	default:
		return "", "", fmt.Errorf("remote operation adapter %q is unsupported", adapter)
	}
	endpoint.Scheme = strings.ToLower(endpoint.Scheme)
	port := endpoint.Port()
	if port == "" {
		if endpoint.Scheme == "https" {
			port = "443"
		} else {
			port = "80"
		}
	}
	endpoint.Host = net.JoinHostPort(strings.ToLower(endpoint.Hostname()), port)
	endpoint.Path = path.Join(endpoint.Path, operation)
	endpoint.RawPath = ""
	return endpoint.String(), model, nil
}

// credentialFingerprint identifies a deployment's credential without
// holding it: where the key is read from, and how it is sent.
func credentialFingerprint(deployment config.ModelDeployment) string {
	auth := deployment.Auth
	if auth == nil {
		return ""
	}
	key := ""
	if auth.HasInlineKey() {
		key, _ = auth.Key()
	}
	description, _ := json.Marshal(struct {
		Header, Prefix, Env, File, Key string
	}{auth.HeaderName(), auth.HeaderValue(""), auth.APIKeyEnv, auth.APIKeyFile, key})
	sum := sha256.Sum256(description)
	return hex.EncodeToString(sum[:])
}
