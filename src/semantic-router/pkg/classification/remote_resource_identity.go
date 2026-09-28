package classification

import (
	"fmt"
	"net"
	"net/url"
	"path"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// remoteOperationIdentity follows the connector's actual request target and
// model selector. Catalog aliases and unused model names cannot bypass capacity.
func remoteOperationIdentity(adapter string, external *config.ExternalModelConfig) (string, string, error) {
	address := strings.TrimSpace(external.ModelEndpoint.Address)
	scheme := strings.ToLower(strings.TrimSpace(external.ModelEndpoint.Protocol))
	if scheme == "" {
		scheme = "http"
	}
	absolute := strings.Contains(address, "://")
	base := address
	if !absolute {
		base = fmt.Sprintf("%s://%s:%d", scheme, address, external.ModelEndpoint.Port)
	}
	endpoint, err := url.Parse(base)
	if err != nil || endpoint.Hostname() == "" || (endpoint.Scheme != "http" && endpoint.Scheme != "https") || endpoint.User != nil || endpoint.RawQuery != "" || endpoint.Fragment != "" {
		return "", "", fmt.Errorf("remote operation endpoint is invalid")
	}
	var operation, model string
	switch adapter {
	case config.RemoteClassifierProtocolHTTPClassify:
		operation = "/classify" // The body has inputs only; ModelName is never sent.
	case config.RemoteClassifierProtocolHTTPChat:
		operation, model = "/v1/chat/completions", external.ModelName
		if absolute {
			operation = "/chat/completions"
		} // Explicit endpoint adapters supply their API base path.
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
	endpoint.RawQuery = ""
	return endpoint.String(), model, nil
}
