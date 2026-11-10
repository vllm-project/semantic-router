package connector

import (
	"context"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/principal"
)

// ForDeployment connects to an attached or remote deployment through its
// connection block: its endpoint (whose query every call carries), auth (the
// key read for every call, so a rotated file or variable applies), headers,
// timeout_ms, max_retries, tls and limits. defaults fill what the deployment
// leaves unset.
func ForDeployment(deployment config.ModelDeployment, defaults Options) (*Client, error) {
	base, query, err := config.SplitEndpoint(deployment.Endpoint)
	if err != nil {
		return nil, err
	}
	options := defaults
	options.AttemptTimeout = deployment.TimeoutOr(defaults.AttemptTimeout)
	options.MaxRetries = deployment.RetryCount(defaults.MaxRetries)
	options.MaxRequestBytes = deployment.RequestLimitOr(defaults.MaxRequestBytes)
	options.MaxResponseBytes = deployment.ResponseLimitOr(defaults.MaxResponseBytes)
	options.Query = query
	if deployment.TLS != nil {
		if options.TLSConfig, err = deployment.TLS.ClientConfig(); err != nil {
			return nil, err
		}
	}
	return New(base, DeploymentAuthorizer(deployment.Auth, deployment.Headers), options)
}

// DeploymentAuthorizer sets a deployment's headers and its credential, in
// auth.header (default Authorization) after auth.prefix (default Bearer). A
// delegated deployment's credential is the one the call's request carries
// (package principal); a call without one is refused before it is sent.
func DeploymentAuthorizer(auth *config.DeploymentAuth, headers map[string]string) func(context.Context, *http.Request) error {
	return func(ctx context.Context, request *http.Request) error {
		for name, value := range headers {
			request.Header.Set(name, value)
		}
		if auth == nil {
			return nil
		}
		if auth.Delegated() {
			credential, err := principal.Delegated(ctx)
			if err != nil {
				return err
			}
			request.Header.Set(auth.HeaderName(), auth.HeaderValue(credential))
			return nil
		}
		key, err := auth.Key()
		if err != nil {
			return fmt.Errorf("credential: %w", err)
		}
		request.Header.Set(auth.HeaderName(), auth.HeaderValue(key))
		return nil
	}
}
