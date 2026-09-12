package extproc

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

const (
	// shadowDispatchErrorBodyBytes bounds how much of a non-2xx shadow
	// response the connector reads for diagnostics. It is never stored.
	shadowDispatchErrorBodyBytes = 8 * 1024
	// shadowDispatchMaxRequestBytes bounds the encoded shadow body. The
	// primary request already passed the router's own limits, so this only
	// guards the connector contract.
	shadowDispatchMaxRequestBytes = 16 << 20
	shadowDispatchOperationName   = "shadow_dispatch"
)

type shadowTarget struct {
	logicalModel  string
	backendName   string
	backendType   string
	upstreamModel string
	profile       *config.ProviderProfile
	format        llmprotocol.WireFormat
	baseURL       string
	path          string
	query         string
}

type preparedShadowCall struct {
	target *shadowTarget
	client *connector.Client
	body   []byte
}

func (d *shadowDispatcher) call(ctx context.Context, job *shadowJob) shadowResult {
	result := shadowResult{startedAt: d.now()}
	prepared, reason, err := d.prepareShadowCall(job)
	if err != nil {
		return d.failShadow(result, reason, err)
	}
	result.shadowBackend = prepared.target.backendName
	operation := connector.Operation{
		Name:      shadowDispatchOperationName,
		Method:    http.MethodPost,
		Path:      prepared.target.path,
		Query:     prepared.target.query,
		RetrySafe: true,
	}
	response, err := prepared.client.DoRequest(ctx, operation, connector.Request{
		Body:    prepared.body,
		Headers: shadowCallHeaders(job, prepared.target),
	})
	if err != nil {
		reason, attempts, status := shadowConnectorFailure(ctx, d.ctx, err)
		result.attempts = attempts
		result.statusCode = status
		return d.failShadow(result, reason, err)
	}
	result.attempts = response.Attempts
	result.statusCode = response.StatusCode
	result.responseBytes = len(response.Body)
	decoded, err := job.engine.TranslateResponse(prepared.target.format, prepared.target.format, response.Body, nil)
	if err != nil {
		return d.failShadow(result, shadowReasonMalformedResponse, err)
	}
	result.stopReason = string(decoded.Response.StopReason)
	result.inputTokens = shadowTokenCount(decoded.Response.Usage.InputTotal)
	result.outputTokens = shadowTokenCount(decoded.Response.Usage.OutputTotal)
	result.text = semanticResponseText(decoded.Response)
	result.verdict = shadowVerdictCompleted
	result.reason = shadowReasonCompleted
	result.finishedAt = d.now()
	return result
}

func (d *shadowDispatcher) failShadow(result shadowResult, reason string, err error) shadowResult {
	result.verdict = shadowVerdictFailed
	result.reason = reason
	result.finishedAt = d.now()
	if err != nil {
		result.err = truncateShadowText(err.Error(), shadowDispatchErrorTextLimit)
	}
	return result
}

// prepareShadowCall resolves the shadow backend, its connector, and the
// encoded body once; the connector then owns every transport attempt.
func (d *shadowDispatcher) prepareShadowCall(job *shadowJob) (*preparedShadowCall, string, error) {
	target, err := resolveShadowTarget(job.routerConfig, job.cfg.Model)
	if err != nil {
		return nil, shadowReasonBackendUnresolved, err
	}
	if job.hasDynamoExt {
		if !strings.EqualFold(strings.TrimSpace(target.backendType), "dynamo") {
			return nil, shadowReasonDynamoBackend, unsupportedDynamoBackendError(target.logicalModel)
		}
		if job.sourceFormat != target.format {
			return nil, shadowReasonDynamoFormat, llmprotocol.NewError(
				llmprotocol.ErrorUnsupportedFeature,
				shadowReasonDynamoFormat,
				"Dynamo nvext requests cannot be translated across wire formats",
				nil,
			)
		}
	}
	client, reason, err := d.connectorFor(job, target)
	if err != nil {
		return nil, reason, err
	}
	body, err := job.encode(job.request, target)
	if err != nil {
		return nil, shadowReasonEncodeFailed, err
	}
	return &preparedShadowCall{target: target, client: client, body: body}, "", nil
}

// connectorFor returns the shared connector for a target and decision bounds,
// creating it on first use. The key includes the router config identity so a
// rebuilt configuration never reuses a connector bound to the old one.
// Certificate verification is skipped only when the operator opted in for
// that decision.
func (d *shadowDispatcher) connectorFor(job *shadowJob, target *shadowTarget) (*connector.Client, string, error) {
	key := fmt.Sprintf("%p|%s|%s|%d|%d|%d|%t",
		job.routerConfig, target.baseURL, target.logicalModel,
		job.cfg.TimeoutSeconds, job.cfg.MaxRetries, job.cfg.MaxResponseBytes, job.cfg.TLSSkipVerify)
	d.mu.Lock()
	defer d.mu.Unlock()
	if client, ok := d.clients[key]; ok {
		return client, "", nil
	}
	authorize, err := shadowAuthorizer(job.routerConfig, target.profile, target.logicalModel)
	if err != nil {
		return nil, shadowReasonCredentialUnresolved, err
	}
	client, err := connector.New(target.baseURL, authorize, shadowConnectorOptions(job.cfg))
	if err != nil {
		return nil, shadowReasonBackendUnresolved, fmt.Errorf("shadow model %q: %w", target.logicalModel, err)
	}
	d.clients[key] = client
	return client, "", nil
}

func shadowConnectorOptions(cfg config.ShadowDispatchPluginConfig) connector.Options {
	options := connector.Options{
		AttemptTimeout:   time.Duration(cfg.TimeoutSeconds) * time.Second,
		MaxRetries:       cfg.MaxRetries,
		MaxRequestBytes:  shadowDispatchMaxRequestBytes,
		MaxResponseBytes: int64(cfg.MaxResponseBytes),
		MaxErrorBytes:    shadowDispatchErrorBodyBytes,
	}
	if cfg.TLSSkipVerify {
		options.TLSConfig = insecureShadowTLSConfig()
	}
	return options
}

// shadowCallHeaders carries the per-request context a shadow copy keeps:
// trace headers, decision header mutations, provider extra headers, and its
// own request identifier. Client headers are never forwarded. Per-request
// headers are filtered once more against the shadow backend's own auth
// header, so a decision mutation can never stand in for the shadow's
// credential; only shadowAuthorizer may set that header.
func shadowCallHeaders(job *shadowJob, target *shadowTarget) map[string]string {
	result := make(map[string]string, len(job.extraHeaders)+2)
	shadowAuthHeader := shadowProfileAuthHeader(target.profile)
	for key, value := range job.extraHeaders {
		if shadowHeaderIsSensitive(key, shadowAuthHeader) {
			continue
		}
		result[key] = value
	}
	if target.profile != nil {
		for key, value := range target.profile.ExtraHeaders {
			result[key] = value
		}
	}
	// prepareShadowCall permits these effective routing inputs only after
	// resolving a same-format Dynamo target. Apply them after static headers so
	// Dynamo's documented header-over-body routing semantics remain intact.
	// Remove every routing input inherited from the independently configured
	// shadow profile or decision. Re-add only the primary request's validated
	// effective state so decision deletes remain deletes on the shadow path.
	for existing := range result {
		if _, ok := canonicalDynamoRoutingHeaderName(existing); ok {
			delete(result, existing)
		}
	}
	for key, value := range job.dynamoHeaders {
		result[key] = value
	}
	result[headers.RequestID] = job.shadowRequestID
	return result
}

// shadowConnectorFailure maps a connector error onto the shadow reason
// taxonomy and reports what the connector learned before giving up.
func shadowConnectorFailure(callCtx, rootCtx context.Context, err error) (reason string, attempts int, status int) {
	var connectorErr *connector.Error
	if !errors.As(err, &connectorErr) {
		return shadowReasonTransportError, 0, 0
	}
	attempts = connectorErr.Attempt
	switch connectorErr.Kind {
	case connector.KindRequest:
		return shadowReasonEncodeFailed, attempts, 0
	case connector.KindAuthorization:
		return shadowReasonCredentialUnresolved, attempts, 0
	case connector.KindStatus:
		return shadowReasonUpstreamStatus, attempts, connectorErr.StatusCode
	case connector.KindRedirect:
		// The connector never follows a redirect, so the prompt and the
		// shadow credential only ever reached the configured origin.
		return shadowReasonRedirectRejected, attempts, connectorErr.StatusCode
	case connector.KindResponse:
		if errors.Is(err, connector.ErrResponseTooLarge) {
			return shadowReasonResponseTooLarge, attempts, 0
		}
	}
	if reason, done := shadowContextFailure(callCtx, rootCtx); done {
		return reason, attempts, 0
	}
	return shadowReasonTransportError, attempts, 0
}

// resolveShadowTarget reuses the primary backend resolution so a shadow can
// only ever reach a backend the operator configured for that model.
func resolveShadowTarget(cfg *config.RouterConfig, model string) (*shadowTarget, error) {
	address, backendName, found, err := cfg.ResolvePrimaryBackendForModel(model)
	if err != nil {
		return nil, fmt.Errorf("resolve backend for shadow model %q: %w", model, err)
	}
	if !found || address == "" {
		return nil, fmt.Errorf("shadow model %q has no configured backend", model)
	}
	endpoint, found := cfg.GetEndpointByName(backendName)
	if !found {
		return nil, fmt.Errorf("shadow model %q resolved unknown backend %q", model, backendName)
	}
	backendType := strings.TrimSpace(endpoint.Type)
	profile, err := cfg.GetProviderProfileForEndpoint(backendName)
	if err != nil {
		return nil, fmt.Errorf("resolve provider profile for shadow model %q: %w", model, err)
	}
	format, err := wireFormatForModel(cfg.GetModelAPIFormat(model))
	if err != nil {
		return nil, fmt.Errorf("shadow model %q: %w", model, err)
	}
	endpointPath, endpointQuery, err := splitShadowEndpoint(shadowEndpointPath(profile, format))
	if err != nil {
		return nil, fmt.Errorf("shadow model %q: %w", model, err)
	}
	return &shadowTarget{
		logicalModel:  model,
		backendName:   backendName,
		backendType:   backendType,
		upstreamModel: cfg.ResolveExternalModelID(model, backendName),
		profile:       profile,
		format:        format,
		baseURL:       shadowEndpointScheme(cfg, backendName, profile) + "://" + address,
		path:          endpointPath,
		query:         endpointQuery,
	}, nil
}

// splitShadowEndpoint parses a resolved provider endpoint into the absolute
// path and the query the connector sends separately. A provider profile may
// append a query, as Azure OpenAI does with api-version, and an operator's
// chat_path override is free text, so the value is validated here rather than
// trusted: it must be a path-only reference with no scheme, host, or fragment,
// and any query must parse.
func splitShadowEndpoint(endpoint string) (string, string, error) {
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return "", "", fmt.Errorf("parse endpoint path %q: %w", endpoint, err)
	}
	if parsed.Scheme != "" || parsed.Host != "" || parsed.User != nil || parsed.Opaque != "" {
		return "", "", fmt.Errorf("endpoint path %q must not name a scheme or host", endpoint)
	}
	if parsed.Fragment != "" || parsed.RawFragment != "" || strings.Contains(endpoint, "#") {
		return "", "", fmt.Errorf("endpoint path %q must not carry a fragment", endpoint)
	}
	if !strings.HasPrefix(parsed.Path, "/") {
		return "", "", fmt.Errorf("endpoint path %q must be absolute", endpoint)
	}
	if _, err := url.ParseQuery(parsed.RawQuery); err != nil {
		return "", "", fmt.Errorf("endpoint path %q has an invalid query: %w", endpoint, err)
	}
	return parsed.Path, parsed.RawQuery, nil
}

func shadowEndpointScheme(cfg *config.RouterConfig, backendName string, profile *config.ProviderProfile) string {
	if profile != nil && profile.BaseURL != "" {
		if parsed, err := url.Parse(profile.BaseURL); err == nil && parsed.Scheme != "" {
			return parsed.Scheme
		}
		return "http"
	}
	if endpoint, ok := cfg.GetEndpointByName(backendName); ok && endpoint != nil &&
		strings.EqualFold(strings.TrimSpace(endpoint.Protocol), "https") {
		return "https"
	}
	return "http"
}

// shadowEndpointPath mirrors setProviderRequestPath so the shadow reaches the
// same provider path the primary dispatch would use for that wire format.
func shadowEndpointPath(profile *config.ProviderProfile, format llmprotocol.WireFormat) string {
	path := requestWirePath(format)
	if profile == nil {
		return path
	}
	if configured, err := profile.ResolveCreatePath(requestWireProtocol(format)); err == nil && configured != "" {
		return configured
	}
	return path
}

// shadowAuthorizer consults only the static router configuration, read at
// call time. Client credentials never travel with a shadow copy, and a backend
// without a configured key is simply called without one; if it needs a key the
// call fails visibly with upstream_status instead of a per-request auth error.
func shadowAuthorizer(
	cfg *config.RouterConfig,
	profile *config.ProviderProfile,
	model string,
) (func(context.Context, *http.Request) error, error) {
	provider, providerAuth, err := resolveProviderAuth(profile)
	if err != nil {
		return nil, err
	}
	if cfg == nil {
		return nil, nil
	}
	return func(_ context.Context, request *http.Request) error {
		accessKey := authz.NewStaticConfigProvider(cfg).GetKey(provider, model, nil)
		if accessKey == "" {
			return nil
		}
		if providerAuth.Prefix != "" {
			accessKey = providerAuth.Prefix + " " + accessKey
		}
		request.Header.Set(providerAuth.Header, accessKey)
		return nil
	}, nil
}

func shadowContextFailure(callCtx, rootCtx context.Context) (string, bool) {
	if rootCtx.Err() != nil {
		return shadowReasonRouterClosing, true
	}
	if callCtx.Err() != nil {
		return shadowReasonTimeout, true
	}
	return "", false
}

func shadowTokenCount(count llmprotocol.TokenCount) int64 {
	if count.Value == nil {
		return 0
	}
	return *count.Value
}
