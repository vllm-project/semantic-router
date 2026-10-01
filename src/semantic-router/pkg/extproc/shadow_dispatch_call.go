package extproc

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

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
	target, err := resolveShadowTarget(job.routerConfig, job.model)
	if err != nil {
		return nil, shadowReasonBackendUnresolved, err
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
	authorize, err := configuredProviderAuthorizer(job.routerConfig, target.profile, target.logicalModel)
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
// credential; only configuredProviderAuthorizer may set that header.
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
	profile, err := cfg.GetProviderProfileForEndpoint(backendName)
	if err != nil {
		return nil, fmt.Errorf("resolve provider profile for shadow model %q: %w", model, err)
	}
	format, err := wireFormatForModel(cfg.GetModelAPIFormat(model))
	if err != nil {
		return nil, fmt.Errorf("shadow model %q: %w", model, err)
	}
	endpointPath, endpointQuery, err := splitProviderEndpoint(providerEndpointPath(profile, format))
	if err != nil {
		return nil, fmt.Errorf("shadow model %q: %w", model, err)
	}
	return &shadowTarget{
		logicalModel:  model,
		backendName:   backendName,
		upstreamModel: cfg.ResolveExternalModelID(model, backendName),
		profile:       profile,
		format:        format,
		baseURL:       providerEndpointScheme(cfg, backendName, profile) + "://" + address,
		path:          endpointPath,
		query:         endpointQuery,
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
