package extproc

import (
	"bytes"
	"context"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// shadowDispatchErrorBodyBytes bounds how much of a non-2xx shadow response
// is read for diagnostics. It is never stored.
const shadowDispatchErrorBodyBytes = 8 * 1024

type shadowTarget struct {
	logicalModel  string
	backendName   string
	upstreamModel string
	profile       *config.ProviderProfile
	format        llmprotocol.WireFormat
	endpointURL   string
}

type preparedShadowCall struct {
	target    *shadowTarget
	accessKey string
	body      []byte
}

func (d *shadowDispatcher) call(ctx context.Context, job *shadowJob) shadowResult {
	result := shadowResult{startedAt: d.now()}
	prepared, reason, err := prepareShadowCall(job)
	if err != nil {
		return d.failShadow(result, reason, err)
	}
	result.shadowBackend = prepared.target.backendName
	for attempt := 1; ; attempt++ {
		// Every attempt reports its own status so a retry that fails
		// differently cannot inherit a stale code from an earlier one.
		result.attempts = attempt
		result.statusCode = 0
		result.responseBytes = 0
		retryable, reason, err := d.attemptShadowCall(ctx, job, prepared, &result)
		if err == nil {
			result.verdict = shadowVerdictCompleted
			result.reason = shadowReasonCompleted
			result.finishedAt = d.now()
			return result
		}
		if retryable && attempt <= job.cfg.MaxRetries && ctx.Err() == nil {
			continue
		}
		return d.failShadow(result, reason, err)
	}
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

// prepareShadowCall resolves the shadow backend, its static credential, and
// the encoded body once; every attempt reuses the same approved request.
func prepareShadowCall(job *shadowJob) (*preparedShadowCall, string, error) {
	target, err := resolveShadowTarget(job.routerConfig, job.cfg.Model)
	if err != nil {
		return nil, shadowReasonBackendUnresolved, err
	}
	accessKey, err := resolveShadowCredential(job.routerConfig, target.profile, job.cfg.Model)
	if err != nil {
		return nil, shadowReasonCredentialUnresolved, err
	}
	body, err := job.encode(job.request, target)
	if err != nil {
		return nil, shadowReasonEncodeFailed, err
	}
	return &preparedShadowCall{target: target, accessKey: accessKey, body: body}, "", nil
}

// attemptShadowCall performs one bounded HTTP attempt and fills result with
// whatever it learned. retryable reports whether another attempt may help.
func (d *shadowDispatcher) attemptShadowCall(
	ctx context.Context,
	job *shadowJob,
	prepared *preparedShadowCall,
	result *shadowResult,
) (retryable bool, reason string, err error) {
	resp, err := d.doShadowRequest(ctx, job, prepared)
	if err != nil {
		if reason, done := shadowContextFailure(ctx, d.ctx); done {
			return false, reason, err
		}
		return true, shadowReasonTransportError, err
	}
	defer func() { _ = resp.Body.Close() }()
	result.statusCode = resp.StatusCode
	if resp.StatusCode != http.StatusOK {
		_, _ = httputil.ReadTruncatedBody(resp.Body, shadowDispatchErrorBodyBytes)
		return shadowRetryableStatus(resp.StatusCode), shadowReasonUpstreamStatus,
			fmt.Errorf("upstream status %d", resp.StatusCode)
	}
	respBody, err := httputil.ReadLimitedBody(resp.Body, int64(job.cfg.MaxResponseBytes))
	if err != nil {
		return false, shadowReadFailureReason(ctx, d.ctx, err), err
	}
	result.responseBytes = len(respBody)
	decoded, err := job.engine.TranslateResponse(prepared.target.format, prepared.target.format, respBody, nil)
	if err != nil {
		return false, shadowReasonMalformedResponse, err
	}
	result.stopReason = string(decoded.Response.StopReason)
	result.inputTokens = shadowTokenCount(decoded.Response.Usage.InputTotal)
	result.outputTokens = shadowTokenCount(decoded.Response.Usage.OutputTotal)
	result.text = semanticResponseText(decoded.Response)
	return false, "", nil
}

func shadowReadFailureReason(callCtx, rootCtx context.Context, err error) string {
	if strings.Contains(err.Error(), "exceeds limit") {
		return shadowReasonResponseTooLarge
	}
	if reason, done := shadowContextFailure(callCtx, rootCtx); done {
		return reason
	}
	return shadowReasonTransportError
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
	return &shadowTarget{
		logicalModel:  model,
		backendName:   backendName,
		upstreamModel: cfg.ResolveExternalModelID(model, backendName),
		profile:       profile,
		format:        format,
		endpointURL:   shadowEndpointURL(cfg, backendName, address, profile, format),
	}, nil
}

func shadowEndpointURL(
	cfg *config.RouterConfig,
	backendName string,
	address string,
	profile *config.ProviderProfile,
	format llmprotocol.WireFormat,
) string {
	return shadowEndpointScheme(cfg, backendName, profile) + "://" + address + shadowEndpointPath(profile, format)
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
	if format == llmprotocol.OpenAIChatV1 {
		if configured, err := profile.ResolveChatPath(); err == nil && configured != "" {
			return configured
		}
		return path
	}
	return providerProtocolPath(profile.BaseURL, path)
}

// resolveShadowCredential consults only the static router configuration.
// Client credentials never travel with a shadow copy, and a backend without a
// configured key is simply called without one; if it needs a key the call
// fails visibly with upstream_status instead of a per-request auth error.
func resolveShadowCredential(
	cfg *config.RouterConfig,
	profile *config.ProviderProfile,
	model string,
) (string, error) {
	provider, _, _, err := resolveProviderAuth(profile)
	if err != nil {
		return "", err
	}
	if cfg == nil {
		return "", nil
	}
	return authz.NewStaticConfigProvider(cfg).GetKey(provider, model, nil), nil
}

func (d *shadowDispatcher) doShadowRequest(
	ctx context.Context,
	job *shadowJob,
	prepared *preparedShadowCall,
) (*http.Response, error) {
	target := prepared.target
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, target.endpointURL, bytes.NewReader(prepared.body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	for key, value := range job.extraHeaders {
		httpReq.Header.Set(key, value)
	}
	if target.profile != nil {
		for key, value := range target.profile.ExtraHeaders {
			httpReq.Header.Set(key, value)
		}
	}
	httpReq.Header.Set(headers.RequestID, job.shadowRequestID)
	if prepared.accessKey != "" {
		_, authHeader, authPrefix, err := resolveProviderAuth(target.profile)
		if err != nil {
			return nil, err
		}
		value := prepared.accessKey
		if authPrefix != "" {
			value = authPrefix + " " + prepared.accessKey
		}
		httpReq.Header.Set(authHeader, value)
	}
	return d.clientFor(job.cfg.TLSSkipVerify).Do(httpReq)
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

func shadowRetryableStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway,
		http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return true
	}
	return false
}

func shadowTokenCount(count llmprotocol.TokenCount) int64 {
	if count.Value == nil {
		return 0
	}
	return *count.Value
}
