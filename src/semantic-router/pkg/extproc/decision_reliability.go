package extproc

import (
	"slices"
	"strconv"
	"strings"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

var _ routing.ReliabilitySession = (*routingSession)(nil)

// Reliability is the planned call's override: the matched decision's
// reliability block.
func (s *routingSession) Reliability() *routing.Reliability {
	return callReliability(s.ctx.VSRSelectedDecision)
}

// callReliability converts a decision's reliability block to the routing
// core's neutral override; nil when the decision sets none.
func callReliability(decision *config.Decision) *routing.Reliability {
	if decision == nil || decision.Reliability == nil {
		return nil
	}
	r := decision.Reliability
	out := &routing.Reliability{
		TotalTimeout:         reliabilityDuration(r.TotalTimeout),
		PerTryTimeout:        reliabilityDuration(r.PerTryTimeout),
		IdleTimeout:          reliabilityDuration(r.IdleTimeout),
		FirstByteTimeout:     reliabilityDuration(r.FirstByteTimeout),
		RetriableStatusCodes: slices.Clone(r.RetriableStatusCodes),
		RetryBackOffBase:     reliabilityDuration(r.RetryBackOffBase),
		RetryBackOffMax:      reliabilityDuration(r.RetryBackOffMax),
		RetryAfterMax:        reliabilityDuration(r.RetryAfterMax),
	}
	if r.RetryCount != nil {
		count := *r.RetryCount
		out.RetryCount = &count
	}
	for _, token := range strings.Split(r.RetryOn, ",") {
		if token = strings.TrimSpace(token); token != "" {
			out.RetryOn = append(out.RetryOn, token)
		}
	}
	return out
}

// reliabilityDuration reads a duration the config loader validated; empty is
// unset.
func reliabilityDuration(value string) *time.Duration {
	d, err := time.ParseDuration(strings.TrimSpace(value))
	if err != nil {
		return nil
	}
	return &d
}

// The per-request headers Envoy's router reads after ext_proc's mutations.
const (
	envoyRouteTimeoutHeader         = "x-envoy-upstream-rq-timeout-ms"
	envoyPerTryTimeoutHeader        = "x-envoy-upstream-rq-per-try-timeout-ms"
	envoyMaxRetriesHeader           = "x-envoy-max-retries"
	envoyRetryOnHeader              = "x-envoy-retry-on"
	envoyRetriableStatusCodesHeader = "x-envoy-retriable-status-codes"
)

// addEnvoyReliabilityHeaders asks Envoy's router to apply the matched
// decision's override to this request. It runs only where the Router answers
// Envoy over gRPC: the native gateway takes the same override from
// routing.Call.Reliability, so its requests never carry these headers.
func (r *OpenAIRouter) addEnvoyReliabilityHeaders(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	override := callReliability(ctx.VSRSelectedDecision)
	if override == nil || response == nil || response.GetImmediateResponse() != nil {
		return
	}
	// The route mutation travels with the held header reply in full-duplex
	// mode, and with the body reply otherwise.
	var mutation *ext_proc.HeaderMutation
	if hold := ctx.fullDuplexHold; hold != nil {
		if hold.routeMutation == nil {
			hold.routeMutation = &ext_proc.HeaderMutation{}
		}
		mutation = hold.routeMutation
	} else if body := response.GetRequestBody(); body != nil {
		if body.Response == nil {
			body.Response = &ext_proc.CommonResponse{}
		}
		if body.Response.HeaderMutation == nil {
			body.Response.HeaderMutation = &ext_proc.HeaderMutation{}
		}
		mutation = body.Response.HeaderMutation
	} else {
		return
	}
	route := routeKeyOf(mutation)
	if route == "" {
		route = ctx.backendModelForCandidate(ctx.VSRSelectedModel)
	}
	mutation.SetHeaders = append(mutation.SetHeaders, envoyReliabilityHeaders(override, r.routeHasRetryPolicy(route))...)
}

// envoyReliabilityHeaders renders an override as Envoy's per-request
// headers: the route and per-try timeouts in milliseconds (0 disables them),
// the retry count, and the retry conditions and statuses Envoy adds to the
// route's. A route without a retry policy has no conditions for the count to
// apply to, so it gets the default ones, as the native merge does. Envoy has
// no per-request header for the other fields; validation keeps them to the
// native gateway.
func envoyReliabilityHeaders(r *routing.Reliability, routeRetries bool) []*core.HeaderValueOption {
	var out []*core.HeaderValueOption
	set := func(name, value string) {
		out = append(out, &core.HeaderValueOption{
			Header:       &core.HeaderValue{Key: name, RawValue: []byte(value)},
			AppendAction: core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD,
		})
	}
	if r.TotalTimeout != nil {
		set(envoyRouteTimeoutHeader, strconv.FormatInt(r.TotalTimeout.Milliseconds(), 10))
	}
	if r.PerTryTimeout != nil {
		set(envoyPerTryTimeoutHeader, strconv.FormatInt(r.PerTryTimeout.Milliseconds(), 10))
	}
	if r.RetryCount != nil {
		set(envoyMaxRetriesHeader, strconv.Itoa(*r.RetryCount))
	}
	retryOn := r.RetryOn
	if len(retryOn) == 0 && !routeRetries && (r.RetryCount != nil || len(r.RetriableStatusCodes) > 0) {
		retryOn = []string{config.DefaultProviderRetryOn}
	}
	if len(retryOn) > 0 {
		set(envoyRetryOnHeader, strings.Join(retryOn, ","))
	}
	if len(r.RetriableStatusCodes) > 0 {
		codes := make([]string, len(r.RetriableStatusCodes))
		for i, code := range r.RetriableStatusCodes {
			codes[i] = strconv.Itoa(code)
		}
		set(envoyRetriableStatusCodesHeader, strings.Join(codes, ","))
	}
	return out
}

// routeHasRetryPolicy reports whether the Envoy route for a provider model
// has a retry policy: the template renders one for retries or a per-try
// timeout.
func (r *OpenAIRouter) routeHasRetryPolicy(model string) bool {
	if r == nil || r.Config == nil {
		return false
	}
	reliability := r.Config.ModelConfig[model].Reliability
	return reliability.RetryCount > 0 || strings.TrimSpace(reliability.PerTryTimeout) != ""
}

// routeKeyOf is the route key a header mutation sets, if any.
func routeKeyOf(mutation *ext_proc.HeaderMutation) string {
	for _, option := range mutation.GetSetHeaders() {
		if strings.EqualFold(option.GetHeader().GetKey(), routing.RouteHeader) {
			return string(option.GetHeader().GetRawValue())
		}
	}
	return ""
}
