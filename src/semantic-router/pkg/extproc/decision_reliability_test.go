package extproc

import (
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func reliableDecision(r config.DecisionReliability) *config.Decision {
	return &config.Decision{Name: "slow_route", Reliability: &r}
}

func headerValues(options []*core.HeaderValueOption) map[string]string {
	out := map[string]string{}
	for _, option := range options {
		out[option.GetHeader().GetKey()] = string(option.GetHeader().GetRawValue())
	}
	return out
}

func TestCallReliabilityConvertsTheDecisionBlock(t *testing.T) {
	require.Nil(t, callReliability(nil))
	require.Nil(t, callReliability(&config.Decision{Name: "plain"}))

	zero := 0
	got := callReliability(reliableDecision(config.DecisionReliability{
		TotalTimeout: "90s", PerTryTimeout: "0s", FirstByteTimeout: "5s", RetryCount: &zero,
		RetryOn: " reset, 5xx ,", RetriableStatusCodes: []int{429}, RetryAfterMax: "10s",
	}))
	require.Equal(t, 90*time.Second, *got.TotalTimeout)
	require.Equal(t, time.Duration(0), *got.PerTryTimeout)
	require.Equal(t, 5*time.Second, *got.FirstByteTimeout)
	require.Nil(t, got.IdleTimeout)
	require.Equal(t, 0, *got.RetryCount)
	require.Equal(t, []string{"reset", "5xx"}, got.RetryOn)
	require.Equal(t, []int{429}, got.RetriableStatusCodes)
	require.Equal(t, 10*time.Second, *got.RetryAfterMax)
	require.Nil(t, got.RetryBackOffBase)
}

func TestEnvoyReliabilityHeadersFollowEnvoysPerRequestHeaders(t *testing.T) {
	three := 3
	total, perTry := 90*time.Second, time.Duration(0)
	override := &routing.Reliability{
		TotalTimeout: &total, PerTryTimeout: &perTry, RetryCount: &three, RetriableStatusCodes: []int{429, 503},
	}
	require.Equal(t, map[string]string{
		"x-envoy-upstream-rq-timeout-ms":         "90000",
		"x-envoy-upstream-rq-per-try-timeout-ms": "0",
		"x-envoy-max-retries":                    "3",
		"x-envoy-retriable-status-codes":         "429,503",
	}, headerValues(envoyReliabilityHeaders(override, true)), "a route with retries keeps its own conditions")

	got := headerValues(envoyReliabilityHeaders(override, false))
	require.Equal(t, config.DefaultProviderRetryOn, got["x-envoy-retry-on"],
		"without a route retry policy the count needs conditions to apply to")

	override.RetryOn = []string{"reset", "retriable-status-codes"}
	require.Equal(t, "reset,retriable-status-codes", headerValues(envoyReliabilityHeaders(override, false))["x-envoy-retry-on"])
	require.Empty(t, envoyReliabilityHeaders(&routing.Reliability{IdleTimeout: &total}, false),
		"Envoy has no per-request idle timeout")
}

func TestEnvoyReliabilityHeadersAreTheConfiguredFive(t *testing.T) {
	one, total := 1, time.Second
	full := &routing.Reliability{
		TotalTimeout: &total, PerTryTimeout: &total, RetryCount: &one,
		RetryOn: []string{"reset"}, RetriableStatusCodes: []int{503},
	}
	names := make([]string, 0, len(config.ReliabilityHeaders))
	for name := range headerValues(envoyReliabilityHeaders(full, true)) {
		names = append(names, name)
	}
	require.ElementsMatch(t, config.ReliabilityHeaders, names,
		"the Envoy rules allow exactly config.ReliabilityHeaders")
}

func TestEnvoyReliabilityHeadersRideTheRouteMutation(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{BackendModels: config.BackendModels{
		ModelConfig: map[string]config.ModelParams{
			"model-a": {Reliability: config.ProviderReliability{RetryCount: 2}},
		},
	}}}
	routeTo := func(model string) *ext_proc.HeaderMutation {
		return &ext_proc.HeaderMutation{SetHeaders: []*core.HeaderValueOption{{
			Header: &core.HeaderValue{Key: routing.RouteHeader, RawValue: []byte(model)},
		}}}
	}
	ctx := &RequestContext{VSRSelectedDecision: reliableDecision(config.DecisionReliability{TotalTimeout: "2s", RetryCount: new(int)})}

	buffered := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{
		RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{HeaderMutation: routeTo("model-a")}},
	}}
	router.addEnvoyReliabilityHeaders(buffered, ctx)
	got := headerValues(buffered.GetRequestBody().GetResponse().GetHeaderMutation().GetSetHeaders())
	require.Equal(t, "2000", got["x-envoy-upstream-rq-timeout-ms"])
	require.Equal(t, "0", got["x-envoy-max-retries"])
	require.NotContains(t, got, "x-envoy-retry-on", "model-a's route has its own retry policy")

	hold := &fullDuplexHeaderHold{routeMutation: routeTo("model-b")}
	ctx.fullDuplexHold = hold
	streamed := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{}}}
	router.addEnvoyReliabilityHeaders(streamed, ctx)
	got = headerValues(hold.routeMutation.GetSetHeaders())
	require.Equal(t, "2000", got["x-envoy-upstream-rq-timeout-ms"])
	require.Equal(t, config.DefaultProviderRetryOn, got["x-envoy-retry-on"], "model-b's route has no retry policy")
	require.Nil(t, streamed.GetRequestBody().GetResponse(), "Envoy ignores header mutations on full-duplex body replies")

	ctx.fullDuplexHold = nil
	immediate := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_ImmediateResponse{
		ImmediateResponse: &ext_proc.ImmediateResponse{},
	}}
	router.addEnvoyReliabilityHeaders(immediate, ctx)
	require.Nil(t, immediate.GetImmediateResponse().GetHeaders(), "an immediate answer never goes upstream")

	plain := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{}}}
	router.addEnvoyReliabilityHeaders(plain, &RequestContext{VSRSelectedDecision: &config.Decision{Name: "plain"}})
	require.Nil(t, plain.GetRequestBody().GetResponse(), "a decision without reliability changes nothing")
}

func TestRoutingSessionReportsTheDecisionsReliability(t *testing.T) {
	session := &routingSession{ctx: &RequestContext{VSRSelectedDecision: reliableDecision(config.DecisionReliability{TotalTimeout: "2s"})}}
	got := session.Reliability()
	require.NotNil(t, got)
	require.Equal(t, 2*time.Second, *got.TotalTimeout)
	require.Nil(t, (&routingSession{ctx: &RequestContext{}}).Reliability())
}
