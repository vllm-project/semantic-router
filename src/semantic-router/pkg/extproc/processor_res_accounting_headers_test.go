package extproc

import (
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func accountingHeaderValue(mutation *ext_proc.HeaderMutation, key string) (string, bool) {
	for _, option := range mutation.GetSetHeaders() {
		if option.GetHeader().GetKey() == key {
			return string(option.GetHeader().GetRawValue()), true
		}
	}
	return "", false
}

func TestRoutingLatencyKeepsSubMillisecondPrecision(t *testing.T) {
	ctx := debugRequestContext()
	ctx.VSRSelectedModel, ctx.RoutingLatency = "local", 412*time.Microsecond

	if got := traceOf(t, buildResponseHeaderMutation(ctx).GetSetHeaders())[headers.TraceRoutingLatencyMs]; got != "0.412" {
		t.Fatalf("routing_latency_ms = %q, want 0.412", got)
	}
}

func TestRoutingLatencyOmittedWithoutRoutingOrOnCacheHit(t *testing.T) {
	notRouted := debugRequestContext()
	notRouted.VSRSelectedModel = "local"
	if got, ok := traceOf(t, buildResponseHeaderMutation(notRouted).GetSetHeaders())[headers.TraceRoutingLatencyMs]; ok {
		t.Fatalf("not routed: routing_latency_ms = %s, want it omitted", got)
	}
	cacheHit := debugRequestContext()
	cacheHit.VSRSelectedModel, cacheHit.RoutingLatency, cacheHit.VSRCacheHit = "local", time.Millisecond, true
	if mutation := buildResponseHeaderMutation(cacheHit); mutation != nil {
		t.Fatalf("cache hit: provider response headers = %v, want none", mutation.GetSetHeaders())
	}
}

func TestResponseCostHeadersReportPricedCost(t *testing.T) {
	ctx := &RequestContext{RequestCost: 0.000054, RequestCostCurrency: "USD", RequestCostPriced: true}
	response := buildResponseBodyContinueResponse(nil, nil)

	addResponseCostHeaders(ctx, response)

	mutation := response.GetResponseBody().GetResponse().GetHeaderMutation()
	if got, _ := accountingHeaderValue(mutation, headers.VSRCost); got != "0.000054" {
		t.Fatalf("%s = %q, want 0.000054", headers.VSRCost, got)
	}
	if got, _ := accountingHeaderValue(mutation, headers.VSRCostCurrency); got != "USD" {
		t.Fatalf("%s = %q, want USD", headers.VSRCostCurrency, got)
	}
}

func TestResponseCostHeadersReportZeroForFreeModel(t *testing.T) {
	ctx := &RequestContext{RequestCostCurrency: "USD", RequestCostPriced: true}
	response := buildResponseBodyContinueResponse(nil, nil)

	addResponseCostHeaders(ctx, response)

	mutation := response.GetResponseBody().GetResponse().GetHeaderMutation()
	if got, ok := accountingHeaderValue(mutation, headers.VSRCost); !ok || got != "0" {
		t.Fatalf("%s = %q (present %v), want 0", headers.VSRCost, got, ok)
	}
}

func TestResponseCostHeadersOmittedWithoutPricing(t *testing.T) {
	response := buildResponseBodyContinueResponse(nil, nil)

	addResponseCostHeaders(&RequestContext{}, response)

	mutation := response.GetResponseBody().GetResponse().GetHeaderMutation()
	if _, ok := accountingHeaderValue(mutation, headers.VSRCost); ok {
		t.Fatalf("%s should be omitted when the served model has no pricing", headers.VSRCost)
	}
}
