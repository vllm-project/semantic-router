package extproc

import (
	"fmt"
	"strings"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agenticfacts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func newRouterWithAgenticFacts(cfg config.AgenticFactsConfig) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterOptions: config.RouterOptions{
				AgenticFacts: cfg,
			},
		},
	}
}

func newAgenticFactsRequestHeaders(method, path string, extra map[string]string) *ext_proc.ProcessingRequest_RequestHeaders {
	values := []*core.HeaderValue{
		{Key: ":method", Value: method},
		{Key: ":path", Value: path},
	}
	for k, v := range extra {
		values = append(values, &core.HeaderValue{Key: k, Value: v})
	}
	return &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: values},
		},
	}
}

func validAgenticFactsEnvelopeJSON(t *testing.T) string {
	t.Helper()
	expires := time.Now().Add(30 * time.Second).UTC().Format(time.RFC3339)
	return fmt.Sprintf(`{"version":"1","delegated_role":"researcher","expires_at":%q}`, expires)
}

func containsHeaderCI(names []string, want string) bool {
	for _, n := range names {
		if strings.EqualFold(n, want) {
			return true
		}
	}
	return false
}

func defaultCarrierHeader() string {
	return config.AgenticFactsConfig{}.GetCarrierHeader()
}

func defaultTrustHeader() string {
	return config.AgenticFactsTrustConfig{}.GetMarkerHeader()
}

func defaultTrustValue() string {
	return config.AgenticFactsTrustConfig{}.GetMarkerValue()
}

// --- agenticFactsEnabled ---

func TestAgenticFactsEnabledHelper(t *testing.T) {
	tests := []struct {
		name   string
		router *OpenAIRouter
		want   bool
	}{
		{name: "nil router", router: nil, want: false},
		{name: "router without config", router: &OpenAIRouter{}, want: false},
		{name: "config with zero-value AgenticFacts", router: &OpenAIRouter{Config: &config.RouterConfig{}}, want: false},
		{name: "operator enabled the contract", router: newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true}), want: true},
		{name: "operator explicitly disabled the contract", router: newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: false}), want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.router.agenticFactsEnabled(); got != tt.want {
				t.Fatalf("agenticFactsEnabled()=%v, want %v", got, tt.want)
			}
		})
	}
}

// --- agenticFactsHeaderNames ---

func TestAgenticFactsHeaderNamesHelper(t *testing.T) {
	t.Run("disabled returns nil", func(t *testing.T) {
		router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: false})
		if got := router.agenticFactsHeaderNames(); got != nil {
			t.Fatalf("expected nil when disabled, got %v", got)
		}
	})

	t.Run("enabled with defaults returns the default header names", func(t *testing.T) {
		router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true})
		got := router.agenticFactsHeaderNames()
		if !containsHeaderCI(got, defaultCarrierHeader()) || !containsHeaderCI(got, defaultTrustHeader()) {
			t.Fatalf("expected default carrier and trust headers, got %v", got)
		}
	})

	t.Run("enabled with overrides returns the configured header names", func(t *testing.T) {
		router := newRouterWithAgenticFacts(config.AgenticFactsConfig{
			Enabled:       true,
			CarrierHeader: "x-custom-carrier",
			Trust:         config.AgenticFactsTrustConfig{MarkerHeader: "x-custom-trust"},
		})
		got := router.agenticFactsHeaderNames()
		if !containsHeaderCI(got, "x-custom-carrier") || !containsHeaderCI(got, "x-custom-trust") {
			t.Fatalf("expected configured header names, got %v", got)
		}
	})
}

// --- agenticFactsBoundsFromConfig ---

func TestAgenticFactsBoundsFromConfigClockSkewSentinel(t *testing.T) {
	t.Run("unset clock skew converts to a negative sentinel", func(t *testing.T) {
		bounds := agenticFactsBoundsFromConfig(config.AgenticFactsBoundsConfig{})
		if bounds.ClockSkew >= 0 {
			t.Fatalf("expected a negative sentinel so Bounds.withDefaults substitutes the package default, got %v", bounds.ClockSkew)
		}
	})

	t.Run("explicit zero clock skew is preserved as zero", func(t *testing.T) {
		bounds := agenticFactsBoundsFromConfig(config.AgenticFactsBoundsConfig{ClockSkew: "0s"})
		if bounds.ClockSkew != 0 {
			t.Fatalf("expected an explicit 0s to convert to exactly zero, got %v", bounds.ClockSkew)
		}
	})

	t.Run("explicit positive clock skew is preserved", func(t *testing.T) {
		bounds := agenticFactsBoundsFromConfig(config.AgenticFactsBoundsConfig{ClockSkew: "10s"})
		if bounds.ClockSkew != 10*time.Second {
			t.Fatalf("expected 10s, got %v", bounds.ClockSkew)
		}
	})
}

func TestAgenticFactsBoundsFromConfigMaxLifetime(t *testing.T) {
	t.Run("unset max lifetime converts to zero", func(t *testing.T) {
		bounds := agenticFactsBoundsFromConfig(config.AgenticFactsBoundsConfig{})
		if bounds.MaxLifetime != 0 {
			t.Fatalf("expected zero so Bounds.withDefaults substitutes the package default, got %v", bounds.MaxLifetime)
		}
	})

	t.Run("explicit max lifetime is preserved", func(t *testing.T) {
		bounds := agenticFactsBoundsFromConfig(config.AgenticFactsBoundsConfig{MaxLifetime: "2m"})
		if bounds.MaxLifetime != 2*time.Minute {
			t.Fatalf("expected 2m, got %v", bounds.MaxLifetime)
		}
	})
}

// --- ingestAgenticFacts ---

func TestIngestAgenticFactsNilContext(t *testing.T) {
	router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true})
	router.ingestAgenticFacts(nil) // must not panic
}

func TestIngestAgenticFactsDisabledIsNoOp(t *testing.T) {
	router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: false})
	ctx := &RequestContext{Headers: map[string]string{
		defaultCarrierHeader(): validAgenticFactsEnvelopeJSON(t),
		defaultTrustHeader():   defaultTrustValue(),
	}}

	router.ingestAgenticFacts(ctx)

	if ctx.AgenticFacts.HasFacts() || ctx.AgenticFacts.Rejected() {
		t.Fatalf("expected zero-value Result when disabled, got %+v", ctx.AgenticFacts)
	}
	if _, ok := ctx.Headers[defaultCarrierHeader()]; !ok {
		t.Fatal("expected carrier header to remain untouched when disabled")
	}
	if _, ok := ctx.Headers[defaultTrustHeader()]; !ok {
		t.Fatal("expected trust header to remain untouched when disabled")
	}
}

func TestIngestAgenticFactsUntrustedRejectsWithoutParsing(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
	}{
		{
			name: "trust marker absent",
			headers: map[string]string{
				defaultCarrierHeader(): "this is not even valid json",
			},
		},
		{
			name: "trust marker present but wrong value",
			headers: map[string]string{
				defaultCarrierHeader(): "this is not even valid json",
				defaultTrustHeader():   "0",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true})
			ctx := &RequestContext{Headers: tt.headers}

			router.ingestAgenticFacts(ctx)

			if !ctx.AgenticFacts.Rejected() {
				t.Fatal("expected the envelope to be rejected as untrusted")
			}
			if ctx.AgenticFacts.HasFacts() {
				t.Fatal("expected no accepted facts on the untrusted path")
			}
			if len(ctx.AgenticFacts.Rejections) != 1 || ctx.AgenticFacts.Rejections[0].Reason != agenticfacts.ReasonUntrusted {
				t.Fatalf("expected exactly one ReasonUntrusted rejection, got %+v", ctx.AgenticFacts.Rejections)
			}
		})
	}
}

func TestIngestAgenticFactsTrustedNoCarrierIsEmptyResult(t *testing.T) {
	router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true})
	ctx := &RequestContext{Headers: map[string]string{
		defaultTrustHeader(): defaultTrustValue(),
	}}

	router.ingestAgenticFacts(ctx)

	if ctx.AgenticFacts.HasFacts() || ctx.AgenticFacts.Rejected() {
		t.Fatalf("expected zero-value Result when no envelope was presented, got %+v", ctx.AgenticFacts)
	}
}

// An ordinary request carries neither header. It must not be recorded as a
// rejected envelope, or Replay would report almost every request as one.
func TestIngestAgenticFactsUntrustedNoCarrierIsEmptyResult(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
	}{
		{
			name:    "no headers at all",
			headers: map[string]string{},
		},
		{
			name: "trust marker with wrong value",
			headers: map[string]string{
				defaultTrustHeader(): "0",
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true})
			ctx := &RequestContext{Headers: tt.headers}

			router.ingestAgenticFacts(ctx)

			if ctx.AgenticFacts.HasFacts() || ctx.AgenticFacts.Rejected() {
				t.Fatalf("expected zero-value Result for a request with no envelope, got %+v", ctx.AgenticFacts)
			}
		})
	}
}

func TestIngestAgenticFactsTrustedMalformedEnvelope(t *testing.T) {
	router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true})
	ctx := &RequestContext{Headers: map[string]string{
		defaultCarrierHeader(): "not json at all",
		defaultTrustHeader():   defaultTrustValue(),
	}}

	router.ingestAgenticFacts(ctx)

	if !ctx.AgenticFacts.Rejected() {
		t.Fatal("expected a malformed envelope to be rejected")
	}
	if ctx.AgenticFacts.HasFacts() {
		t.Fatal("expected no accepted facts for a malformed envelope")
	}
}

func TestIngestAgenticFactsTrustedValidEnvelope(t *testing.T) {
	router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true})
	ctx := &RequestContext{Headers: map[string]string{
		defaultCarrierHeader(): validAgenticFactsEnvelopeJSON(t),
		defaultTrustHeader():   defaultTrustValue(),
	}}

	router.ingestAgenticFacts(ctx)

	if ctx.AgenticFacts.Rejected() {
		t.Fatalf("expected the envelope to be accepted, got rejections %+v", ctx.AgenticFacts.Rejections)
	}
	if !ctx.AgenticFacts.HasFacts() {
		t.Fatal("expected accepted facts for a valid envelope")
	}
	if ctx.AgenticFacts.Accepted.DelegatedRole != "researcher" {
		t.Fatalf("expected delegated_role to round-trip, got %q", ctx.AgenticFacts.Accepted.DelegatedRole)
	}
}

func TestIngestAgenticFactsScrubsHeadersFromContext(t *testing.T) {
	router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true})
	ctx := &RequestContext{Headers: map[string]string{
		defaultCarrierHeader(): validAgenticFactsEnvelopeJSON(t),
		defaultTrustHeader():   defaultTrustValue(),
		"authorization":        "Bearer sk-test",
	}}

	router.ingestAgenticFacts(ctx)

	if _, ok := ctx.Headers[defaultCarrierHeader()]; ok {
		t.Fatal("expected the carrier header to be scrubbed from ctx.Headers")
	}
	if _, ok := ctx.Headers[defaultTrustHeader()]; ok {
		t.Fatal("expected the trust marker header to be scrubbed from ctx.Headers")
	}
	if _, ok := ctx.Headers["authorization"]; !ok {
		t.Fatal("expected unrelated headers to survive untouched")
	}
}

func TestIngestAgenticFactsCustomHeaderNames(t *testing.T) {
	router := newRouterWithAgenticFacts(config.AgenticFactsConfig{
		Enabled:       true,
		CarrierHeader: "x-custom-carrier",
		Trust: config.AgenticFactsTrustConfig{
			MarkerHeader: "x-custom-trust",
			MarkerValue:  "yes",
		},
	})
	ctx := &RequestContext{Headers: map[string]string{
		"x-custom-carrier": validAgenticFactsEnvelopeJSON(t),
		"x-custom-trust":   "yes",
	}}

	router.ingestAgenticFacts(ctx)

	if ctx.AgenticFacts.Rejected() {
		t.Fatalf("expected acceptance using configured header names, got rejections %+v", ctx.AgenticFacts.Rejections)
	}
	if !ctx.AgenticFacts.HasFacts() {
		t.Fatal("expected accepted facts using configured header names")
	}
}

// --- end-to-end via handleRequestHeaders ---

func TestHandleRequestHeadersAgenticFactsStripsHeadersOnNormalPath(t *testing.T) {
	router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: true})
	ctx := &RequestContext{Headers: make(map[string]string)}

	request := newAgenticFactsRequestHeaders("POST", "/v1/chat/completions", map[string]string{
		defaultCarrierHeader(): validAgenticFactsEnvelopeJSON(t),
		defaultTrustHeader():   defaultTrustValue(),
	})

	response, err := router.handleRequestHeaders(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}
	if !ctx.AgenticFacts.HasFacts() {
		t.Fatalf("expected accepted facts, got %+v", ctx.AgenticFacts)
	}
	if _, ok := ctx.Headers[defaultCarrierHeader()]; ok {
		t.Fatal("expected carrier header to be scrubbed from ctx.Headers")
	}

	headersResp := response.GetRequestHeaders()
	if headersResp == nil {
		t.Fatal("expected a request headers response")
	}
	removals := headersResp.Response.GetHeaderMutation().GetRemoveHeaders()
	if !containsHeaderCI(removals, defaultCarrierHeader()) {
		t.Fatalf("expected the carrier header in RemoveHeaders, got %v", removals)
	}
	if !containsHeaderCI(removals, defaultTrustHeader()) {
		t.Fatalf("expected the trust header in RemoveHeaders, got %v", removals)
	}
}

func TestHandleRequestHeadersAgenticFactsStripsHeadersOnSkipProcessingPath(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			RouterOptions: config.RouterOptions{
				SkipProcessing: config.SkipProcessingConfig{Enabled: true},
				AgenticFacts:   config.AgenticFactsConfig{Enabled: true},
			},
		},
	}
	ctx := &RequestContext{Headers: make(map[string]string)}

	request := newAgenticFactsRequestHeaders("POST", "/v1/chat/completions", map[string]string{
		"x-vsr-skip-processing": "true",
		defaultCarrierHeader():  validAgenticFactsEnvelopeJSON(t),
		defaultTrustHeader():    defaultTrustValue(),
	})

	response, err := router.handleRequestHeaders(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}
	if !ctx.SkipProcessing {
		t.Fatal("expected SkipProcessing to be set")
	}
	// ingestAgenticFacts runs unconditionally before the SkipProcessing branch,
	// so diagnostics are still populated even though selection never consumes
	// them on this path.
	if !ctx.AgenticFacts.HasFacts() {
		t.Fatalf("expected accepted facts even on the skip-processing path, got %+v", ctx.AgenticFacts)
	}

	headersResp := response.GetRequestHeaders()
	if headersResp == nil {
		t.Fatal("expected a request headers response")
	}
	removals := headersResp.Response.GetHeaderMutation().GetRemoveHeaders()
	if !containsHeaderCI(removals, defaultCarrierHeader()) {
		t.Fatalf("expected the carrier header in RemoveHeaders on the skip-processing path too, got %v", removals)
	}
	if !containsHeaderCI(removals, defaultTrustHeader()) {
		t.Fatalf("expected the trust header in RemoveHeaders on the skip-processing path too, got %v", removals)
	}
}

func TestHandleRequestHeadersAgenticFactsDisabledLeavesHeadersUntouched(t *testing.T) {
	router := newRouterWithAgenticFacts(config.AgenticFactsConfig{Enabled: false})
	ctx := &RequestContext{Headers: make(map[string]string)}

	request := newAgenticFactsRequestHeaders("POST", "/v1/chat/completions", map[string]string{
		defaultCarrierHeader(): validAgenticFactsEnvelopeJSON(t),
		defaultTrustHeader():   defaultTrustValue(),
	})

	response, err := router.handleRequestHeaders(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders failed: %v", err)
	}
	if ctx.AgenticFacts.HasFacts() || ctx.AgenticFacts.Rejected() {
		t.Fatalf("expected zero-value Result when disabled, got %+v", ctx.AgenticFacts)
	}
	if _, ok := ctx.Headers[defaultCarrierHeader()]; !ok {
		t.Fatal("expected carrier header to survive untouched when the contract is disabled")
	}

	headersResp := response.GetRequestHeaders()
	if headersResp == nil {
		t.Fatal("expected a request headers response")
	}
	removals := headersResp.Response.GetHeaderMutation().GetRemoveHeaders()
	if containsHeaderCI(removals, defaultCarrierHeader()) {
		t.Fatalf("did not expect the carrier header in RemoveHeaders when disabled, got %v", removals)
	}
}
