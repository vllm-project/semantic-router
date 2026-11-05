package upstream

import (
	"io"
	"net/http"
	"reflect"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestRewriteDefaultPathMatchesOnlyTheV1Segment(t *testing.T) {
	const prefix = "/compatible-mode/v1"
	tests := map[string]string{
		"/v1":                     prefix,
		"/v1/chat/completions":    prefix + "/chat/completions",
		"/v1?api-version=2":       prefix + "?api-version=2",
		"/v1beta/models":          "/v1beta/models",
		"/v2/chat/completions":    "/v2/chat/completions",
		"/":                       "/",
		"/x/v1/chat/completions/": "/x/v1/chat/completions/",
	}
	for in, want := range tests {
		if got := rewriteDefaultPath(in, prefix); got != want {
			t.Errorf("rewriteDefaultPath(%q) = %q, want %q", in, got, want)
		}
	}
	if got := rewriteDefaultPath("/v1/models", ""); got != "/v1/models" {
		t.Fatalf("an empty prefix rewrote the path to %q", got)
	}
}

func TestEndpointAuthorityOmitsDefaultPorts(t *testing.T) {
	tests := []struct {
		spec EndpointSpec
		want string
	}{
		{EndpointSpec{Scheme: "http", Host: "10.0.0.1", Port: 80}, "10.0.0.1"},
		{EndpointSpec{Scheme: "http", Host: "10.0.0.1", Port: 443}, "10.0.0.1:443"},
		{EndpointSpec{Scheme: "https", Host: "api.example.test", Port: 443}, "api.example.test"},
		{EndpointSpec{Scheme: "https", Host: "api.example.test", Port: 8443}, "api.example.test:8443"},
		{EndpointSpec{Scheme: "http", Host: "::1", Port: 8000}, "[::1]:8000"},
		{EndpointSpec{Scheme: "http", Host: "::1", Port: 80}, "[::1]"},
	}
	for _, tt := range tests {
		if got := tt.spec.Authority(); got != tt.want {
			t.Errorf("Authority(%+v) = %q, want %q", tt.spec, got, tt.want)
		}
	}
}

// A client request can carry any header the registry keeps from providers:
// the Router's control headers such as x-vsr-debug, internal context,
// identity and secrets. None reaches the provider.
func TestOutboundDropsEveryHeaderKeptFromProviders(t *testing.T) {
	kept := headers.KeptFromProviders()
	if !slices.Contains(kept, headers.VSRDebug) || !slices.Contains(kept, headers.VSRSkipProcessing) {
		t.Fatalf("KeptFromProviders() = %v, want the Router's control headers", kept)
	}
	header := http.Header{"x-request-id": {"r-1"}, "x-session-id": {"s-1"}}
	for _, name := range kept {
		header.Set(name, "client-"+name)
	}
	spec := &ClusterSpec{Name: "m"}
	ep := &EndpointSpec{Name: "e", Scheme: "http", Host: "10.0.0.1", Port: 8000, Weight: 1}
	routed, err := outbound(t.Context(), &Request{Method: "POST", Path: "/v1/chat/completions", Header: header}, spec, ep, false)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range kept {
		if value := routed.Header.Get(name); value != "" {
			t.Errorf("%s = %q reaches the provider", name, value)
		}
	}
	if routed.Header.Get("x-request-id") != "r-1" || routed.Header.Get("x-session-id") != "s-1" {
		t.Fatalf("headers registered to reach providers were dropped: %v", routed.Header)
	}
}

func TestOutboundAppliesTheTemplateRouteRules(t *testing.T) {
	spec := &ClusterSpec{
		Name:       "m",
		PathPrefix: "/compatible-mode/v1",
		RouteHeaders: []Header{
			{Name: "X-Tenant", Value: "operator"},
			{Name: "x-authz-user-id", Value: "route-cannot-add-internal"},
		},
		RemoveHeaders: []string{"x-jwt-sub"},
	}
	ep := &EndpointSpec{Name: "e", Scheme: "http", Host: "10.0.0.1", Port: 8000, Weight: 1}
	req := &Request{
		Method: "POST",
		Path:   "/v1/chat/completions?x=%20y",
		Header: http.Header{
			"x-tenant":             {"client"},
			"authorization":        {"Bearer provider-key"},
			"x-vsr-looper-request": {"true"},
			"x-vsr-looper-secret":  {"s"},
			"x-authz-user-groups":  {"admins"},
			"x-authz-team-id":      {"team-a"},
			"x-authz-tenant-id":    {"tenant-a"},
			"x-jwt-sub":            {"alice"},
			":authority":           {"client.example"},
			"host":                 {"client.example"},
			"content-length":       {"999"},
			"connection":           {"keep-alive, x-hop"},
			"x-hop":                {"1"},
			"te":                   {"trailers"},
		},
		Body: []byte(`{"model":"m"}`),
	}

	routed, err := outbound(t.Context(), req, spec, ep, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := routed.URL.RequestURI(); got != "/v1/chat/completions?x=%20y" {
		t.Fatalf("routed request URI = %q, want the path unchanged", got)
	}
	if routed.Host != "10.0.0.1:8000" || routed.URL.Host != "10.0.0.1:8000" {
		t.Fatalf("host = %q, URL host = %q", routed.Host, routed.URL.Host)
	}
	wantHeader := http.Header{
		"X-Tenant":      {"operator"},
		"Authorization": {"Bearer provider-key"},
		"User-Agent":    nil,
	}
	if !reflect.DeepEqual(routed.Header, wantHeader) {
		t.Fatalf("header = %v, want %v", routed.Header, wantHeader)
	}
	body, _ := io.ReadAll(routed.Body)
	if string(body) != `{"model":"m"}` || routed.ContentLength != int64(len(body)) || routed.GetBody == nil {
		t.Fatalf("body = %q, length %d", body, routed.ContentLength)
	}

	defaulted, err := outbound(t.Context(), req, spec, ep, true)
	if err != nil {
		t.Fatal(err)
	}
	if got := defaulted.URL.RequestURI(); got != "/compatible-mode/v1/chat/completions?x=%20y" {
		t.Fatalf("default route request URI = %q", got)
	}
}

func TestOutboundKeepsAnExplicitUserAgent(t *testing.T) {
	ep := &EndpointSpec{Name: "e", Scheme: "http", Host: "10.0.0.1", Port: 80, Weight: 1}
	req := &Request{Method: "GET", Path: "//double/slash", Header: http.Header{"User-Agent": {"client/1.0"}}}
	out, err := outbound(t.Context(), req, &ClusterSpec{Name: "m"}, ep, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := out.Header.Get("User-Agent"); got != "client/1.0" {
		t.Fatalf("User-Agent = %q", got)
	}
	if got := out.URL.RequestURI(); got != "//double/slash" {
		t.Fatalf("request URI = %q, want //double/slash", got)
	}
	if out.Body != nil || out.ContentLength != 0 {
		t.Fatalf("an empty body was sent: %v %d", out.Body, out.ContentLength)
	}
}

func TestRequestValidationRejectsWhatCannotBeSent(t *testing.T) {
	tests := map[string]*Request{
		"nil request":      nil,
		"empty method":     {Method: "", Path: "/v1"},
		"relative path":    {Method: "POST", Path: "v1/chat"},
		"space in path":    {Method: "POST", Path: "/v1/chat completions"},
		"bad header name":  {Method: "POST", Path: "/v1", Header: http.Header{"bad name": {"x"}}},
		"bad header value": {Method: "POST", Path: "/v1", Header: http.Header{"X-A": {"a\r\nb"}}},
	}
	for name, req := range tests {
		if err := req.validate(); err == nil || err.Kind != KindInvalidRequest {
			t.Errorf("%s: validate() = %v, want %s", name, err, KindInvalidRequest)
		}
	}
	ok := &Request{Method: "POST", Path: "/v1/chat/completions", Header: http.Header{":path": {"/x"}}}
	if err := ok.validate(); err != nil {
		t.Fatalf("valid request rejected: %v", err)
	}
}

func TestRemoveHopByHopCleansResponses(t *testing.T) {
	h := http.Header{
		"Connection":   {"close, X-Foo"},
		"X-Foo":        {"bar"},
		"Keep-Alive":   {"timeout=5"},
		"Content-Type": {"text/event-stream"},
	}
	removeHopByHop(h)
	if !reflect.DeepEqual(h, http.Header{"Content-Type": {"text/event-stream"}}) {
		t.Fatalf("header = %v", h)
	}
}
