package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func TestDebugHeadersRequested(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		want    bool
	}{
		{"nil headers", nil, false},
		{"absent", map[string]string{"x-other": "true"}, false},
		{"true", map[string]string{headers.VSRDebug: "true"}, true},
		{"mixed case value", map[string]string{headers.VSRDebug: "True"}, true},
		{"surrounding whitespace", map[string]string{headers.VSRDebug: " true "}, true},
		{"false", map[string]string{headers.VSRDebug: "false"}, false},
		{"empty value", map[string]string{headers.VSRDebug: ""}, false},
		{"case-insensitive key", map[string]string{"X-Vsr-Debug": "true"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: tt.headers}
			if got := debugHeadersRequested(ctx); got != tt.want {
				t.Errorf("debugHeadersRequested() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDebugHeadersRequested_NilContext(t *testing.T) {
	if debugHeadersRequested(nil) {
		t.Error("debugHeadersRequested(nil) = true, want false")
	}
}

func newDebugGateRequestHeaders(extra map[string]string) *ext_proc.ProcessingRequest_RequestHeaders {
	values := []*core.HeaderValue{
		{Key: ":method", Value: "POST"},
		{Key: ":path", Value: "/v1/chat/completions"},
		{Key: headers.VSRDebug, Value: "true"},
	}
	for key, value := range extra {
		values = append(values, &core.HeaderValue{Key: key, Value: value})
	}
	return &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: values}},
	}
}

func debugGateHeader(mutation *ext_proc.HeaderMutation, key string) string {
	for _, option := range mutation.GetSetHeaders() {
		if option.GetHeader().GetKey() == key {
			return string(option.GetHeader().GetRawValue())
		}
	}
	return ""
}

// A client asks for the debug headers with x-vsr-debug: true; whether it gets
// them, and with them the guard rules its request matched, is the operator's
// global.router.headers.debug, never the client's.
func TestDebugHeadersFollowTheOperatorPolicy(t *testing.T) {
	principals := config.DebugPrincipals{Users: []string{"oncall"}, Groups: []string{"sre"}}
	authenticated := config.RouterHeadersConfig{Debug: config.DebugHeadersAuthenticated, DebugPrincipals: principals}
	tests := []struct {
		name     string
		policy   config.RouterHeadersConfig
		identity config.IdentityConfig
		headers  map[string]string
		want     bool
	}{
		{name: "unset refuses every client", headers: map[string]string{headers.AuthzUserID: "oncall"}},
		{name: "none refuses every client", policy: config.RouterHeadersConfig{Debug: config.DebugHeadersNone}},
		{name: "any honors an anonymous client", policy: config.RouterHeadersConfig{Debug: config.DebugHeadersAny}, want: true},
		{name: "authenticated honors a listed user", policy: authenticated, headers: map[string]string{headers.AuthzUserID: "oncall"}, want: true},
		{name: "authenticated honors a listed group", policy: authenticated, headers: map[string]string{headers.AuthzUserID: "alice", headers.AuthzUserGroups: "dev, sre"}, want: true},
		{name: "authenticated refuses other users", policy: authenticated, headers: map[string]string{headers.AuthzUserID: "alice", headers.AuthzUserGroups: "dev"}},
		{name: "authenticated refuses anonymous clients", policy: authenticated},
		{
			name:     "authenticated reads the configured identity headers",
			policy:   authenticated,
			identity: config.IdentityConfig{UserIDHeader: "x-jwt-sub", UserGroupsHeader: "x-jwt-groups"},
			headers:  map[string]string{headers.AuthzUserID: "oncall", "x-jwt-sub": "alice", "x-jwt-groups": "sre"},
			want:     true,
		},
		{
			name:     "authenticated ignores the default names once others are configured",
			policy:   authenticated,
			identity: config.IdentityConfig{UserIDHeader: "x-jwt-sub", UserGroupsHeader: "x-jwt-groups"},
			headers:  map[string]string{headers.AuthzUserID: "oncall", headers.AuthzUserGroups: "sre"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{
				RouterOptions: config.RouterOptions{Headers: tt.policy},
				Authz:         config.AuthzConfig{Identity: tt.identity},
			}}
			ctx := &RequestContext{Headers: make(map[string]string)}
			if _, err := router.handleRequestHeaders(newDebugGateRequestHeaders(tt.headers), ctx); err != nil {
				t.Fatalf("handleRequestHeaders() error = %v", err)
			}
			ctx.VSRSelectedDecisionName = "guarded"
			ctx.VSRMatchedJailbreak = []string{"prompt_injection"}
			mutation := buildResponseHeaderMutation(ctx, true)

			if got := debugGateHeader(mutation, headers.VSRMatchedJailbreak) != ""; got != tt.want {
				t.Fatalf("%s present = %v, want %v", headers.VSRMatchedJailbreak, got, tt.want)
			}
			if got := debugGateHeader(mutation, headers.VSRSelectedDecision); got != "guarded" {
				t.Fatalf("%s = %q, want the default surface unchanged", headers.VSRSelectedDecision, got)
			}
		})
	}
}

func TestGateDebugHeadersWithoutConfigRefuses(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{"X-Vsr-Debug": "true"}}
	(&OpenAIRouter{}).gateDebugHeaders(ctx)
	if debugHeadersRequested(ctx) {
		t.Fatal("a Router without configuration must not honor x-vsr-debug")
	}
}
