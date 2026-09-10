package extproc

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func newRequestHeaders(method, path string) *ext_proc.ProcessingRequest_RequestHeaders {
	return &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":method", Value: method},
				{Key: ":path", Value: path},
			}},
		},
	}
}

func TestDetectSourceFormat(t *testing.T) {
	tests := []struct {
		path string
		want llmprotocol.WireFormat
	}{
		{path: "/v1/chat/completions", want: llmprotocol.OpenAIChatV1},
		{path: "/v1/responses", want: llmprotocol.OpenAIResponsesV1},
		{path: "/v1/responses?stream=true", want: llmprotocol.OpenAIResponsesV1},
		{path: "/v1/messages", want: llmprotocol.AnthropicMessagesV1},
		{path: "/v1/messages/count_tokens", want: llmprotocol.AnthropicMessagesV1},
	}
	for _, test := range tests {
		t.Run(test.path, func(t *testing.T) {
			ctx := &RequestContext{}
			detectSourceFormat(test.path, ctx)
			if ctx.SourceFormat != test.want {
				t.Fatalf("source format = %q, want %q", ctx.SourceFormat, test.want)
			}
		})
	}
}

func TestApplyHeaderPassThroughPolicyDropsOnlyTransportHeaders(t *testing.T) {
	ctx := &RequestContext{Headers: map[string]string{
		"host":              "example.test",
		"content-length":    "42",
		"connection":        "keep-alive",
		"transfer-encoding": "chunked",
		"anthropic-version": "2024-10-22",
		"x-application":     "kept",
	}}
	applyHeaderPassThroughPolicy(ctx)
	for _, key := range []string{"host", "content-length", "connection", "transfer-encoding"} {
		if _, found := ctx.Headers[key]; found {
			t.Fatalf("transport header %q was retained", key)
		}
	}
	for _, key := range []string{"anthropic-version", "x-application"} {
		if _, found := ctx.Headers[key]; !found {
			t.Fatalf("application header %q was removed", key)
		}
	}
	applyHeaderPassThroughPolicy(nil)
}

func TestHandleRequestHeadersStripsIdentityHeadersWithoutExternalAuth(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		Authz: config.AuthzConfig{
			Identity: config.IdentityConfig{
				UserIDHeader:     "x-user-id",
				UserGroupsHeader: "x-user-groups",
			},
		},
	}}
	ctx := &RequestContext{Headers: make(map[string]string)}
	request := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":method", Value: "POST"},
				{Key: ":path", Value: "/v1/chat/completions"},
				{Key: "X-Authz-User-Id", Value: "admin"},
				{Key: "x-authz-user-groups", Value: "platform-admins"},
				{Key: "X-User-Id", Value: "alice"},
				{Key: "x-user-groups", Value: "premium"},
				{Key: "x-application", Value: "kept"},
			}},
		},
	}

	response, err := router.handleRequestHeaders(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	for _, name := range []string{
		headers.AuthzUserID,
		headers.AuthzUserGroups,
		"x-user-id",
		"x-user-groups",
	} {
		if _, found := ctx.Headers[name]; found {
			t.Fatalf("untrusted identity header %q remained in semantic headers: %#v", name, ctx.Headers)
		}
	}
	if got := ctx.Headers["x-application"]; got != "kept" {
		t.Fatalf("unrelated application header = %q, want kept", got)
	}
	if ctx.TrustedIdentity.UserID != "" || len(ctx.TrustedIdentity.Groups) != 0 {
		t.Fatalf("untrusted identity was accepted: %+v", ctx.TrustedIdentity)
	}

	mutation := response.GetRequestHeaders().Response.GetHeaderMutation()
	for _, name := range []string{headers.AuthzUserID, headers.AuthzUserGroups, "x-user-id", "x-user-groups"} {
		if !containsStringForTest(mutation.GetRemoveHeaders(), name) {
			t.Fatalf("request mutation did not remove identity header %q: %#v", name, mutation.GetRemoveHeaders())
		}
	}
}

func TestHandleRequestHeadersPreservesTrustedIdentityHeadersForExternalAuth(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		Authz: config.AuthzConfig{
			Identity: config.IdentityConfig{
				UserIDHeader:     "x-user-id",
				UserGroupsHeader: "x-user-groups",
			},
			Providers: []config.AuthzProviderConfig{{Type: "header-injection"}},
		},
	}}
	ctx := &RequestContext{Headers: make(map[string]string)}
	request := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{
			Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
				{Key: ":method", Value: "POST"},
				{Key: ":path", Value: "/v1/chat/completions"},
				{Key: "x-user-id", Value: "alice"},
				{Key: "x-user-groups", Value: "premium"},
			}},
		},
	}

	response, err := router.handleRequestHeaders(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	for name, want := range map[string]string{
		"x-user-id":     "alice",
		"x-user-groups": "premium",
	} {
		if got := ctx.Headers[name]; got != want {
			t.Fatalf("trusted identity header %q = %q, want %q", name, got, want)
		}
	}
	if got := ctx.TrustedIdentity; got.UserID != "alice" || len(got.Groups) != 1 || got.Groups[0] != "premium" {
		t.Fatalf("trusted identity = %+v", got)
	}

	mutation := response.GetRequestHeaders().Response.GetHeaderMutation()
	for _, name := range []string{"x-user-id", "x-user-groups"} {
		if !containsStringForTest(mutation.GetRemoveHeaders(), name) {
			t.Fatalf("trusted identity header %q was not removed before upstream forwarding: %#v", name, mutation.GetRemoveHeaders())
		}
	}
}

func TestHandleRequestHeadersSkipProcessingStillRemovesIdentityHeaders(t *testing.T) {
	router := newRouterWithSkipProcessingGate(true)
	ctx := &RequestContext{Headers: make(map[string]string)}
	request := newSkipProcessingRequestHeaders("POST", "/v1/chat/completions", "true")
	request.RequestHeaders.Headers.Headers = append(
		request.RequestHeaders.Headers.Headers,
		&core.HeaderValue{Key: headers.AuthzUserID, Value: "admin"},
		&core.HeaderValue{Key: headers.AuthzUserGroups, Value: "platform-admins"},
	)

	response, err := router.handleRequestHeaders(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	if _, found := ctx.Headers[headers.AuthzUserID]; found {
		t.Fatal("identity user header remained after skip-processing capture")
	}
	if _, found := ctx.Headers[headers.AuthzUserGroups]; found {
		t.Fatal("identity groups header remained after skip-processing capture")
	}
	if ctx.TrustedIdentity.UserID != "" || len(ctx.TrustedIdentity.Groups) != 0 {
		t.Fatalf("skip-processing accepted untrusted identity: %+v", ctx.TrustedIdentity)
	}
	removed := response.GetRequestHeaders().Response.GetHeaderMutation().GetRemoveHeaders()
	for _, name := range []string{headers.AuthzUserID, headers.AuthzUserGroups} {
		if !containsStringForTest(removed, name) {
			t.Fatalf("skip-processing mutation did not remove identity header %q: %#v", name, removed)
		}
	}
}

func TestHandleRequestHeadersDerivesConfiguredMixedCaseIdentity(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		Authz: config.AuthzConfig{
			Identity: config.IdentityConfig{
				UserIDHeader:     "X-JWT-Sub",
				UserGroupsHeader: "X-JWT-Groups",
			},
			Providers: []config.AuthzProviderConfig{{Type: "header-injection"}},
		},
	}}
	ctx := &RequestContext{Headers: make(map[string]string)}
	request := newRequestHeaders("POST", "/v1/chat/completions")
	request.RequestHeaders.Headers.Headers = append(request.RequestHeaders.Headers.Headers,
		&core.HeaderValue{Key: "x-jwt-sub", Value: "alice"},
		&core.HeaderValue{Key: "X-JWT-GROUPS", Value: "team-a, team-b"},
		&core.HeaderValue{Key: "X-AUTHZ-TENANT-ID", Value: "tenant-a"},
		&core.HeaderValue{Key: "x-authz-team-id", Value: "team-a"},
	)
	response, err := router.handleRequestHeaders(request, ctx)
	if err != nil {
		t.Fatalf("handleRequestHeaders() error = %v", err)
	}
	want := authz.TrustedIdentity{
		UserID: "alice", Groups: []string{"team-a", "team-b"},
		TenantID: "tenant-a", TeamID: "team-a",
	}
	if ctx.TrustedIdentity.UserID != want.UserID ||
		len(ctx.TrustedIdentity.Groups) != len(want.Groups) ||
		ctx.TrustedIdentity.Groups[0] != want.Groups[0] || ctx.TrustedIdentity.Groups[1] != want.Groups[1] ||
		ctx.TrustedIdentity.TenantID != want.TenantID || ctx.TrustedIdentity.TeamID != want.TeamID {
		t.Fatalf("trusted identity = %+v, want %+v", ctx.TrustedIdentity, want)
	}
	removed := response.GetRequestHeaders().Response.GetHeaderMutation().GetRemoveHeaders()
	for _, name := range []string{
		"X-JWT-Sub", "X-JWT-Groups", headers.AuthzTenantID, headers.AuthzTeamID,
	} {
		if !containsStringForTest(removed, name) {
			t.Fatalf("identity header %q was not scheduled for upstream removal: %#v", name, removed)
		}
	}
}

func TestMaintainedAuthzRBACConfigsDeriveTrustedIdentity(t *testing.T) {
	_, testFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	repoRoot := filepath.Clean(filepath.Join(filepath.Dir(testFile), "..", "..", "..", ".."))

	for _, rel := range []string{
		filepath.Join("e2e", "config", "config.authz-rbac-demo.yaml"),
		filepath.Join("e2e", "config", "config.authz-rbac.yaml"),
	} {
		t.Run(rel, func(t *testing.T) {
			data, err := os.ReadFile(filepath.Join(repoRoot, rel))
			if err != nil {
				t.Fatalf("read maintained RBAC config: %v", err)
			}
			cfg, err := config.ParseYAMLBytes(data)
			if err != nil {
				t.Fatalf("parse maintained RBAC config: %v", err)
			}
			if !cfg.Authz.HasExternalAuthProvider() {
				t.Fatal("maintained RBAC config must declare an external header-injection provider")
			}

			router := &OpenAIRouter{Config: cfg}
			ctx := &RequestContext{Headers: map[string]string{
				cfg.Authz.Identity.GetUserIDHeader():     "alice",
				cfg.Authz.Identity.GetUserGroupsHeader(): "platform-admins, premium-tier",
			}}
			router.deriveTrustedIdentity(ctx)

			if ctx.TrustedIdentity.UserID != "alice" {
				t.Fatalf("trusted user id = %q, want alice", ctx.TrustedIdentity.UserID)
			}
			if len(ctx.TrustedIdentity.Groups) != 2 ||
				ctx.TrustedIdentity.Groups[0] != "platform-admins" ||
				ctx.TrustedIdentity.Groups[1] != "premium-tier" {
				t.Fatalf("trusted groups = %#v, want [platform-admins premium-tier]", ctx.TrustedIdentity.Groups)
			}
		})
	}
}

func TestDeriveTrustedIdentityKeepsSessionOverridePriority(t *testing.T) {
	configuredSession := "x-learning-session"
	router := &OpenAIRouter{Config: &config.RouterConfig{
		RouterLearning: config.RouterLearningConfig{Protection: config.RouterLearningProtectionConfig{
			Identity: config.RouterLearningIdentityConfig{Headers: config.RouterLearningIdentityHeadersConfig{
				Session: &configuredSession,
			}},
		}},
	}}
	ctx := &RequestContext{Headers: map[string]string{
		"X-SESSION-ID":             "operator-session",
		"x-learning-session":       "configured-session",
		"X-CLAUDE-CODE-SESSION-ID": "anthropic-session",
	}}

	router.deriveTrustedIdentity(ctx)
	if got := ctx.TrustedIdentity.SessionID; got != "operator-session" {
		t.Fatalf("session identity = %q, want x-session-id override", got)
	}
}

func TestValidatePublicGenerationEndpoints(t *testing.T) {
	router := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore())}
	tests := []struct {
		method string
		path   string
		status typev3.StatusCode
	}{
		{method: "POST", path: "/v1/chat/completions"},
		{method: "POST", path: "/v1/responses"},
		{method: "POST", path: "/v1/messages"},
		{method: "GET", path: "/v1/messages", status: typev3.StatusCode_MethodNotAllowed},
		{method: "POST", path: "/v1/messages/count_tokens", status: typev3.StatusCode_NotFound},
	}
	for _, test := range tests {
		t.Run(test.method+" "+test.path, func(t *testing.T) {
			response := router.validateRequestHeaders(test.method, test.path)
			if test.status == 0 {
				if response != nil {
					t.Fatalf("valid generation endpoint returned immediate response: %+v", response)
				}
				return
			}
			if response == nil || response.GetImmediateResponse() == nil ||
				response.GetImmediateResponse().GetStatus().GetCode() != test.status {
				t.Fatalf("status = %+v, want %s", response, test.status)
			}
		})
	}
}

func TestResponseObjectPathParsing(t *testing.T) {
	if got := extractResponseIDFromPath("/v1/responses/resp_123?expand=true"); got != "resp_123" {
		t.Fatalf("response id = %q", got)
	}
	if got := extractResponseIDFromInputItemsPath("/v1/responses/resp_123/input_items"); got != "resp_123" {
		t.Fatalf("input-items response id = %q", got)
	}
	if got := extractResponseIDFromPath("/v1/responses/not-a-response"); got != "" {
		t.Fatalf("invalid response id = %q", got)
	}
}
