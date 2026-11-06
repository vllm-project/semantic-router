package config

import (
	"reflect"
	"strings"
	"testing"
)

func routerHeadersConfig(block string) []byte {
	return []byte(`
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: cheap
  models:
    - name: cheap
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
routing:
  modelCards:
    - name: cheap
  decisions:
    - name: route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: cheap
global:
  router:
` + block)
}

func TestDebugHeadersAreOffByDefault(t *testing.T) {
	cfg, err := ParseYAMLBytes(routerHeadersConfig("    list_backend_models: false\n"))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if cfg.Headers.AllowsDebugHeaders("", nil) || cfg.Headers.AllowsDebugHeaders("admin", []string{"admins"}) {
		t.Fatalf("debug headers must be off without global.router.headers.debug, got %#v", cfg.Headers)
	}
	if exported := CanonicalGlobalFromRouterConfig(cfg).Router.Headers; !reflect.DeepEqual(exported, RouterHeadersConfig{}) {
		t.Fatalf("exported headers = %#v, want them omitted", exported)
	}
	cfg, err = ParseYAMLBytes(routerHeadersConfig("    headers:\n      debug: none\n"))
	if err != nil {
		t.Fatalf("ParseYAMLBytes(debug: none) error = %v", err)
	}
	if cfg.Headers.Debug != DebugHeadersNone || cfg.Headers.AllowsDebugHeaders("admin", nil) {
		t.Fatalf("debug: none = %#v, want debug headers off", cfg.Headers)
	}
}

func TestDebugHeadersPolicyReachesRuntimeAndExport(t *testing.T) {
	cfg, err := ParseYAMLBytes(routerHeadersConfig(`
    headers:
      debug: authenticated
      debug_principals:
        users: [oncall]
        groups: [sre]
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	want := RouterHeadersConfig{
		Debug:           DebugHeadersAuthenticated,
		DebugPrincipals: DebugPrincipals{Users: []string{"oncall"}, Groups: []string{"sre"}},
	}
	if !reflect.DeepEqual(cfg.Headers, want) {
		t.Fatalf("runtime headers = %#v, want %#v", cfg.Headers, want)
	}
	if exported := CanonicalGlobalFromRouterConfig(cfg).Router.Headers; !reflect.DeepEqual(exported, want) {
		t.Fatalf("exported headers = %#v, want %#v", exported, want)
	}
}

func TestAllowsDebugHeaders(t *testing.T) {
	principals := DebugPrincipals{Users: []string{"oncall"}, Groups: []string{"sre"}}
	tests := []struct {
		name   string
		policy RouterHeadersConfig
		user   string
		groups []string
		want   bool
	}{
		{name: "none", policy: RouterHeadersConfig{Debug: DebugHeadersNone}, user: "oncall", groups: []string{"sre"}},
		{name: "unset", policy: RouterHeadersConfig{DebugPrincipals: principals}, user: "oncall", groups: []string{"sre"}},
		{name: "any anonymous", policy: RouterHeadersConfig{Debug: DebugHeadersAny}, want: true},
		{name: "listed user", policy: RouterHeadersConfig{Debug: DebugHeadersAuthenticated, DebugPrincipals: principals}, user: "oncall", want: true},
		{name: "listed group", policy: RouterHeadersConfig{Debug: DebugHeadersAuthenticated, DebugPrincipals: principals}, user: "alice", groups: []string{"dev", "sre"}, want: true},
		{name: "other user", policy: RouterHeadersConfig{Debug: DebugHeadersAuthenticated, DebugPrincipals: principals}, user: "alice", groups: []string{"dev"}},
		{name: "anonymous", policy: RouterHeadersConfig{Debug: DebugHeadersAuthenticated, DebugPrincipals: principals}},
		{name: "names match exactly", policy: RouterHeadersConfig{Debug: DebugHeadersAuthenticated, DebugPrincipals: principals}, user: "OnCall", groups: []string{"SRE"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.policy.AllowsDebugHeaders(tt.user, tt.groups); got != tt.want {
				t.Fatalf("AllowsDebugHeaders(%q, %v) = %v, want %v", tt.user, tt.groups, got, tt.want)
			}
		})
	}
}

func TestRouterHeadersRejectsInvalidSettings(t *testing.T) {
	tests := map[string]struct {
		block string
		want  string
	}{
		"unknown mode": {
			block: "    headers:\n      debug: everyone\n",
			want:  `global.router.headers.debug: "everyone" is not supported, use one of: none, any, authenticated`,
		},
		"authenticated without principals": {
			block: "    headers:\n      debug: authenticated\n",
			want:  "needs at least one user or group in debug_principals",
		},
		"principals without authenticated": {
			block: "    headers:\n      debug: any\n      debug_principals:\n        users: [oncall]\n",
			want:  "global.router.headers.debug_principals applies only with debug: authenticated",
		},
		"padded principal": {
			block: "    headers:\n      debug: authenticated\n      debug_principals:\n        groups: [' sre']\n",
			want:  `global.router.headers.debug_principals: " sre" must be a name without surrounding spaces`,
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := ParseYAMLBytes(routerHeadersConfig(test.block))
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ParseYAMLBytes() error = %v, want %q", err, test.want)
			}
		})
	}
}
