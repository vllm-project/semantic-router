package config

import (
	"reflect"
	"strings"
	"testing"
)

func routingInputConfig(block string) []byte {
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
    routing_input:
` + block)
}

func TestRoutingInputReachesRuntimeAndExport(t *testing.T) {
	cfg, err := ParseYAMLBytes(routingInputConfig(`
      skip_user_messages_matching:
        - '(?s)^\s*Client context:.*End client context\.\s*$'
      strip_blocks:
        - start: '<recalled>'
          end: '</recalled>'
`))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	want := RoutingInputConfig{
		SkipUserMessagesMatching: []string{`(?s)^\s*Client context:.*End client context\.\s*$`},
		StripBlocks:              []RoutingInputStripBlock{{Start: "<recalled>", End: "</recalled>"}},
	}
	if !reflect.DeepEqual(cfg.RoutingInput, want) {
		t.Fatalf("runtime routing_input = %#v, want %#v", cfg.RoutingInput, want)
	}
	if !cfg.RoutingInput.Enabled() {
		t.Fatal("configured routing_input must report Enabled")
	}
	if exported := CanonicalGlobalFromRouterConfig(cfg).Router.RoutingInput; !reflect.DeepEqual(exported, want) {
		t.Fatalf("exported routing_input = %#v, want %#v", exported, want)
	}
}

func TestRoutingInputIsOffByDefault(t *testing.T) {
	cfg, err := ParseYAMLBytes(routingInputConfig("      {}\n"))
	if err != nil {
		t.Fatalf("ParseYAMLBytes() error = %v", err)
	}
	if cfg.RoutingInput.Enabled() {
		t.Fatalf("empty routing_input must be disabled, got %#v", cfg.RoutingInput)
	}
}

func TestRoutingInputRejectsInvalidSettings(t *testing.T) {
	tooMany := "      skip_user_messages_matching:\n" + strings.Repeat("        - 'x'\n", MaxRoutingInputSkipPatterns+1)
	tests := map[string]struct {
		block string
		want  string
	}{
		"pattern that does not compile": {
			block: "      skip_user_messages_matching: ['(unclosed']\n",
			want:  "skip_user_messages_matching[0]: invalid RE2 pattern",
		},
		"pattern that matches empty text": {
			block: "      skip_user_messages_matching: ['^\\s*$']\n",
			want:  "skip_user_messages_matching[0]: pattern matches empty text",
		},
		"blank pattern": {
			block: "      skip_user_messages_matching: ['  ']\n",
			want:  "skip_user_messages_matching[0]: pattern must not be empty",
		},
		"oversized pattern": {
			block: "      skip_user_messages_matching: ['" + strings.Repeat("a", MaxRoutingInputPatternBytes+1) + "']\n",
			want:  "pattern must be at most 1024 bytes",
		},
		"too many patterns": {
			block: tooMany,
			want:  "at most 32 patterns are allowed",
		},
		"empty start marker": {
			block: "      strip_blocks:\n        - start: ''\n          end: '</m>'\n",
			want:  "strip_blocks[0].start: marker must not be empty",
		},
		"missing end marker": {
			block: "      strip_blocks:\n        - start: '<m>'\n",
			want:  "strip_blocks[0].end: marker must not be empty",
		},
		"oversized marker": {
			block: "      strip_blocks:\n        - start: '<m>'\n          end: '" + strings.Repeat("m", MaxRoutingInputMarkerBytes+1) + "'\n",
			want:  "strip_blocks[0].end: marker must be at most 256 bytes",
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			_, err := ParseYAMLBytes(routingInputConfig(test.block))
			if err == nil || !strings.Contains(err.Error(), "global.router.routing_input") || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("ParseYAMLBytes() error = %v, want %q", err, test.want)
			}
		})
	}
}
