package config

import (
	"fmt"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestRequestParamsDefaultValidationAndRoundTrip(t *testing.T) {
	template := `routing:
  decisions:
    - name: bounded
      plugins:
        - type: request_params
          configuration:
            %s
            max_tokens_limit: 8192
`
	for _, field := range []string{"default_max_tokens: 0", "default_max_tokens: -1", "default_max_tokens: 1.5", "default_max_token: 4096"} {
		if _, err := ParseRoutingYAMLBytes([]byte(fmt.Sprintf(template, field))); err == nil {
			t.Errorf("accepted invalid output default: %s", field)
		}
	}
	cfg, err := ParseRoutingYAMLBytes([]byte(fmt.Sprintf(template, "default_max_tokens: 4096")))
	if err != nil {
		t.Fatal(err)
	}
	params := cfg.Decisions[0].GetRequestParamsConfig()
	if params.DefaultMaxTokens == nil || params.DefaultMaxTokens.Value != 4096 || *params.MaxTokensLimit != 8192 {
		t.Fatal("typed payload lost output default")
	}
	encoded, err := yaml.Marshal(CanonicalConfigFromRouterConfig(cfg))
	if err != nil {
		t.Fatal(err)
	}
	again, err := ParseYAMLBytes(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if again.Decisions[0].GetRequestParamsConfig().DefaultMaxTokens.Value != 4096 {
		t.Fatal("canonical export lost output default")
	}
	legacy, err := ParseRoutingYAMLBytes([]byte(fmt.Sprintf(template, "strip_unknown: true")))
	if err != nil {
		t.Fatal(err)
	}
	if legacy.Decisions[0].GetRequestParamsConfig().DefaultMaxTokens != nil {
		t.Fatal("absent default was inferred")
	}
	if err := ValidateRequestParamsPluginConfig(nil); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), "default_max_tokens: 4096") {
		t.Fatal("canonical field missing")
	}
}
