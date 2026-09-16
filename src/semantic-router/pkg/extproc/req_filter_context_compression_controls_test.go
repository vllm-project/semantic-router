package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestContextCompressionRequestControlsRequireRouteAuthorization(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-vsr-compression-control": "bypass,target=100",
		},
	}
	applyContextCompressionRequestControls(
		ctx,
		&config.ContextCompressionPluginConfig{
			Enabled: true,
			RequestControls: &config.ContextCompressionRequestControlsConfig{
				Enabled: false,
				Allowed: []string{"bypass", "target"},
			},
		},
	)
	if ctx.ContextCompressionSkipReason != "" ||
		ctx.ContextCompressionTargetTokens != nil {
		t.Fatal("disabled request controls trusted caller input")
	}
}

func TestContextCompressionRequestControlsClampAllowedTarget(t *testing.T) {
	ctx := &RequestContext{
		Headers: map[string]string{
			"x-control": "bypass,target=999999",
		},
	}
	applyContextCompressionRequestControls(
		ctx,
		&config.ContextCompressionPluginConfig{
			Enabled: true,
			RequestControls: &config.ContextCompressionRequestControlsConfig{
				Enabled:         true,
				Header:          "x-control",
				Allowed:         []string{"target"},
				MaxTargetTokens: 4096,
			},
		},
	)
	if ctx.ContextCompressionSkipReason != "" {
		t.Fatal("unallowed bypass directive was honored")
	}
	if ctx.ContextCompressionTargetTokens == nil ||
		*ctx.ContextCompressionTargetTokens != 4096 {
		t.Fatalf("target override = %#v", ctx.ContextCompressionTargetTokens)
	}
}
