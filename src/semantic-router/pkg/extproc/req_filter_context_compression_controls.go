package extproc

import (
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const defaultContextCompressionControlHeader = "x-vsr-compression-control"

func applyContextCompressionRequestControls(
	ctx *RequestContext,
	plugin *config.ContextCompressionPluginConfig,
) {
	if ctx == nil || plugin == nil {
		return
	}
	controls := plugin.RequestControls
	if controls == nil || !controls.Enabled {
		return
	}
	header := contextCompressionControlHeader(controls)
	allowed := contextCompressionAllowedDirectives(controls)
	for _, rawDirective := range strings.Split(
		strings.ToLower(headerValueCI(ctx, header)),
		",",
	) {
		applyContextCompressionDirective(ctx, controls, allowed, rawDirective)
	}
}

func contextCompressionControlHeader(
	controls *config.ContextCompressionRequestControlsConfig,
) string {
	header := strings.TrimSpace(controls.Header)
	if header == "" {
		return defaultContextCompressionControlHeader
	}
	return header
}

func contextCompressionAllowedDirectives(
	controls *config.ContextCompressionRequestControlsConfig,
) map[string]struct{} {
	allowed := make(map[string]struct{}, len(controls.Allowed))
	for _, directive := range controls.Allowed {
		allowed[strings.ToLower(strings.TrimSpace(directive))] = struct{}{}
	}
	return allowed
}

func applyContextCompressionDirective(
	ctx *RequestContext,
	controls *config.ContextCompressionRequestControlsConfig,
	allowed map[string]struct{},
	rawDirective string,
) {
	name, value, hasValue := strings.Cut(strings.TrimSpace(rawDirective), "=")
	if _, ok := allowed[name]; !ok {
		return
	}
	if name == "bypass" {
		ctx.ContextCompressionSkipReason = "request_bypass"
		return
	}
	if name == "target" && hasValue {
		ctx.ContextCompressionTargetTokens = boundedCompressionTarget(
			value,
			controls.MaxTargetTokens,
		)
	}
}

func boundedCompressionTarget(raw string, maximum int) *int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || value <= 0 {
		return nil
	}
	if maximum > 0 && value > maximum {
		value = maximum
	}
	return &value
}
