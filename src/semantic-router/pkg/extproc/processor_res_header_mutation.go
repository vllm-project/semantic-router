package extproc

import (
	"strconv"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// warningsSizeLimit caps the encoded x-vsr-warnings header at a
// conservative size well under any HTTP/2 frame limit. More than fifty
// warnings already represent a deeper translation regression worth
// investigating via the structured log rather than the header.
const warningsSizeLimit = 4096

// protocolDefault is the wire protocol the trace reports when the request
// context did not resolve an explicit one. The router's default contract is
// OpenAI-compatible.
const protocolDefault = "openai"

type responseHeaderMutationBuilder struct {
	setHeaders []*core.HeaderValueOption
	seen       map[string]struct{}
}

func newResponseHeaderMutationBuilder() *responseHeaderMutationBuilder {
	return &responseHeaderMutationBuilder{
		setHeaders: make([]*core.HeaderValueOption, 0, 16),
		seen:       make(map[string]struct{}),
	}
}

func (builder *responseHeaderMutationBuilder) addString(key string, value string) {
	if value == "" {
		return
	}
	if _, exists := builder.seen[key]; exists {
		return
	}
	builder.seen[key] = struct{}{}
	builder.setHeaders = append(builder.setHeaders, &core.HeaderValueOption{
		Header: &core.HeaderValue{
			Key:      key,
			RawValue: []byte(value),
		},
	})
}

// addKeystone emits the keystone headers that ride on every Router-processed
// response: x-vsr-schema-version stamps the contract revision and
// x-vsr-response-path names how the response was produced. The path
// defaults to "upstream" when the caller has not classified a more specific
// path (cache, fast_response, looper, blocked, rate_limited, error, fallback).
func (builder *responseHeaderMutationBuilder) addKeystone(ctx *RequestContext) {
	path := ctx.ResponsePath
	if path == "" {
		path = headers.ResponsePathUpstream
	}
	builder.addString(headers.VSRSchemaVersion, headers.SchemaVersionValue)
	builder.addString(headers.VSRResponsePath, path)
}

// addConfigVersion names the configuration snapshot version that served.
func (builder *responseHeaderMutationBuilder) addConfigVersion(ctx *RequestContext) {
	if ctx.ConfigVersion > 0 {
		builder.addString(headers.VSRConfigVersion, strconv.FormatUint(ctx.ConfigVersion, 10))
	}
}

// addPublicRoutingHeaders adds the routing facts of the public set: the
// recipe, decision and model the Router selected, and the replay-id entry
// point. Each is omitted when the request has none.
func (builder *responseHeaderMutationBuilder) addPublicRoutingHeaders(ctx *RequestContext) {
	builder.addString(headers.VSRSelectedRecipe, string(ctx.Routing.RecipeName()))
	builder.addString(headers.VSRSelectedDecision, ctx.VSRSelectedDecisionName)
	builder.addString(headers.VSRSelectedModel, ctx.VSRSelectedModel)
	builder.addString(headers.RouterReplayID, ctx.RouterReplayID)
}

// addWarnings encodes the response_guard codes and the protocol diagnostics
// known so far into x-vsr-warnings.
func (builder *responseHeaderMutationBuilder) addWarnings(ctx *RequestContext) {
	builder.addString(headers.VSRWarnings, responseWarnings(ctx))
}

// addTrace adds x-vsr-trace for a caller allowed the debug headers.
func (builder *responseHeaderMutationBuilder) addTrace(ctx *RequestContext) {
	builder.addString(headers.VSRTrace, responseTrace(ctx))
}

// responseWarnings encodes x-vsr-warnings and counts and logs, once per
// request, each diagnostic it carries.
func responseWarnings(ctx *RequestContext) string {
	value, included := formatWarnings(ctx.ResponseGuard.warnings(), ctx.ProtocolDiagnostics)
	if reported := min(ctx.protocolDiagnosticsReported, included); included > reported {
		recordProtocolDiagnostics(ctx, ctx.ProtocolDiagnostics[reported:included])
		ctx.protocolDiagnosticsReported = included
	}
	return value
}

// bufferedWarnings is the x-vsr-warnings value the buffered body phase
// writes once the response_guard has read the answer, and whether the
// headers phase could not have sent it: the guard annotated the answer, or
// the body phase found translation losses.
func bufferedWarnings(ctx *RequestContext) (string, bool) {
	if ctx.VSRCacheHit || len(ctx.ResponseGuard.warnings()) == 0 && len(ctx.ProtocolDiagnostics) <= ctx.protocolDiagnosticsReported {
		return "", false
	}
	return responseWarnings(ctx), true
}

// formatWarnings encodes x-vsr-warnings as an RFC 8941 list: the guard codes,
// then one protocol member per diagnostic while they fit warningsSizeLimit,
// then a truncated member counting the diagnostics left out. It returns the
// value and how many diagnostics it carries.
func formatWarnings(codes []string, diagnostics llmprotocol.Diagnostics) (string, int) {
	members := make([]string, 0, len(codes)+len(diagnostics))
	for _, code := range codes {
		members = append(members, headers.Token(code).String())
	}
	// Reserve the largest possible truncation member before adding entries,
	// so one diagnostic obeys the same bound as a long list.
	trailer := headers.Token(headers.WarningTruncated).With("count", headers.Integer(int64(len(diagnostics)))).String()
	size := len(strings.Join(members, ", "))
	included := 0
	for _, diagnostic := range diagnostics {
		member := headers.Token(headers.WarningProtocol).
			With("action", headers.Token(string(diagnostic.Action))).
			With("reason", headers.String(diagnostic.Reason)).
			With("field", headers.String(diagnostic.Field)).String()
		if size+len(", ")+len(member)+len(", ")+len(trailer) > warningsSizeLimit {
			break
		}
		if size > 0 {
			size += len(", ")
		}
		size += len(member)
		members = append(members, member)
		included++
	}
	if included < len(diagnostics) {
		members = append(members, headers.Token(headers.WarningTruncated).
			With("count", headers.Integer(int64(len(diagnostics)-included))).String())
	}
	return strings.Join(members, ", "), included
}

func recordProtocolDiagnostics(ctx *RequestContext, diagnostics llmprotocol.Diagnostics) {
	if len(diagnostics) == 0 {
		return
	}
	inbound := normalizeProtocol(string(ctx.SourceFormat))
	outbound := normalizeProtocol(string(ctx.TargetFormat))
	for _, diagnostic := range diagnostics {
		metrics.RecordTranslationWarning(inbound, outbound, string(diagnostic.Action), diagnostic.Reason)
		logging.ComponentDebugEvent("extproc", "translation_lossy", map[string]interface{}{
			"request_id":        ctx.RequestID,
			"inbound_protocol":  inbound,
			"outbound_protocol": outbound,
			"field":             diagnostic.Field,
			"reason":            diagnostic.Reason,
			"action":            diagnostic.Action,
		})
	}
}

// normalizeProtocol returns the canonical protocol token for the trace and
// metrics, defaulting empty to "openai".
func normalizeProtocol(value string) string {
	v := strings.TrimSpace(value)
	switch llmprotocol.WireFormat(v) {
	case llmprotocol.AnthropicMessagesV1:
		return "anthropic"
	case llmprotocol.OpenAIResponsesV1:
		return "responses"
	case llmprotocol.OpenAIImagesV1:
		return "images"
	case llmprotocol.OpenAIChatV1, "":
		return protocolDefault
	default:
		return v
	}
}

func (builder *responseHeaderMutationBuilder) mutation() *ext_proc.HeaderMutation {
	if len(builder.setHeaders) == 0 {
		return nil
	}
	return &ext_proc.HeaderMutation{SetHeaders: builder.setHeaders}
}

// buildResponseHeaderMutation writes the public set and, for a caller allowed
// the debug headers, x-vsr-trace on a provider's response, success or error.
// A cache hit replays a response another request produced, so it writes
// none: the routing facts and warnings would not be this request's.
func buildResponseHeaderMutation(ctx *RequestContext) *ext_proc.HeaderMutation {
	if ctx == nil || ctx.VSRCacheHit {
		return nil
	}
	builder := newResponseHeaderMutationBuilder()
	builder.addKeystone(ctx)
	builder.addConfigVersion(ctx)
	builder.addPublicRoutingHeaders(ctx)
	builder.addWarnings(ctx)
	builder.addTrace(ctx)
	return builder.mutation()
}

// providerRouterHeaders are the headers of the Router's namespace a
// provider's response carries. The Router removes them before it writes its
// own, so no provider can forge a routing fact, warning or trace.
func providerRouterHeaders(responseHeaders *core.HeaderMap) []string {
	var names []string
	for _, header := range responseHeaders.GetHeaders() {
		if name := strings.ToLower(header.GetKey()); strings.HasPrefix(name, headers.VSRPrefix) {
			names = append(names, name)
		}
	}
	return names
}
