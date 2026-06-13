package extproc

import (
	"fmt"
	"strconv"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ir"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// lossinessHeaderSizeLimit caps the encoded x-vsr-lossiness-warnings
// header at a conservative size well under any HTTP/2 frame limit.
// >50 warnings already represent a deeper translation regression worth
// investigating via the structured log rather than the header.
const lossinessHeaderSizeLimit = 4096

// protocolDefault is the wire-shape token emitted in x-vsr-inbound-
// protocol / x-vsr-outbound-protocol when the request context did not
// resolve an explicit protocol. The router's default contract is
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

func (builder *responseHeaderMutationBuilder) addBool(key string, value bool) {
	builder.addString(key, strconv.FormatBool(value))
}

func (builder *responseHeaderMutationBuilder) addFloat(key string, value float64) {
	if value <= 0 {
		return
	}
	builder.addString(key, fmt.Sprintf("%.4f", value))
}

func (builder *responseHeaderMutationBuilder) addNonNegativeFloat(key string, value float64) {
	if value < 0 {
		return
	}
	builder.addString(key, fmt.Sprintf("%.4f", value))
}

func (builder *responseHeaderMutationBuilder) addInt(key string, value int) {
	if value <= 0 {
		return
	}
	builder.addString(key, strconv.Itoa(value))
}

// addNonNegativeInt emits the value when it is >= 0, so an explicit zero is
// still written. Mirrors addNonNegativeFloat; used for tri-state retention
// fields where 0 is a meaningful, explicitly-set value (not "unset").
func (builder *responseHeaderMutationBuilder) addNonNegativeInt(key string, value int) {
	if value < 0 {
		return
	}
	builder.addString(key, strconv.Itoa(value))
}

func (builder *responseHeaderMutationBuilder) addJoined(key string, values []string) {
	if len(values) == 0 {
		return
	}
	builder.addString(key, strings.Join(values, ","))
}

// addLossinessWarnings encodes ctx.IRExtensions.Warnings into the
// x-vsr-lossiness-warnings header, increments the per-warning Prometheus
// counter, and emits a structured log event per warning. Returns
// without emitting anything when warnings is empty.
//
// Format: comma-separated entries, each "severity;reason;field". The
// optional Warning.Detail stays out of the header (lives in the
// structured log) to keep the header short. If the encoded list would
// exceed lossinessHeaderSizeLimit the builder truncates and appends a
// synthetic "error;warnings_truncated;count=N" trailer.
func (builder *responseHeaderMutationBuilder) addLossinessWarnings(
	ctx *RequestContext,
	warnings []ir.Warning,
) {
	if len(warnings) == 0 {
		return
	}

	inbound := normalizeProtocol(ctx.ClientProtocol)
	outbound := normalizeProtocol(ctx.APIFormat)

	var sb strings.Builder
	truncatedAt := -1
	for i, w := range warnings {
		entry := formatLossinessEntry(w)
		separatorLen := 0
		if sb.Len() > 0 {
			separatorLen = 1
		}
		if sb.Len()+separatorLen+len(entry) > lossinessHeaderSizeLimit && sb.Len() > 0 {
			truncatedAt = i
			break
		}
		if separatorLen > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(entry)
		recordWarning(ctx, inbound, outbound, w)
	}

	if truncatedAt >= 0 {
		trailer := fmt.Sprintf("%s;%s;count=%d",
			ir.WarningSeverityError,
			ir.ReasonWarningsTruncated,
			len(warnings)-truncatedAt,
		)
		if sb.Len() > 0 {
			sb.WriteByte(',')
		}
		sb.WriteString(trailer)
	}

	builder.addString(headers.VSRLossinessWarnings, sb.String())
}

func formatLossinessEntry(w ir.Warning) string {
	return fmt.Sprintf("%s;%s;%s",
		w.Severity,
		sanitizeWarningField(string(w.Reason)),
		sanitizeWarningField(w.Field),
	)
}

// sanitizeWarningField percent-encodes the format separators ',' and
// ';' so a pathological JSON-path field name cannot break the
// single-line encoding, and strips CR/LF so a hostile value cannot
// inject a new header line. PR2's parser never produces such paths;
// this is belt-and-suspenders.
func sanitizeWarningField(field string) string {
	if !strings.ContainsAny(field, ",;\r\n") {
		return field
	}
	var sb strings.Builder
	sb.Grow(len(field))
	for _, r := range field {
		switch r {
		case ',':
			sb.WriteString("%2C")
		case ';':
			sb.WriteString("%3B")
		case '\r':
			sb.WriteString("%0D")
		case '\n':
			sb.WriteString("%0A")
		default:
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

func recordWarning(ctx *RequestContext, inbound, outbound string, w ir.Warning) {
	metrics.RecordTranslationWarning(inbound, outbound, w.Severity.String(), string(w.Reason))
	logging.ComponentDebugEvent("extproc", "translation_lossy", map[string]interface{}{
		"request_id":        ctx.RequestID,
		"inbound_protocol":  inbound,
		"outbound_protocol": outbound,
		"field":             w.Field,
		"reason":            w.Reason,
		"severity":          w.Severity.String(),
		"detail":            w.Detail,
	})
}

// normalizeProtocol returns the canonical protocol token for headers
// and metrics, defaulting empty to "openai".
func normalizeProtocol(value string) string {
	v := strings.TrimSpace(value)
	if v == "" {
		return protocolDefault
	}
	return v
}

func (builder *responseHeaderMutationBuilder) mutation() *ext_proc.HeaderMutation {
	if len(builder.setHeaders) == 0 {
		return nil
	}
	return &ext_proc.HeaderMutation{SetHeaders: builder.setHeaders}
}

func buildResponseHeaderMutation(
	ctx *RequestContext,
	isSuccessful bool,
) *ext_proc.HeaderMutation {
	if ctx == nil {
		return nil
	}

	builder := newResponseHeaderMutationBuilder()

	// Protocol markers and lossiness warnings ride on every response
	// (success or 4xx/5xx) so clients can always tell which translation
	// cell handled the call. Cache-hit responses are an exception: the
	// IRExtensions.Warnings slice is per-request, so a cached response
	// would attribute warnings from a different request — we skip the
	// new headers entirely on cache hits and let the cached payload
	// flow unchanged.
	if !ctx.VSRCacheHit {
		builder.addString(headers.VSRInboundProtocol, normalizeProtocol(ctx.ClientProtocol))
		builder.addString(headers.VSROutboundProtocol, normalizeProtocol(ctx.APIFormat))
		if ctx.IRExtensions != nil {
			builder.addLossinessWarnings(ctx, ctx.IRExtensions.Warnings)
		}
	}

	if !isSuccessful || ctx.VSRCacheHit {
		return builder.mutation()
	}

	addStandardDecisionHeaders(builder, ctx)
	addMatchedSignalHeaders(builder, ctx)
	addRetentionDirectiveHeaders(builder, ctx)
	return builder.mutation()
}

// addRetentionDirectiveHeaders emits the matched decision's EMIT retention
// directive to the response as x-vsr-retention-* headers so the inference pool
// and operators can observe the router's retention intent at the wire
// (issue #2009). Only fields the directive explicitly set are emitted, mirroring
// the tri-state semantics of config.RetentionDirective; an unset field is
// omitted rather than sent as a default. Emitted only on successful,
// non-cache-hit responses (same gate as the standard decision headers).
func addRetentionDirectiveHeaders(builder *responseHeaderMutationBuilder, ctx *RequestContext) {
	r := ctx.EmittedRetention
	if r == nil {
		return
	}
	if r.Drop != nil {
		builder.addBool(headers.VSRRetentionDrop, *r.Drop)
	}
	if r.TTLTurns != nil {
		// Tri-state: emit whenever explicitly set, including ttl_turns: 0
		// (a valid no-op the validator permits). The runtime TTL override
		// still applies only when > 0; the header reflects intent, not effect.
		builder.addNonNegativeInt(headers.VSRRetentionTTLTurns, *r.TTLTurns)
	}
	if r.KeepCurrentModel != nil {
		builder.addBool(headers.VSRRetentionKeepCurrentModel, *r.KeepCurrentModel)
	}
	if r.PreferPrefixRetention != nil {
		builder.addBool(headers.VSRRetentionPreferPrefix, *r.PreferPrefixRetention)
	}
}

// addStandardDecisionHeaders adds the per-request decision headers
// (selected category, model, reasoning, modality, etc.) emitted only on
// successful non-cache-hit responses.
func addStandardDecisionHeaders(builder *responseHeaderMutationBuilder, ctx *RequestContext) {
	builder.addString(headers.VSRSelectedCategory, ctx.VSRSelectedCategory)
	builder.addString(headers.VSRSelectedDecision, ctx.VSRSelectedDecisionName)
	if ctx.VSRSelectedDecisionName != "" {
		builder.addNonNegativeFloat(headers.VSRSelectedConfidence, ctx.VSRSelectedDecisionConfidence)
	}
	if ctx.ModalityClassification != nil && ctx.ModalityClassification.Modality != "" {
		modalityValue := ctx.ModalityClassification.Modality
		if ctx.ModalityClassification.Method != "" {
			modalityValue += ";" + ctx.ModalityClassification.Method
		}
		builder.addString(headers.VSRSelectedModality, modalityValue)
	}
	builder.addString(headers.VSRSelectedReasoning, ctx.VSRReasoningMode)
	builder.addString(headers.VSRSelectedModel, ctx.VSRSelectedModel)
	builder.addString(headers.VSRSessionPhase, sessionPolicyPhase(ctx))
	builder.addBool(headers.VSRInjectedSystemPrompt, ctx.VSRInjectedSystemPrompt)
	builder.addString(headers.RouterReplayID, ctx.RouterReplayID)
	if ctx.VSRCacheSimilarity > 0 {
		builder.addFloat("x-vsr-cache-similarity", float64(ctx.VSRCacheSimilarity))
	}
	addSessionBudgetResponseHeaders(builder, ctx)
}

// addSessionBudgetResponseHeaders reports the session token-budget decision
// (Opp-5) on a normal, non-terminate response. The terminate short-circuit emits
// its own headers (including x-vsr-budget-exceeded) on the immediate response.
func addSessionBudgetResponseHeaders(builder *responseHeaderMutationBuilder, ctx *RequestContext) {
	if ctx.SessionBudget == nil {
		return
	}
	builder.addString(headers.VSRBudgetStage, ctx.SessionBudget.Stage.String())
	builder.addString(headers.VSRBudgetRatio, fmt.Sprintf("%.2f", ctx.SessionBudget.Ratio))
}

// addMatchedSignalHeaders adds the signal-evaluation headers (matched
// keywords, embeddings, etc.) describing which signal rules fired for
// this request.
func addMatchedSignalHeaders(builder *responseHeaderMutationBuilder, ctx *RequestContext) {
	builder.addJoined(headers.VSRMatchedKeywords, ctx.VSRMatchedKeywords)
	builder.addJoined(headers.VSRMatchedEmbeddings, ctx.VSRMatchedEmbeddings)
	builder.addJoined(headers.VSRMatchedDomains, ctx.VSRMatchedDomains)
	builder.addJoined(headers.VSRMatchedFactCheck, ctx.VSRMatchedFactCheck)
	builder.addJoined(headers.VSRMatchedUserFeedback, ctx.VSRMatchedUserFeedback)
	builder.addJoined(headers.VSRMatchedReask, ctx.VSRMatchedReask)
	builder.addJoined(headers.VSRMatchedPreference, ctx.VSRMatchedPreference)
	builder.addJoined(headers.VSRMatchedLanguage, ctx.VSRMatchedLanguage)
	builder.addJoined(headers.VSRMatchedContext, ctx.VSRMatchedContext)
	builder.addInt(headers.VSRContextTokenCount, ctx.VSRContextTokenCount)
	builder.addJoined(headers.VSRMatchedStructure, ctx.VSRMatchedStructure)
	builder.addJoined(headers.VSRMatchedComplexity, ctx.VSRMatchedComplexity)
	builder.addJoined(headers.VSRMatchedModality, ctx.VSRMatchedModality)
	builder.addJoined(headers.VSRMatchedAuthz, ctx.VSRMatchedAuthz)
	builder.addJoined(headers.VSRMatchedJailbreak, ctx.VSRMatchedJailbreak)
	builder.addJoined(headers.VSRMatchedPII, ctx.VSRMatchedPII)
	builder.addJoined(headers.VSRMatchedKB, ctx.VSRMatchedKB)
	builder.addJoined(headers.VSRMatchedConversation, ctx.VSRMatchedConversation)
	builder.addJoined(headers.VSRMatchedEvent, ctx.VSRMatchedEvent)
	builder.addJoined(headers.VSRMatchedProjection, ctx.VSRMatchedProjection)
}

func sessionPolicyPhase(ctx *RequestContext) string {
	if ctx == nil || ctx.VSRSessionPolicy == nil {
		return ""
	}
	phase, ok := ctx.VSRSessionPolicy["phase"].(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(phase)
}
