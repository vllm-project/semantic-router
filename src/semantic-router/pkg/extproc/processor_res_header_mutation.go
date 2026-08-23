package extproc

import (
	"fmt"
	"strconv"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

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

// addKeystone emits the v0.4 keystone headers that ride on every
// VSR-processed response: x-vsr-schema-version stamps the contract revision
// and x-vsr-response-path names how the response was produced. The path
// defaults to "upstream" when the caller has not classified a more specific
// path (cache, fast_response, looper, rate_limited, error, image_generation).
// See issue #2203.
func (builder *responseHeaderMutationBuilder) addKeystone(ctx *RequestContext) {
	path := ctx.ResponsePath
	if path == "" {
		path = headers.ResponsePathUpstream
	}
	builder.addString(headers.VSRSchemaVersion, headers.SchemaVersionValue)
	builder.addString(headers.VSRResponsePath, path)
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

	// Keystone headers ride on every non-cache-hit response (success or
	// 4xx/5xx). Cache hits preserve the stored response surface unchanged.
	if !ctx.VSRCacheHit {
		// Keystone headers (schema-version + response-path) ride on every
		// non-cache-hit response. This function only handles upstream responses,
		// so the path defaults to "upstream".
		builder.addKeystone(ctx)
	}

	if !isSuccessful || ctx.VSRCacheHit {
		return builder.mutation()
	}

	// Final routing facts ride on every successful response. The intermediate
	// decision/classification details, matched signals, and the retention
	// directive headers are demoted to the x-vsr-debug surface (#2205, #2200):
	// they remain recoverable from the replay record via x-vsr-replay-id. The
	// retention directive's runtime effects (cache write skip, TTL override,
	// model-switch-gate stay) are applied internally regardless of the header,
	// so demoting the wire header does not change behavior.
	addFinalDecisionHeaders(builder, ctx)
	if debugHeadersRequested(ctx) {
		addRetentionDirectiveHeaders(builder, ctx)
		addDecisionDetailHeaders(builder, ctx)
		addMatchedSignalHeaders(builder, ctx)
	}
	return builder.mutation()
}

// addRetentionDirectiveHeaders emits the matched decision's EMIT retention
// directive to the response as x-vsr-retention-* headers so operators can
// observe the router's retention intent at the wire (issue #2009). Per the v0.4
// contract (#2200) these are demoted to the x-vsr-debug surface: the directive's
// runtime effects are applied internally, so the headers are observability only.
// Only fields the directive explicitly set are emitted, mirroring the tri-state
// semantics of config.RetentionDirective; an unset field is omitted rather than
// sent as a default.
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

// addFinalDecisionHeaders adds the final routing facts that ride on the default
// surface of every successful non-cache-hit response: the selected decision and
// its confidence, the selection algorithm, the selected model, and the replay-id
// entry point.
func addFinalDecisionHeaders(builder *responseHeaderMutationBuilder, ctx *RequestContext) {
	builder.addString(headers.VSRSelectedRecipe, string(ctx.Routing.RecipeName()))
	builder.addString(headers.VSRSelectedDecision, ctx.VSRSelectedDecisionName)
	if ctx.VSRSelectedDecisionName != "" {
		builder.addNonNegativeFloat(headers.VSRSelectedConfidence, ctx.VSRSelectedDecisionConfidence)
	}
	builder.addString(headers.VSRSelectedAlgorithm, ctx.VSRSelectionMethod)
	builder.addString(headers.VSRSelectedModel, ctx.VSRSelectedModel)
	builder.addString(headers.RouterReplayID, ctx.RouterReplayID)
}

// addDecisionDetailHeaders adds the intermediate decision/classification details
// (category, modality, reasoning, session phase, injected-system-prompt, cache
// similarity). Per the v0.4 contract (#2205) these are demoted off the default
// surface and emitted only under x-vsr-debug; they remain in the replay record.
func addDecisionDetailHeaders(builder *responseHeaderMutationBuilder, ctx *RequestContext) {
	builder.addString(headers.VSRSelectedCategory, ctx.VSRSelectedCategory)
	if ctx.ModalityClassification != nil && ctx.ModalityClassification.Modality != "" {
		modalityValue := ctx.ModalityClassification.Modality
		if ctx.ModalityClassification.Method != "" {
			modalityValue += ";" + ctx.ModalityClassification.Method
		}
		builder.addString(headers.VSRSelectedModality, modalityValue)
	}
	builder.addString(headers.VSRSelectedReasoning, ctx.VSRReasoningMode)
	builder.addString(headers.VSRSessionPhase, sessionPolicyPhase(ctx))
	builder.addString(headers.VSRLearningMethods, learningPolicyMethodsHeader(ctx))
	builder.addString(headers.VSRLearningActions, learningPolicyPairHeader(ctx, learningPolicyFieldAction))
	builder.addString(headers.VSRLearningScopes, learningPolicyPairHeader(ctx, learningPolicyFieldScope))
	builder.addString(headers.VSRLearningReasons, learningPolicyPairHeader(ctx, learningPolicyFieldReason))
	builder.addBool(headers.VSRInjectedSystemPrompt, ctx.VSRInjectedSystemPrompt)
	if ctx.VSRCacheSimilarity > 0 {
		builder.addFloat("x-vsr-cache-similarity", float64(ctx.VSRCacheSimilarity))
	}
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
	if policy, ok := protectionLearningPolicyForContext(ctx); ok {
		if phase := policy.SessionPhase(); phase != "" {
			return phase
		}
	}
	return ""
}

func learningPolicyMethodsHeader(ctx *RequestContext) string {
	policies := learningPoliciesForHeaders(ctx)
	if len(policies) == 0 {
		return ""
	}
	values := make([]string, 0, len(policies))
	for _, policy := range policies {
		values = append(values, sanitizeDelimitedHeaderField(string(policy.Method)))
	}
	return strings.Join(values, ",")
}

func learningPolicyPairHeader(ctx *RequestContext, field routerLearningPolicyField) string {
	policies := learningPoliciesForHeaders(ctx)
	if len(policies) == 0 {
		return ""
	}
	pairs := make([]string, 0, len(policies))
	for _, policy := range policies {
		value := policy.StringField(field)
		if value == "" {
			continue
		}
		pairs = append(pairs, sanitizeDelimitedHeaderField(string(policy.Method))+"="+sanitizeDelimitedHeaderField(value))
	}
	return strings.Join(pairs, ",")
}

// sanitizeDelimitedHeaderField keeps one server-owned value inside the
// comma/semicolon-delimited diagnostic header grammar. Encoding rather than
// dropping separators makes the representation deterministic and prevents
// control characters from creating another header line.
func sanitizeDelimitedHeaderField(value string) string {
	if !strings.ContainsAny(value, ",;\r\n") {
		return value
	}
	var encoded strings.Builder
	encoded.Grow(len(value))
	for _, character := range value {
		switch character {
		case ',':
			encoded.WriteString("%2C")
		case ';':
			encoded.WriteString("%3B")
		case '\r':
			encoded.WriteString("%0D")
		case '\n':
			encoded.WriteString("%0A")
		default:
			encoded.WriteRune(character)
		}
	}
	return encoded.String()
}

func learningPoliciesForHeaders(ctx *RequestContext) []routerLearningPolicy {
	if ctx == nil {
		return nil
	}
	if !ctx.VSRLearningPolicies.Empty() {
		policies := make([]routerLearningPolicy, 0, 2)
		if policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodAdaptation); ok {
			policies = append(policies, policy)
		}
		if policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection); ok {
			policies = append(policies, policy)
		}
		return policies
	}
	if ctx.VSRLearningPolicy == nil || ctx.VSRLearningPolicy.Empty() {
		return nil
	}
	policy := *ctx.VSRLearningPolicy
	if policy.Method == "" {
		policy.Method = routerLearningMethodProtection
	}
	return []routerLearningPolicy{policy}
}
