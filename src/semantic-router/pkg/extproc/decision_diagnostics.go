package extproc

import (
	"encoding/json"
	"sort"
	"strings"
	"unicode/utf8"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/types/known/structpb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	decisionDiagnosticsSchemaVersion = 1
	decisionDiagnosticsNamespace     = "vllm.semantic_router"
	decisionDiagnosticsField         = "decision_diagnostics"
)

type decisionDiagnosticsPayload struct {
	SchemaVersion      int                            `json:"schemaVersion"`
	Decision           string                         `json:"decision"`
	Category           string                         `json:"category,omitempty"`
	SelectedModel      string                         `json:"selectedModel,omitempty"`
	SelectionAlgorithm string                         `json:"selectionAlgorithm,omitempty"`
	SelectionMethod    string                         `json:"selectionMethod,omitempty"`
	DecisionConfidence float64                        `json:"decisionConfidence"`
	MatchedRules       []string                       `json:"matchedRules"`
	Signals            []decisionDiagnosticSignal     `json:"signals"`
	Projections        []decisionDiagnosticProjection `json:"projections"`
	Truncated          bool                           `json:"truncated"`
}

type decisionDiagnosticSignal struct {
	Key        string   `json:"key"`
	Type       string   `json:"type"`
	Name       string   `json:"name"`
	Executed   bool     `json:"executed"`
	Matched    bool     `json:"matched"`
	Value      *float64 `json:"value,omitempty"`
	Confidence *float64 `json:"confidence,omitempty"`
}

type decisionDiagnosticProjection struct {
	Name    string   `json:"name"`
	Matched bool     `json:"matched"`
	Score   *float64 `json:"score,omitempty"`
}

func attachDecisionDiagnostics(response *ext_proc.ProcessingResponse, ctx *RequestContext) {
	if response == nil || ctx == nil || ctx.VSRSelectedDecision == nil {
		return
	}
	cfg := ctx.VSRSelectedDecision.GetDecisionDiagnosticsConfig()
	if cfg == nil {
		return
	}

	payload, ok := buildDecisionDiagnosticsPayload(ctx, cfg)
	if !ok {
		return
	}
	metadata, ok := decisionDiagnosticsStructValue(payload, cfg.MaxPayloadBytes)
	if !ok {
		return
	}
	setDecisionDiagnosticsMetadata(response, metadata)
}

func attachDecisionDiagnosticsOnSuccess(
	response *ext_proc.ProcessingResponse,
	requestErr error,
	ctx *RequestContext,
) {
	if requestErr != nil {
		return
	}
	attachDecisionDiagnostics(response, ctx)
}

func decisionDiagnosticsStructValue(
	payload decisionDiagnosticsPayload,
	maxPayloadBytes int,
) (*structpb.Struct, bool) {
	raw, err := json.Marshal(payload)
	if err != nil || len(raw) > maxPayloadBytes {
		return nil, false
	}
	var fields map[string]interface{}
	if unmarshalErr := json.Unmarshal(raw, &fields); unmarshalErr != nil {
		return nil, false
	}
	metadata, err := structpb.NewStruct(fields)
	if err != nil {
		return nil, false
	}
	return metadata, true
}

func setDecisionDiagnosticsMetadata(response *ext_proc.ProcessingResponse, metadata *structpb.Struct) {
	if response.DynamicMetadata == nil {
		response.DynamicMetadata = &structpb.Struct{Fields: map[string]*structpb.Value{}}
	}
	if response.DynamicMetadata.Fields == nil {
		response.DynamicMetadata.Fields = map[string]*structpb.Value{}
	}
	namespace := response.DynamicMetadata.Fields[decisionDiagnosticsNamespace].GetStructValue()
	if namespace == nil {
		namespace = &structpb.Struct{Fields: map[string]*structpb.Value{}}
		response.DynamicMetadata.Fields[decisionDiagnosticsNamespace] = structpb.NewStructValue(namespace)
	}
	if namespace.Fields == nil {
		namespace.Fields = map[string]*structpb.Value{}
	}
	namespace.Fields[decisionDiagnosticsField] = structpb.NewStructValue(metadata)
}

func buildDecisionDiagnosticsPayload(
	ctx *RequestContext,
	cfg *config.DecisionDiagnosticsPluginConfig,
) (decisionDiagnosticsPayload, bool) {
	if !decisionDiagnosticsConfigUsable(ctx, cfg) {
		return decisionDiagnosticsPayload{}, false
	}

	payload := newDecisionDiagnosticsPayload(ctx, cfg.MaxSignals, cfg.MaxTextRunes)
	appendDecisionDiagnosticSignals(&payload, ctx, cfg)
	return fitDecisionDiagnosticsPayload(payload, cfg.MaxPayloadBytes)
}

func decisionDiagnosticsConfigUsable(ctx *RequestContext, cfg *config.DecisionDiagnosticsPluginConfig) bool {
	return ctx != nil && ctx.VSRSelectedDecision != nil && cfg != nil && cfg.Enabled &&
		cfg.MaxSignals > 0 && cfg.MaxProjections > 0 && cfg.MaxTextRunes > 0 && cfg.MaxPayloadBytes > 0
}

func newDecisionDiagnosticsPayload(ctx *RequestContext, maxSignals int, maxTextRunes int) decisionDiagnosticsPayload {
	payload := decisionDiagnosticsPayload{
		SchemaVersion:      decisionDiagnosticsSchemaVersion,
		Decision:           truncateDiagnosticText(ctx.VSRSelectedDecisionName, maxTextRunes),
		Category:           truncateDiagnosticText(ctx.VSRSelectedCategory, maxTextRunes),
		SelectedModel:      truncateDiagnosticText(ctx.VSRSelectedModel, maxTextRunes),
		SelectionAlgorithm: truncateDiagnosticText(decisionDiagnosticsAlgorithm(ctx.VSRSelectedDecision), maxTextRunes),
		SelectionMethod:    truncateDiagnosticText(ctx.VSRSelectionMethod, maxTextRunes),
		DecisionConfidence: ctx.VSRSelectedDecisionConfidence,
		MatchedRules:       []string{},
		Signals:            []decisionDiagnosticSignal{},
		Projections:        []decisionDiagnosticProjection{},
	}
	if payload.Decision == "" {
		payload.Decision = truncateDiagnosticText(ctx.VSRSelectedDecision.Name, maxTextRunes)
	}
	payload.MatchedRules, payload.Truncated = boundedDecisionMatchedRules(ctx, maxSignals, maxTextRunes)
	payload.Truncated = diagnosticTextWasTruncated(ctx.VSRSelectedDecisionName, payload.Decision) ||
		diagnosticTextWasTruncated(ctx.VSRSelectedCategory, payload.Category) ||
		diagnosticTextWasTruncated(ctx.VSRSelectedModel, payload.SelectedModel) ||
		diagnosticTextWasTruncated(decisionDiagnosticsAlgorithm(ctx.VSRSelectedDecision), payload.SelectionAlgorithm) ||
		diagnosticTextWasTruncated(ctx.VSRSelectionMethod, payload.SelectionMethod) || payload.Truncated
	return payload
}

func decisionDiagnosticsAlgorithm(decision *config.Decision) string {
	if decision == nil || decision.Algorithm == nil {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(decision.Algorithm.Type))
}

func appendDecisionDiagnosticSignals(
	payload *decisionDiagnosticsPayload,
	ctx *RequestContext,
	cfg *config.DecisionDiagnosticsPluginConfig,
) {
	refs := collectDecisionDiagnosticSignalRefs(&ctx.VSRSelectedDecision.Rules)
	matched := matchedDecisionDiagnosticSignals(ctx)
	for _, ref := range refs {
		if len(payload.Signals) >= cfg.MaxSignals {
			payload.Truncated = true
			break
		}
		key := ref.signalKey()
		boundedKey := truncateDiagnosticText(key, cfg.MaxTextRunes)
		if boundedKey != key {
			payload.Truncated = true
		}
		signal := newDecisionDiagnosticSignal(ctx, ref, boundedKey, matched[key], cfg.MaxTextRunes)
		payload.Signals = append(payload.Signals, signal)

		if ref.signalType == config.SignalTypeProjection {
			appendDecisionDiagnosticProjection(payload, ctx, ref, matched[key], cfg)
		}
	}
}

func newDecisionDiagnosticSignal(
	ctx *RequestContext,
	ref decisionDiagnosticSignalRef,
	boundedKey string,
	matched bool,
	maxTextRunes int,
) decisionDiagnosticSignal {
	signal := decisionDiagnosticSignal{
		Key:      boundedKey,
		Type:     truncateDiagnosticText(ref.signalType, maxTextRunes),
		Name:     truncateDiagnosticText(ref.name, maxTextRunes),
		Executed: ctx.VSRExecutedSignalTypes[ref.signalType],
		Matched:  matched,
	}
	if value, exists := ctx.VSRSignalValues[ref.signalKey()]; exists {
		signal.Value = diagnosticFloat64(value)
	}
	if confidence, exists := ctx.VSRSignalConfidences[ref.signalKey()]; exists {
		signal.Confidence = diagnosticFloat64(confidence)
	}
	return signal
}

func appendDecisionDiagnosticProjection(
	payload *decisionDiagnosticsPayload,
	ctx *RequestContext,
	ref decisionDiagnosticSignalRef,
	matched bool,
	cfg *config.DecisionDiagnosticsPluginConfig,
) {
	if len(payload.Projections) >= cfg.MaxProjections {
		payload.Truncated = true
		return
	}
	name := truncateDiagnosticText(ref.name, cfg.MaxTextRunes)
	if name != ref.name {
		payload.Truncated = true
	}
	projection := decisionDiagnosticProjection{Name: name, Matched: matched}
	if score, exists := projectionDiagnosticScore(ctx, ref.name); exists {
		projection.Score = diagnosticFloat64(score)
	}
	payload.Projections = append(payload.Projections, projection)
}

func fitDecisionDiagnosticsPayload(
	payload decisionDiagnosticsPayload,
	maxPayloadBytes int,
) (decisionDiagnosticsPayload, bool) {
	for {
		raw, err := json.Marshal(payload)
		if err != nil {
			return decisionDiagnosticsPayload{}, false
		}
		if len(raw) <= maxPayloadBytes {
			return payload, true
		}
		payload.Truncated = true
		switch {
		case len(payload.Projections) > 0:
			payload.Projections = payload.Projections[:len(payload.Projections)-1]
		case len(payload.Signals) > 0:
			payload.Signals = payload.Signals[:len(payload.Signals)-1]
		case len(payload.MatchedRules) > 0:
			payload.MatchedRules = payload.MatchedRules[:len(payload.MatchedRules)-1]
		default:
			return decisionDiagnosticsPayload{}, false
		}
	}
}

type decisionDiagnosticSignalRef struct {
	signalType string
	name       string
}

func (r decisionDiagnosticSignalRef) signalKey() string {
	return strings.ToLower(strings.TrimSpace(r.signalType)) + ":" + strings.TrimSpace(r.name)
}

func collectDecisionDiagnosticSignalRefs(root *config.RuleNode) []decisionDiagnosticSignalRef {
	refsByKey := map[string]decisionDiagnosticSignalRef{}
	var walk func(*config.RuleNode)
	walk = func(node *config.RuleNode) {
		if node == nil {
			return
		}
		if strings.TrimSpace(node.Type) != "" && strings.TrimSpace(node.Name) != "" {
			ref := decisionDiagnosticSignalRef{
				signalType: strings.ToLower(strings.TrimSpace(node.Type)),
				name:       strings.TrimSpace(node.Name),
			}
			refsByKey[ref.signalKey()] = ref
		}
		for i := range node.Conditions {
			walk(&node.Conditions[i])
		}
	}
	walk(root)

	keys := make([]string, 0, len(refsByKey))
	for key := range refsByKey {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	refs := make([]decisionDiagnosticSignalRef, 0, len(keys))
	for _, key := range keys {
		refs = append(refs, refsByKey[key])
	}
	return refs
}

func matchedDecisionDiagnosticSignals(ctx *RequestContext) map[string]bool {
	matched := map[string]bool{}
	add := func(signalType string, names []string) {
		prefix := strings.ToLower(strings.TrimSpace(signalType)) + ":"
		for _, name := range names {
			trimmed := strings.TrimSpace(name)
			if trimmed == "" {
				continue
			}
			if strings.HasPrefix(strings.ToLower(trimmed), prefix) {
				matched[prefix+strings.TrimSpace(trimmed[len(prefix):])] = true
			} else {
				matched[prefix+trimmed] = true
			}
		}
	}
	add(config.SignalTypeKeyword, ctx.VSRMatchedKeywords)
	add(config.SignalTypeEmbedding, ctx.VSRMatchedEmbeddings)
	add(config.SignalTypeDomain, ctx.VSRMatchedDomains)
	add(config.SignalTypeFactCheck, ctx.VSRMatchedFactCheck)
	add(config.SignalTypeUserFeedback, ctx.VSRMatchedUserFeedback)
	add(config.SignalTypeReask, ctx.VSRMatchedReask)
	add(config.SignalTypePreference, ctx.VSRMatchedPreference)
	add(config.SignalTypeLanguage, ctx.VSRMatchedLanguage)
	add(config.SignalTypeContext, ctx.VSRMatchedContext)
	add(config.SignalTypeStructure, ctx.VSRMatchedStructure)
	add(config.SignalTypeComplexity, ctx.VSRMatchedComplexity)
	add(config.SignalTypeModality, ctx.VSRMatchedModality)
	add(config.SignalTypeAuthz, ctx.VSRMatchedAuthz)
	add(config.SignalTypeJailbreak, ctx.VSRMatchedJailbreak)
	add(config.SignalTypePII, ctx.VSRMatchedPII)
	add(config.SignalTypeKB, ctx.VSRMatchedKB)
	add(config.SignalTypeConversation, ctx.VSRMatchedConversation)
	add(config.SignalTypeEvent, ctx.VSRMatchedEvent)
	add(config.SignalTypeProjection, ctx.VSRMatchedProjection)
	for _, label := range ctx.VSRMatchedDecisionRules {
		if ref, ok := decisionDiagnosticSignalRefFromLabel(label); ok {
			matched[ref.signalKey()] = true
		}
	}
	return matched
}

func boundedDecisionMatchedRules(ctx *RequestContext, maxSignals int, maxTextRunes int) ([]string, bool) {
	eligible := map[string]bool{}
	for _, ref := range collectDecisionDiagnosticSignalRefs(&ctx.VSRSelectedDecision.Rules) {
		eligible[ref.signalKey()] = true
	}

	labels := make([]string, 0, len(ctx.VSRMatchedDecisionRules))
	seen := map[string]bool{}
	truncated := false
	for _, label := range ctx.VSRMatchedDecisionRules {
		ref, ok := decisionDiagnosticSignalRefFromLabel(label)
		if !ok || !eligible[ref.signalKey()] || seen[ref.signalKey()] {
			continue
		}
		bounded := truncateDiagnosticText(ref.signalKey(), maxTextRunes)
		truncated = truncated || bounded != ref.signalKey()
		labels = append(labels, bounded)
		seen[ref.signalKey()] = true
	}
	sort.Strings(labels)
	if len(labels) > maxSignals {
		labels = labels[:maxSignals]
		truncated = true
	}
	return labels, truncated
}

func decisionDiagnosticSignalRefFromLabel(label string) (decisionDiagnosticSignalRef, bool) {
	signalType, name, ok := strings.Cut(strings.TrimSpace(label), ":")
	if !ok || strings.TrimSpace(signalType) == "" || strings.TrimSpace(name) == "" {
		return decisionDiagnosticSignalRef{}, false
	}
	return decisionDiagnosticSignalRef{
		signalType: strings.ToLower(strings.TrimSpace(signalType)),
		name:       strings.TrimSpace(name),
	}, true
}

func projectionDiagnosticScore(ctx *RequestContext, name string) (float64, bool) {
	if value, ok := ctx.VSRProjectionScores[name]; ok {
		return value, true
	}
	value, ok := ctx.VSRProjectionScores[config.SignalTypeProjection+":"+name]
	return value, ok
}

func diagnosticFloat64(value float64) *float64 {
	copy := value
	return &copy
}

func truncateDiagnosticText(value string, maxRunes int) string {
	value = strings.TrimSpace(value)
	if maxRunes < 1 || utf8.RuneCountInString(value) <= maxRunes {
		return value
	}
	runes := []rune(value)
	return string(runes[:maxRunes])
}

func diagnosticTextWasTruncated(original, bounded string) bool {
	return strings.TrimSpace(original) != bounded
}
