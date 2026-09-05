package classification

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

// ErrTokenSpansTruncated marks a token_spans.v1 response the provider declared
// partial through truncated_at. The spans that were returned are valid; the
// caller decides whether a partial scan is acceptable. Treating it as success
// would make a provider that saw half the text indistinguishable from one that
// saw all of it and found nothing, which is the failure class behind #3333 and
// #3364.
var ErrTokenSpansTruncated = errors.New("token_spans.v1 response is partial: provider truncated its input")

// HTTPTokenClassifierInference implements TokenClassifierBackend over the
// token_spans.v1 contract: POST {"inputs": text} and receive spans with
// Unicode code-point offsets into that exact string. The shape mirrors the
// HuggingFace token-classification pipeline with aggregation, the same way
// http_classify mirrors the text-classification pipeline, so a stock service
// works without a shim.
type HTTPTokenClassifierInference struct {
	connector *connector.Client
	timeout   time.Duration
	mapping   *PIIMapping
}

func newHTTPTokenClassifierInference(cfg *config.ExternalModelConfig, mapping *PIIMapping, deadline time.Duration) (*HTTPTokenClassifierInference, error) {
	if cfg == nil {
		return nil, fmt.Errorf("token_spans external model config is required")
	}
	if cfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("token_spans endpoint address is required")
	}
	if mapping == nil || len(mapping.LabelToIdx) == 0 {
		return nil, fmt.Errorf("PII label mapping is required for token_spans")
	}

	scheme := strings.ToLower(strings.TrimSpace(cfg.ModelEndpoint.Protocol))
	if scheme == "" {
		scheme = "http"
	}
	baseURL := fmt.Sprintf("%s://%s:%d", scheme, strings.TrimSpace(cfg.ModelEndpoint.Address), cfg.ModelEndpoint.Port)

	timeout := 5 * time.Second
	if deadline > 0 {
		timeout = deadline
	} else if cfg.TimeoutSeconds > 0 {
		timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	remote, err := connector.New(baseURL, bearerAuthorizer(cfg.AccessKey), connector.Options{
		AttemptTimeout:   timeout,
		MaxRetries:       1,
		MaxRequestBytes:  cfg.GetMaxRequestBytes(),
		MaxResponseBytes: cfg.GetMaxResponseBytes(),
		MaxErrorBytes:    maxClassifyErrorBodyBytes,
	})
	if err != nil {
		return nil, fmt.Errorf("create token_spans connector: %w", err)
	}
	return &HTTPTokenClassifierInference{connector: remote, timeout: timeout, mapping: mapping}, nil
}

// tokenSpanWire is one span as the provider sends it. entity_group and word
// are the HuggingFace pipeline's names for label and text; both spellings are
// accepted. Offsets are code points. The optional byte pair, if present, must
// agree with the code-point pair; this keeps the door open to the RFC #2779
// shape that carries both without making either one a translation.
type tokenSpanWire struct {
	Label       string   `json:"label"`
	EntityGroup string   `json:"entity_group"`
	Score       *float32 `json:"score"`
	Text        *string  `json:"text"`
	Word        *string  `json:"word"`
	Start       *int     `json:"start"`
	End         *int     `json:"end"`
	ByteStart   *int     `json:"byte_start"`
	ByteEnd     *int     `json:"byte_end"`
}

type tokenSpansEnvelope struct {
	Spans       []tokenSpanWire `json:"spans"`
	TruncatedAt *int            `json:"truncated_at"`
	Model       string          `json:"model"`
}

var httpTokenClassifyOperation = connector.Operation{
	Name:      "token_spans",
	Method:    http.MethodPost,
	Path:      "/classify",
	RetrySafe: true,
}

// ClassifyTokens implements TokenClassifierBackend.
func (h *HTTPTokenClassifierInference) ClassifyTokens(text string) ([]candle_binding.TokenEntity, error) {
	return h.classifyTokens(context.Background(), text)
}

func (h *HTTPTokenClassifierInference) classifyTokens(ctx context.Context, text string) ([]candle_binding.TokenEntity, error) {
	ctx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()

	reqBody, err := json.Marshal(httpClassifyRequest{Inputs: text})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal token_spans request: %w", err)
	}
	responseBody, err := h.connector.Do(ctx, httpTokenClassifyOperation, reqBody)
	if err != nil {
		return nil, formatHTTPClassifyConnectorError(err)
	}
	spans, truncatedAt, err := decodeTokenSpansResponse(responseBody)
	if err != nil {
		return nil, err
	}
	return alignTokenSpans(h.mapping, text, spans, truncatedAt)
}

// decodeTokenSpansResponse accepts either a bare JSON list of spans or the
// envelope form that can also carry truncated_at.
func decodeTokenSpansResponse(body []byte) ([]tokenSpanWire, *int, error) {
	trimmed := strings.TrimSpace(string(body))
	if strings.HasPrefix(trimmed, "[") {
		var spans []tokenSpanWire
		if err := json.Unmarshal(body, &spans); err != nil {
			return nil, nil, fmt.Errorf("failed to parse token_spans response: %w", err)
		}
		return spans, nil, nil
	}
	var env tokenSpansEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		return nil, nil, fmt.Errorf("failed to parse token_spans response: %w", err)
	}
	return env.Spans, env.TruncatedAt, nil
}

// alignTokenSpans validates every span against the contract and converts
// code-point offsets to the byte offsets TokenEntity carries internally. Any
// violation rejects the whole response: a provider whose offsets are off by
// one is redacting the wrong characters, and that must not be a warning.
func alignTokenSpans(mapping *PIIMapping, text string, spans []tokenSpanWire, truncatedAt *int) ([]candle_binding.TokenEntity, error) {
	input := newSpanInput(text)
	if err := input.checkTruncatedAt(truncatedAt); err != nil {
		return nil, err
	}

	known := knownPIILabels(mapping)
	seen := make(map[spanKey]struct{}, len(spans))
	entities := make([]candle_binding.TokenEntity, 0, len(spans))
	for i, sp := range spans {
		entity, err := alignTokenSpan(i, sp, input, known, truncatedAt)
		if err != nil {
			return nil, err
		}
		key := spanKey{entity.EntityType, *sp.Start, *sp.End}
		if _, dup := seen[key]; dup {
			return nil, fmt.Errorf("token_spans response contains duplicate span %s [%d,%d)", key.label, key.start, key.end)
		}
		seen[key] = struct{}{}
		entities = append(entities, entity)
	}
	if truncatedAt != nil {
		return entities, ErrTokenSpansTruncated
	}
	return entities, nil
}

// spanKey identifies a span for duplicate detection: same label, same
// code-point range.
type spanKey struct {
	label      string
	start, end int
}

// spanInput is the request text as the contract sees it: code points, plus
// the byte offset of each code point so spans convert to TokenEntity once.
type spanInput struct {
	runes  []rune
	byteAt []int // byteAt[i] is the byte offset of code point i; byteAt[len(runes)] == len(text)
}

func newSpanInput(text string) spanInput {
	runes := []rune(text)
	byteAt := make([]int, len(runes)+1)
	for i, r := range runes {
		byteAt[i+1] = byteAt[i] + utf8.RuneLen(r)
	}
	return spanInput{runes: runes, byteAt: byteAt}
}

func (in spanInput) checkTruncatedAt(truncatedAt *int) error {
	if truncatedAt != nil && (*truncatedAt < 0 || *truncatedAt > len(in.runes)) {
		return fmt.Errorf("token_spans truncated_at %d is outside a %d code-point input", *truncatedAt, len(in.runes))
	}
	return nil
}

// alignTokenSpan validates one span and converts it to a TokenEntity.
func alignTokenSpan(i int, sp tokenSpanWire, input spanInput, known map[string]struct{}, truncatedAt *int) (candle_binding.TokenEntity, error) {
	label, err := spanLabel(i, sp, known)
	if err != nil {
		return candle_binding.TokenEntity{}, err
	}
	start, end, err := spanBounds(i, label, sp, input, truncatedAt)
	if err != nil {
		return candle_binding.TokenEntity{}, err
	}
	text, err := spanText(i, label, sp, input, start, end)
	if err != nil {
		return candle_binding.TokenEntity{}, err
	}
	score, err := spanScore(i, label, sp)
	if err != nil {
		return candle_binding.TokenEntity{}, err
	}
	bStart, bEnd, err := spanBytes(i, label, sp, input, start, end)
	if err != nil {
		return candle_binding.TokenEntity{}, err
	}
	return candle_binding.TokenEntity{
		EntityType: label,
		Start:      bStart,
		End:        bEnd,
		Text:       text,
		Confidence: score,
	}, nil
}

// spanLabel resolves label / entity_group, strips any BIO prefix and checks the
// result against the configured mapping.
func spanLabel(i int, sp tokenSpanWire, known map[string]struct{}) (string, error) {
	label := sp.Label
	if label == "" {
		label = sp.EntityGroup
	}
	label = stripBIOPrefix(label)
	if label == "" {
		return "", fmt.Errorf("token_spans span %d has no label", i)
	}
	if _, ok := known[label]; !ok {
		return "", fmt.Errorf("token_spans span %d label %q is not in the configured PII mapping", i, label)
	}
	return label, nil
}

// spanBounds checks the code-point range and its position relative to any
// declared truncation.
func spanBounds(i int, label string, sp tokenSpanWire, input spanInput, truncatedAt *int) (int, int, error) {
	if sp.Start == nil || sp.End == nil {
		return 0, 0, fmt.Errorf("token_spans span %d (%s) is missing start or end", i, label)
	}
	start, end := *sp.Start, *sp.End
	if start < 0 || start >= end || end > len(input.runes) {
		return 0, 0, fmt.Errorf("token_spans span %d (%s) has offsets [%d,%d) outside a %d code-point input", i, label, start, end, len(input.runes))
	}
	if truncatedAt != nil && end > *truncatedAt {
		return 0, 0, fmt.Errorf("token_spans span %d (%s) ends at %d, after truncated_at %d", i, label, end, *truncatedAt)
	}
	return start, end, nil
}

// spanText resolves text / word and requires it to equal the code-point slice;
// a mismatch is how an off-by-one in the offset unit shows up.
func spanText(i int, label string, sp tokenSpanWire, input spanInput, start, end int) (string, error) {
	var text string
	switch {
	case sp.Text != nil:
		text = *sp.Text
	case sp.Word != nil:
		text = *sp.Word
	default:
		return "", fmt.Errorf("token_spans span %d (%s) is missing text", i, label)
	}
	if got := string(input.runes[start:end]); got != text {
		return "", fmt.Errorf("token_spans span %d (%s) text %q does not match input [%d,%d) %q; check the offset unit", i, label, text, start, end, got)
	}
	return text, nil
}

func spanScore(i int, label string, sp tokenSpanWire) (float32, error) {
	if sp.Score == nil || *sp.Score < 0 || *sp.Score > 1 {
		return 0, fmt.Errorf("token_spans span %d (%s) score is missing or outside [0,1]", i, label)
	}
	return *sp.Score, nil
}

// spanBytes converts the code-point range to bytes and, if the provider also
// sent a byte pair, requires it to agree.
func spanBytes(i int, label string, sp tokenSpanWire, input spanInput, start, end int) (int, int, error) {
	bStart, bEnd := input.byteAt[start], input.byteAt[end]
	if sp.ByteStart != nil && *sp.ByteStart != bStart {
		return 0, 0, fmt.Errorf("token_spans span %d (%s) byte_start %d disagrees with code-point start %d (byte %d)", i, label, *sp.ByteStart, start, bStart)
	}
	if sp.ByteEnd != nil && *sp.ByteEnd != bEnd {
		return 0, 0, fmt.Errorf("token_spans span %d (%s) byte_end %d disagrees with code-point end %d (byte %d)", i, label, *sp.ByteEnd, end, bEnd)
	}
	return bStart, bEnd, nil
}

// knownPIILabels collects the mapping's label names with any BIO prefix
// removed, so a provider may say PERSON whether the mapping file was written
// as PERSON or B-PERSON.
func knownPIILabels(mapping *PIIMapping) map[string]struct{} {
	known := make(map[string]struct{})
	if mapping == nil {
		return known
	}
	for label := range mapping.LabelToIdx {
		known[stripBIOPrefix(label)] = struct{}{}
	}
	for _, label := range mapping.IdxToLabel {
		known[stripBIOPrefix(label)] = struct{}{}
	}
	return known
}

// Close releases idle connections owned by the remote connector.
func (h *HTTPTokenClassifierInference) Close() error {
	if h == nil || h.connector == nil {
		return nil
	}
	return h.connector.Close()
}
