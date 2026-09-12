package classification

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// ErrTokenSpansTruncated marks a token_spans.v1 response the provider declared
// partial through truncated_at. The spans that were returned are valid; the
// caller decides whether a partial scan is acceptable. Treating it as success
// would make a provider that saw half the text indistinguishable from one that
// saw all of it and found nothing, which is the failure class behind #3333 and
// #3364.
var ErrTokenSpansTruncated = tasks.ErrTokenSpansTruncated

// HTTPTokenClassifierInference implements TokenClassifierBackend over the
// token_spans.v1 contract: POST {"inputs": text} and receive spans with
// Unicode code-point offsets into that exact string. The shape mirrors the
// HuggingFace token-classification pipeline with aggregation, the same way
// http_classify mirrors the text-classification pipeline, so a stock service
// works without a shim.
type HTTPTokenClassifierInference struct {
	connector *connector.Client
	timeout   time.Duration
	// known is the mapping's label set with BIO prefixes stripped, built once
	// so each response is checked against a ready map. outside holds the
	// no-entity labels (class zero and "O"), which a provider must never send
	// as spans: the native path drops class zero before it becomes an entity,
	// and a remote O span reaching PIIEntities or masking would break parity.
	known   map[string]struct{}
	outside map[string]struct{}
	// model is the configured llm_model_name of the external model. A response
	// envelope may name the model that produced it; when it does, it must be
	// this one, so a reply from a differently deployed model cannot pass as
	// valid.
	model string
}

func newHTTPTokenClassifierInference(cfg *config.ExternalModelConfig, labels tasks.TokenLabelSet, deadline time.Duration) (*HTTPTokenClassifierInference, error) {
	if cfg == nil {
		return nil, fmt.Errorf("token_spans external model config is required")
	}
	if cfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("token_spans endpoint address is required")
	}
	known, outside, err := compileTokenLabels(labels)
	if err != nil {
		return nil, err
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
	return &HTTPTokenClassifierInference{connector: remote, timeout: timeout, known: known, outside: outside, model: strings.TrimSpace(cfg.ModelName)}, nil
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

// tokenSpansEnvelope is the object form. Spans and Error stay raw so that a
// missing member, a null and a non-array are distinguishable from an empty
// list; the decoder decides what each means.
type tokenSpansEnvelope struct {
	Spans       json.RawMessage `json:"spans"`
	TruncatedAt *int            `json:"truncated_at"`
	Error       json.RawMessage `json:"error"`
	Model       string          `json:"model"`
}

var httpTokenClassifyOperation = connector.Operation{
	Name:      "token_spans",
	Method:    http.MethodPost,
	Path:      "/classify",
	RetrySafe: true,
}

// ClassifyTokens implements TokenClassifierBackend.
func (h *HTTPTokenClassifierInference) ClassifyTokens(ctx context.Context, text string) (tasks.TokenClassificationResult, error) {
	return h.classifyTokenResult(ctx, text)
}

func (h *HTTPTokenClassifierInference) classifyTokens(ctx context.Context, text string) ([]tasks.TokenEntity, error) {
	result, err := h.classifyTokenResult(ctx, text)
	return result.Entities, err
}

func (h *HTTPTokenClassifierInference) classifyTokenResult(ctx context.Context, text string) (tasks.TokenClassificationResult, error) {
	if !utf8.ValidString(text) {
		return tasks.TokenClassificationResult{}, fmt.Errorf("token_spans input is not valid UTF-8")
	}
	ctx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()

	reqBody, err := json.Marshal(httpClassifyRequest{Inputs: text})
	if err != nil {
		return tasks.TokenClassificationResult{}, fmt.Errorf("failed to marshal token_spans request: %w", err)
	}
	responseBody, err := h.connector.Do(ctx, httpTokenClassifyOperation, reqBody)
	if err != nil {
		return tasks.TokenClassificationResult{}, formatHTTPClassifyConnectorError(err)
	}
	spans, truncatedAt, model, err := decodeTokenSpansResponse(responseBody)
	if err != nil {
		return tasks.TokenClassificationResult{}, err
	}
	if identityErr := h.checkModelIdentity(model); identityErr != nil {
		return tasks.TokenClassificationResult{}, identityErr
	}
	entities, err := alignTokenSpans(h.known, h.outside, text, spans, truncatedAt)
	scoresAvailable := true
	result := tasks.TokenClassificationResult{Entities: entities, ScoresAvailable: &scoresAvailable}
	if truncatedAt != nil && (err == nil || errors.Is(err, ErrTokenSpansTruncated)) {
		byteOffset := newSpanInput(text).byteAt[*truncatedAt]
		result.TruncatedAt = &byteOffset
	}
	return result, err
}

// checkModelIdentity enforces the model-identity rule of token_spans.v1: the
// envelope's model member is optional, but when present it must equal the
// configured llm_model_name. The bare-list form carries no model and is
// accepted as is.
func (h *HTTPTokenClassifierInference) checkModelIdentity(model string) error {
	model = strings.TrimSpace(model)
	if model == "" || h.model == "" || model == h.model {
		return nil
	}
	// The response's model name is provider text and is not echoed; the
	// configured name is ours and is.
	return fmt.Errorf("token_spans response names a different model (%d bytes) than the configured %q", len(model), h.model)
}

// decodeTokenSpansResponse accepts either a bare JSON list of spans, which is
// what a stock HuggingFace token-classification pipeline returns, or the
// envelope form that can also carry truncated_at and model. Everything else is
// rejected: an empty body, a bare object without a spans array, {"spans": null},
// a spans member that is not an array, or a 200 that carries an error member.
// None of those may become a clean zero-entity result, because PII would then
// treat a broken provider as text with nothing to redact.
func decodeTokenSpansResponse(body []byte) ([]tokenSpanWire, *int, string, error) {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return nil, nil, "", fmt.Errorf("token_spans response is empty")
	}
	if trimmed[0] == '[' {
		var spans []tokenSpanWire
		if err := json.Unmarshal(trimmed, &spans); err != nil {
			return nil, nil, "", fmt.Errorf("failed to parse token_spans response: %w", err)
		}
		return spans, nil, "", nil
	}
	if trimmed[0] != '{' {
		return nil, nil, "", fmt.Errorf("token_spans response must be a JSON array or object")
	}
	var env tokenSpansEnvelope
	if err := json.Unmarshal(trimmed, &env); err != nil {
		return nil, nil, "", fmt.Errorf("failed to parse token_spans response: %w", err)
	}
	if isPresentJSON(env.Error) {
		// Never interpolate the provider's error payload: it routinely echoes
		// the text it was asked to classify, and this error reaches logs.
		return nil, nil, "", fmt.Errorf("token_spans provider reported an error (%d bytes, not logged)", len(bytes.TrimSpace(env.Error)))
	}
	if !isPresentJSON(env.Spans) {
		return nil, nil, "", fmt.Errorf("token_spans response has no spans array")
	}
	spansRaw := bytes.TrimSpace(env.Spans)
	if spansRaw[0] != '[' {
		return nil, nil, "", fmt.Errorf("token_spans spans member must be an array")
	}
	var spans []tokenSpanWire
	if err := json.Unmarshal(spansRaw, &spans); err != nil {
		return nil, nil, "", fmt.Errorf("failed to parse token_spans spans: %w", err)
	}
	return spans, env.TruncatedAt, env.Model, nil
}

// isPresentJSON reports whether a raw member was sent with a value other than
// null. A missing member and an explicit null both count as absent.
func isPresentJSON(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	return len(trimmed) > 0 && !bytes.Equal(trimmed, []byte("null"))
}

// alignTokenSpans validates every span against the contract and converts
// code-point offsets to the byte offsets TokenEntity carries internally. Any
// violation rejects the whole response: a provider whose offsets are off by
// one is redacting the wrong characters, and that must not be a warning.
func alignTokenSpans(known, outside map[string]struct{}, text string, spans []tokenSpanWire, truncatedAt *int) ([]tasks.TokenEntity, error) {
	input := newSpanInput(text)
	if err := input.checkTruncatedAt(truncatedAt); err != nil {
		return nil, err
	}

	seen := make(map[spanKey]struct{}, len(spans))
	entities := make([]tasks.TokenEntity, 0, len(spans))
	for i, sp := range spans {
		entity, err := alignTokenSpan(i, sp, input, known, outside, truncatedAt)
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
func alignTokenSpan(i int, sp tokenSpanWire, input spanInput, known, outside map[string]struct{}, truncatedAt *int) (tasks.TokenEntity, error) {
	label, err := spanLabel(i, sp, known, outside)
	if err != nil {
		return tasks.TokenEntity{}, err
	}
	start, end, err := spanBounds(i, label, sp, input, truncatedAt)
	if err != nil {
		return tasks.TokenEntity{}, err
	}
	text, err := spanText(i, label, sp, input, start, end)
	if err != nil {
		return tasks.TokenEntity{}, err
	}
	score, err := spanScore(i, label, sp)
	if err != nil {
		return tasks.TokenEntity{}, err
	}
	bStart, bEnd, err := spanBytes(i, label, sp, input, start, end)
	if err != nil {
		return tasks.TokenEntity{}, err
	}
	return tasks.TokenEntity{
		EntityType: label,
		Start:      bStart,
		End:        bEnd,
		Text:       text,
		Confidence: score,
	}, nil
}

// spanLabel resolves label / entity_group, strips any BIO prefix and checks the
// result against the configured mapping. The outside label is rejected before
// the known-label check: it is in the mapping, but it names the absence of an
// entity, and the native backend never emits it as a span.
func spanLabel(i int, sp tokenSpanWire, known, outside map[string]struct{}) (string, error) {
	if sp.Label != "" && sp.EntityGroup != "" && stripBIOPrefix(sp.Label) != stripBIOPrefix(sp.EntityGroup) {
		return "", fmt.Errorf("token_spans span %d has conflicting label and entity_group", i)
	}
	label := sp.Label
	if label == "" {
		label = sp.EntityGroup
	}
	label = stripBIOPrefix(label)
	if label == "" {
		return "", fmt.Errorf("token_spans span %d has no label", i)
	}
	if _, isOutside := outside[label]; isOutside {
		return "", fmt.Errorf("token_spans span %d carries the outside label %q; providers send entity spans only", i, label)
	}
	if _, ok := known[label]; !ok {
		// An unknown label is provider text; report its size, not its value.
		return "", fmt.Errorf("token_spans span %d label (%d bytes) is not in the configured task label mapping", i, len(label))
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
	if sp.Text != nil && sp.Word != nil && *sp.Text != *sp.Word {
		return "", fmt.Errorf("token_spans span %d (%s) has conflicting text (%d code points) and word (%d code points)", i, label, len([]rune(*sp.Text)), len([]rune(*sp.Word)))
	}
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
		// Neither the span text nor the input slice is echoed: both are user
		// content, and this error reaches logs.
		return "", fmt.Errorf("token_spans span %d (%s) text (%d code points) does not match input [%d,%d) (%d code points); check the offset unit", i, label, len([]rune(text)), start, end, len([]rune(got)))
	}
	return text, nil
}

func spanScore(i int, label string, sp tokenSpanWire) (float32, error) {
	if sp.Score == nil || math.IsNaN(float64(*sp.Score)) || math.IsInf(float64(*sp.Score), 0) || *sp.Score < 0 || *sp.Score > 1 {
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

// Close releases idle connections owned by the remote connector.
func (h *HTTPTokenClassifierInference) Close() error {
	if h == nil || h.connector == nil {
		return nil
	}
	return h.connector.Close()
}
