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
	runes := []rune(text)
	// byteAt[i] is the byte offset of code point i; byteAt[len(runes)] == len(text).
	byteAt := make([]int, len(runes)+1)
	for i, r := range runes {
		byteAt[i+1] = byteAt[i] + utf8.RuneLen(r)
	}
	if truncatedAt != nil && (*truncatedAt < 0 || *truncatedAt > len(runes)) {
		return nil, fmt.Errorf("token_spans truncated_at %d is outside a %d code-point input", *truncatedAt, len(runes))
	}

	known := knownPIILabels(mapping)
	type spanKey struct {
		label      string
		start, end int
	}
	seen := make(map[spanKey]struct{}, len(spans))
	entities := make([]candle_binding.TokenEntity, 0, len(spans))

	for i, sp := range spans {
		label := sp.Label
		if label == "" {
			label = sp.EntityGroup
		}
		label = stripBIOPrefix(label)
		if label == "" {
			return nil, fmt.Errorf("token_spans span %d has no label", i)
		}
		if _, ok := known[label]; !ok {
			return nil, fmt.Errorf("token_spans span %d label %q is not in the configured PII mapping", i, label)
		}
		if sp.Start == nil || sp.End == nil {
			return nil, fmt.Errorf("token_spans span %d (%s) is missing start or end", i, label)
		}
		start, end := *sp.Start, *sp.End
		if start < 0 || start >= end || end > len(runes) {
			return nil, fmt.Errorf("token_spans span %d (%s) has offsets [%d,%d) outside a %d code-point input", i, label, start, end, len(runes))
		}
		if truncatedAt != nil && end > *truncatedAt {
			return nil, fmt.Errorf("token_spans span %d (%s) ends at %d, after truncated_at %d", i, label, end, *truncatedAt)
		}
		var spanText string
		switch {
		case sp.Text != nil:
			spanText = *sp.Text
		case sp.Word != nil:
			spanText = *sp.Word
		default:
			return nil, fmt.Errorf("token_spans span %d (%s) is missing text", i, label)
		}
		if got := string(runes[start:end]); got != spanText {
			return nil, fmt.Errorf("token_spans span %d (%s) text %q does not match input [%d,%d) %q; check the offset unit", i, label, spanText, start, end, got)
		}
		if sp.Score == nil || *sp.Score < 0 || *sp.Score > 1 {
			return nil, fmt.Errorf("token_spans span %d (%s) score is missing or outside [0,1]", i, label)
		}
		bStart, bEnd := byteAt[start], byteAt[end]
		if sp.ByteStart != nil && *sp.ByteStart != bStart {
			return nil, fmt.Errorf("token_spans span %d (%s) byte_start %d disagrees with code-point start %d (byte %d)", i, label, *sp.ByteStart, start, bStart)
		}
		if sp.ByteEnd != nil && *sp.ByteEnd != bEnd {
			return nil, fmt.Errorf("token_spans span %d (%s) byte_end %d disagrees with code-point end %d (byte %d)", i, label, *sp.ByteEnd, end, bEnd)
		}
		k := spanKey{label, start, end}
		if _, dup := seen[k]; dup {
			return nil, fmt.Errorf("token_spans response contains duplicate span %s [%d,%d)", label, start, end)
		}
		seen[k] = struct{}{}
		entities = append(entities, candle_binding.TokenEntity{
			EntityType: label,
			Start:      bStart,
			End:        bEnd,
			Text:       spanText,
			Confidence: *sp.Score,
		})
	}
	if truncatedAt != nil {
		return entities, ErrTokenSpansTruncated
	}
	return entities, nil
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
