package classification

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"reflect"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// probabilitySumTolerance bounds how far an http_classify response's scores
// may drift from summing to 1.0, tolerating floating-point rounding from
// whatever serialized them without accepting a response with an
// unaccounted-for probability mass.
const probabilitySumTolerance = 0.02

// sequenceLabelMapping is the minimal label<->index contract
// alignScoresToMapping/assignScoreToMapping need to validate an http_classify
// response against a classifier's configured label mapping. JailbreakMapping
// and CategoryMapping (mapping.go) both satisfy it via thin wrapper methods,
// so category's http_classify backend (#2760) can reuse the same validator
// jailbreak uses instead of duplicating it.
type sequenceLabelMapping interface {
	// IndexForLabel returns the configured class index for a label name.
	IndexForLabel(label string) (int, bool)
	// LabelFromIndex returns the label name for a configured class index.
	LabelFromIndex(classIndex int) (string, bool)
	// LabelCount returns the number of configured labels.
	LabelCount() int
}

// HTTPClassifierInference implements SequenceClassifierBackend by calling an
// external sequence-classifier endpoint over HTTP. The wire contract mirrors
// the widely-used HuggingFace text-classification pipeline shape: a request
// carrying the input text, and a response listing every label's score (not
// just the top prediction), so a server wrapping an existing `transformers`
// pipeline or a Text Embeddings Inference (TEI) classify endpoint can be
// plugged in with minimal glue:
//
//	POST {endpoint}/classify
//	  {"inputs": "<text>"}
//	  -> [{"label": "safe", "score": 0.99}, {"label": "jailbreak", "score": 0.01}]
//
// Response labels are matched against the configured label mapping by name
// (not by array position, which the server is free to order however it
// likes - e.g. sorted by score) so the resulting class index always lines up
// with the same mapping every other backend uses. This type is not
// jailbreak-specific: any SequenceClassifierBackend consumer whose mapping
// satisfies sequenceLabelMapping (JailbreakMapping, CategoryMapping, ...)
// can construct one - validated against a real, independently-trained
// external model (a 14-label category classifier) during #2918, not just a
// synthetic mock.
type HTTPClassifierInference struct {
	httpClient *http.Client
	baseURL    string
	accessKey  string
	timeout    time.Duration
	mapping    sequenceLabelMapping
}

// NewHTTPClassifierInference creates a new http_classify-backed inference
// instance from an external model config and label mapping.
func NewHTTPClassifierInference(cfg *config.ExternalModelConfig, mapping sequenceLabelMapping) (*HTTPClassifierInference, error) {
	if cfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("http_classify endpoint address is required")
	}
	if isNilMapping(mapping) {
		return nil, fmt.Errorf("label mapping is required for http_classify")
	}

	scheme := cfg.ModelEndpoint.Protocol
	if scheme == "" {
		scheme = "http"
	}
	baseURL := fmt.Sprintf("%s://%s:%d", scheme, cfg.ModelEndpoint.Address, cfg.ModelEndpoint.Port)

	// http_classify is a single lightweight forward pass, not a generative
	// call - it should fail fast rather than share http_chat's more
	// generous default.
	timeout := 5 * time.Second
	if cfg.TimeoutSeconds > 0 {
		timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	return &HTTPClassifierInference{
		httpClient: &http.Client{Timeout: timeout},
		baseURL:    baseURL,
		accessKey:  cfg.AccessKey,
		timeout:    timeout,
		mapping:    mapping,
	}, nil
}

// isNilMapping reports whether mapping is nil - either a literal nil
// interface, or an interface holding a nil concrete pointer (e.g. a
// (*JailbreakMapping)(nil) passed by a caller that forgot to check). A plain
// `mapping == nil` comparison misses the second case: an interface value
// carrying type information is never == nil even when the pointer it wraps
// is, so a caller's nil *JailbreakMapping would otherwise slip past this
// check and panic later inside LabelCount/IndexForLabel instead of failing
// fast here.
func isNilMapping(mapping sequenceLabelMapping) bool {
	if mapping == nil {
		return true
	}
	v := reflect.ValueOf(mapping)
	return v.Kind() == reflect.Ptr && v.IsNil()
}

type httpClassifyRequest struct {
	Inputs string `json:"inputs"`
}

type httpClassifyLabelScore struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

// Classify implements the SequenceClassifierBackend interface. It derives its
// deadline from the caller's ctx (so the request can be cancelled if the
// caller gives up first) bounded by h.timeout, rather than always running to
// its own internal timeout regardless of the caller's lifecycle.
func (h *HTTPClassifierInference) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	ctx, cancel := context.WithTimeout(ctx, h.timeout)
	defer cancel()

	httpReq, err := h.buildClassifyRequest(ctx, text)
	if err != nil {
		return SequenceClassificationResult{}, err
	}

	scores, err := h.doClassifyRequest(httpReq)
	if err != nil {
		return SequenceClassificationResult{}, err
	}

	return alignScoresToMapping(h.mapping, scores)
}

// buildClassifyRequest builds the outgoing http_classify HTTP request.
func (h *HTTPClassifierInference) buildClassifyRequest(ctx context.Context, text string) (*http.Request, error) {
	reqBody, err := json.Marshal(httpClassifyRequest{Inputs: text})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal http_classify request: %w", err)
	}

	url := fmt.Sprintf("%s/classify", h.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create http_classify request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if h.accessKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", h.accessKey))
	}
	return httpReq, nil
}

// doClassifyRequest sends the request and parses the label/score list from a
// successful response.
func (h *HTTPClassifierInference) doClassifyRequest(httpReq *http.Request) ([]httpClassifyLabelScore, error) {
	resp, err := h.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http_classify request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read http_classify response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("http_classify endpoint returned status %d: %s", resp.StatusCode, string(body))
	}

	var scores []httpClassifyLabelScore
	if err := json.Unmarshal(body, &scores); err != nil {
		return nil, fmt.Errorf("failed to parse http_classify response: %w", err)
	}
	if len(scores) == 0 {
		return nil, fmt.Errorf("http_classify response contained no labels")
	}
	return scores, nil
}

// alignScoresToMapping matches response labels against the configured label
// mapping by name (not by array position) and builds the full-distribution
// result. It requires the response to be a complete, valid distribution over
// exactly the configured labels - not just "at least one label matched" -
// because an incomplete response (e.g. a top-k response that omits the
// positive label) would otherwise default the missing entries to 0.0 and
// silently under-report risk instead of surfacing the mismatch. mapping may
// be any classifier's label mapping (JailbreakMapping, CategoryMapping, ...)
// that satisfies sequenceLabelMapping.
func alignScoresToMapping(mapping sequenceLabelMapping, scores []httpClassifyLabelScore) (SequenceClassificationResult, error) {
	numClasses := mapping.LabelCount()
	probabilities := make([]float32, numClasses)
	seenIdx := make([]bool, numClasses)
	seenLabels := make([]string, 0, len(scores))
	var sum float32

	for _, s := range scores {
		seenLabels = append(seenLabels, s.Label)
		idx, err := assignScoreToMapping(mapping, seenIdx, s)
		if err != nil {
			return SequenceClassificationResult{}, err
		}
		probabilities[idx] = s.Score
		sum += s.Score
	}

	for idx, present := range seenIdx {
		if present {
			continue
		}
		missingLabel, _ := mapping.LabelFromIndex(idx)
		return SequenceClassificationResult{}, fmt.Errorf(
			"http_classify response is missing label %q from the configured label mapping (got %v)", missingLabel, seenLabels)
	}

	if math.Abs(float64(sum)-1.0) > probabilitySumTolerance {
		return SequenceClassificationResult{}, fmt.Errorf(
			"http_classify response scores sum to %v, want ~1.0 (labels: %v)", sum, seenLabels)
	}

	return SequenceClassificationResult{Probabilities: probabilities}, nil
}

// assignScoreToMapping validates a single response label/score pair against
// the configured label mapping and marks its index as seen, returning the
// resolved index. Split out of alignScoresToMapping to keep its cyclomatic
// complexity within the repo's lint gate.
func assignScoreToMapping(mapping sequenceLabelMapping, seenIdx []bool, s httpClassifyLabelScore) (int, error) {
	idx, ok := mapping.IndexForLabel(s.Label)
	if !ok || idx < 0 || idx >= len(seenIdx) {
		return 0, fmt.Errorf(
			"http_classify response label %q is not in the configured label mapping", s.Label)
	}
	if seenIdx[idx] {
		return 0, fmt.Errorf("http_classify response contains duplicate label %q", s.Label)
	}
	if score64 := float64(s.Score); math.IsNaN(score64) || math.IsInf(score64, 0) || s.Score < 0 || s.Score > 1 {
		return 0, fmt.Errorf(
			"http_classify response label %q has an invalid score %v (must be finite and within [0, 1])", s.Label, s.Score)
	}
	seenIdx[idx] = true
	return idx, nil
}
