package classification

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	systemPromptBase = `You are an expert annotator who identifies hallucinated spans in a generated answer with respect to a given context (the only trusted evidence). A hallucinated span is a substring of the answer that is not supported by the context. Spans consistent with the context are not hallucinations. Quote each hallucinated span verbatim from the answer and classify it into exactly one category and one subcategory.
Categories (the kinds of unsupported span):
- contradiction: conflicts with the context (a wrong value, number, date, name, or relationship)
- fabricated_reference: an entity, name, identifier, or section that is absent from the context
- unsupported_addition: a claim, detail, or behavior the context never states
Subcategories:
- entity: a wrong or invented name, entity, or object
- temporal: an incorrect date, time, duration, or ordering
- numerical: an incorrect number, quantity, or amount
- value: a wrong value, setting, or attribute value
- relational: an incorrect relationship or association between things
- identifier: an invented identifier or name not found in the context
- section: a reference to a section, part, or location that does not exist
- attribute: an invented or incorrect attribute or property
- claim: an added factual claim the context does not support
- behavior: an added or changed action or behavior the context never states
- elaboration: extra detail or elaboration beyond what the context supports
- subjective: an unsupported subjective or evaluative statement
- unspecified: unsupported, with no more specific subtype

Reply with ONLY a JSON object (no markdown, no code fences):
{"hallucinated_spans": [{"text": "...", "category": "...", "subcategory": "..."}]}.
If nothing is unsupported, reply {"hallucinated_spans": []}.`

	systemPromptExpl = `You are an expert annotator who identifies hallucinated spans in a generated answer with respect to a given context (the only trusted evidence). A hallucinated span is a substring of the answer that is not supported by the context. Spans consistent with the context are not hallucinations. Quote each hallucinated span verbatim from the answer and classify it into exactly one category and one subcategory, and give a short explanation of why it is unsupported.
Categories (the kinds of unsupported span):
- contradiction: conflicts with the context (a wrong value, number, date, name, or relationship)
- fabricated_reference: an entity, name, identifier, or section that is absent from the context
- unsupported_addition: a claim, detail, or behavior the context never states
Subcategories:
- entity: a wrong or invented name, entity, or object
- temporal: an incorrect date, time, duration, or ordering
- numerical: an incorrect number, quantity, or amount
- value: a wrong value, setting, or attribute value
- relational: an incorrect relationship or association between things
- identifier: an invented identifier or name not found in the context
- section: a reference to a section, part, or location that does not exist
- attribute: an invented or incorrect attribute or property
- claim: an added factual claim the context does not support
- behavior: an added or changed action or behavior the context never states
- elaboration: extra detail or elaboration beyond what the context supports
- subjective: an unsupported subjective or evaluative statement
- unspecified: unsupported, with no more specific subtype

Reply with ONLY a JSON object (no markdown, no code fences):
{"hallucinated_spans": [{"text": "...", "category": "...", "subcategory": "...", "explanation": "..."}]}.
If nothing is unsupported, reply {"hallucinated_spans": []}.`
)

// endpointCategories and endpointSubcategories are the hallucination taxonomy
// advertised in the structured-output schema. They are the single source of truth
// for both the request schema and local validation of the returned spans.
var (
	endpointCategories    = []string{"contradiction", "fabricated_reference", "unsupported_addition"}
	endpointSubcategories = []string{"entity", "temporal", "numerical", "value", "relational", "identifier", "section", "attribute", "claim", "behavior", "elaboration", "subjective", "unspecified"}
)

// normalizeTaxonomyValue lower-cases and trims a returned taxonomy value and
// returns it only when it is a member of the allowed set; otherwise it returns "".
func normalizeTaxonomyValue(value string, allowed []string) string {
	normalized := strings.ToLower(strings.TrimSpace(value))
	for _, candidate := range allowed {
		if normalized == candidate {
			return normalized
		}
	}
	return ""
}

type EndpointHallucinationDetector struct {
	config      *config.HallucinationModelConfig
	initialized bool
	mu          sync.RWMutex
	client      *http.Client
	endpoint    string
}

func NewEndpointHallucinationDetector(cfg *config.HallucinationModelConfig) (*EndpointHallucinationDetector, error) {
	if cfg == nil {
		return nil, fmt.Errorf("hallucination model config is required")
	}
	if cfg.Endpoint == "" {
		return nil, fmt.Errorf("hallucination endpoint is required when backend is endpoint")
	}
	if cfg.ModelID == "" {
		return nil, fmt.Errorf("hallucination model_id is required")
	}

	return &EndpointHallucinationDetector{
		config:   cfg,
		client:   &http.Client{Timeout: 10 * time.Second},
		endpoint: strings.TrimRight(cfg.Endpoint, "/"),
	}, nil
}

func (d *EndpointHallucinationDetector) Initialize() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.initialized {
		return nil
	}
	d.initialized = true
	logging.ComponentEvent("classifier", "hallucination_detector_initialized", map[string]interface{}{
		"backend":   "endpoint",
		"model_ref": d.config.ModelID,
		"endpoint":  d.endpoint,
	})
	return nil
}

func (d *EndpointHallucinationDetector) IsInitialized() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.initialized
}

// IsNLIInitialized always returns false for the endpoint backend. The endpoint
// does not ship a local NLI explainer model, and HasHallucinationExplainer /
// the /classify/nli readiness APIs specifically represent that Candle-only NLI
// capability. Advertising NLI readiness here would let those APIs report ready
// while the NLI path fails. The endpoint's own generative explanation still flows
// through DetectWithNLI independently of this flag.
func (d *EndpointHallucinationDetector) IsNLIInitialized() bool {
	return false
}

func (d *EndpointHallucinationDetector) buildRequestPayload(reqContext, question, answer string) ([]byte, error) {
	prompt := fmt.Sprintf("User request: %s\n\nContext (trusted evidence):\n%s\n\nAnswer to verify:\n%s", question, reqContext, answer)

	systemPrompt := systemPromptBase
	if d.config.IncludeExplanation {
		systemPrompt = systemPromptExpl
	}

	spanProps := map[string]interface{}{
		"text":        map[string]interface{}{"type": "string"},
		"category":    map[string]interface{}{"type": "string", "enum": endpointCategories},
		"subcategory": map[string]interface{}{"type": "string", "enum": endpointSubcategories},
	}
	requiredFields := []string{"text", "category", "subcategory"}

	if d.config.IncludeExplanation {
		spanProps["explanation"] = map[string]interface{}{"type": "string"}
		requiredFields = append(requiredFields, "explanation")
	}

	reqBody := map[string]interface{}{
		"model":       d.config.ModelID,
		"temperature": 0.0,
		"stream":      false,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": prompt},
		},
		"response_format": map[string]interface{}{
			"type": "json_schema",
			"json_schema": map[string]interface{}{
				"name": "hallucination_detection",
				"schema": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"hallucinated_spans": map[string]interface{}{
							"type": "array",
							"items": map[string]interface{}{
								"type":                 "object",
								"properties":           spanProps,
								"required":             requiredFields,
								"additionalProperties": false,
							},
						},
					},
					"required":             []string{"hallucinated_spans"},
					"additionalProperties": false,
				},
				"strict": true,
			},
		},
	}
	return json.Marshal(reqBody)
}

// parseOpenAIResponse parses the OpenAI-compatible response and maps the returned
// spans onto backend-neutral EnhancedHallucinationSpan values. It returns an error
// for any malformed response so the caller can fail open via the detection_error
// path instead of recording a false clean verdict. The taxonomy is validated
// locally, each span is verified to be a substring of the answer, and deterministic
// Start/End offsets are populated. NLI fields are left UNKNOWN/0 because the
// endpoint backend does not produce NLI labels.
func (d *EndpointHallucinationDetector) parseOpenAIResponse(respBytes []byte, answer string) ([]EnhancedHallucinationSpan, error) {
	var openaiResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(respBytes, &openaiResp); err != nil {
		return nil, fmt.Errorf("response parsing failed: %w", err)
	}
	if len(openaiResp.Choices) == 0 {
		return nil, fmt.Errorf("response contained no choices")
	}

	var parsed struct {
		HallucinatedSpans []struct {
			Text        string `json:"text"`
			Category    string `json:"category"`
			Subcategory string `json:"subcategory"`
			Explanation string `json:"explanation"`
		} `json:"hallucinated_spans"`
	}

	if err := json.Unmarshal([]byte(openaiResp.Choices[0].Message.Content), &parsed); err != nil {
		return nil, fmt.Errorf("JSON schema parsing failed: %w", err)
	}

	spans := make([]EnhancedHallucinationSpan, 0, len(parsed.HallucinatedSpans))
	for _, s := range parsed.HallucinatedSpans {
		if s.Text == "" {
			continue
		}
		// The span must be quoted verbatim from the answer; skip anything that is
		// not actually present so we never emit fabricated offsets.
		start := strings.Index(answer, s.Text)
		if start < 0 {
			logging.Debugf("Endpoint hallucination span not found in answer, skipping: %q", s.Text)
			continue
		}

		category := normalizeTaxonomyValue(s.Category, endpointCategories)
		subcategory := normalizeTaxonomyValue(s.Subcategory, endpointSubcategories)

		spans = append(spans, EnhancedHallucinationSpan{
			Text:                    s.Text,
			Start:                   start,
			End:                     start + len(s.Text),
			HallucinationConfidence: 1.0,
			NLILabel:                0,
			NLILabelStr:             "UNKNOWN",
			NLIConfidence:           0,
			Severity:                2,
			Explanation:             endpointSpanExplanation(s.Explanation, category, subcategory),
		})
	}
	return spans, nil
}

// endpointSpanExplanation builds a backend-neutral explanation, preferring the
// model-supplied explanation and falling back to the validated taxonomy.
func endpointSpanExplanation(explanation, category, subcategory string) string {
	if explanation != "" {
		return explanation
	}
	switch {
	case category != "" && subcategory != "":
		return fmt.Sprintf("Unsupported span (%s / %s) detected by endpoint detector", category, subcategory)
	case category != "":
		return fmt.Sprintf("Unsupported span (%s) detected by endpoint detector", category)
	default:
		return "Unsupported span detected by endpoint detector"
	}
}

// DetectWithNLI runs a single structured detection call against the endpoint.
// It fails open by returning an error (not a clean verdict) for any transport,
// status, read, or parse failure: the response filter already passes traffic
// through on error and records the detection_error path rather than not_detected.
// A clean result is reserved for an empty answer (nothing to verify) and for a
// successfully parsed empty span list.
func (d *EndpointHallucinationDetector) DetectWithNLI(reqContext, question, answer string) (*EnhancedHallucinationResult, error) {
	if answer == "" {
		return d.cleanResult(), nil
	}
	if reqContext == "" {
		return nil, fmt.Errorf("context is required for hallucination detection")
	}

	bodyBytes, err := d.buildRequestPayload(reqContext, question, answer)
	if err != nil {
		return nil, fmt.Errorf("failed to build hallucination detection request: %w", err)
	}
	req, err := http.NewRequestWithContext(context.Background(), "POST", d.endpoint+"/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create hallucination detection request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("endpoint hallucination detection request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("endpoint hallucination detection returned non-200 status: %d", resp.StatusCode)
	}

	// Cap the response read at 10MB to avoid unbounded memory use from a
	// misbehaving endpoint.
	respBytes, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		return nil, fmt.Errorf("endpoint hallucination detection response read failed: %w", err)
	}

	spans, err := d.parseOpenAIResponse(respBytes, answer)
	if err != nil {
		return nil, fmt.Errorf("endpoint hallucination detection parse failed: %w", err)
	}

	return &EnhancedHallucinationResult{
		HallucinationDetected: len(spans) > 0,
		Confidence:            1.0,
		Spans:                 spans,
	}, nil
}

// cleanResult is the "nothing to verify" verdict, used only when the answer is
// empty. Endpoint failures return an error instead so they are never recorded as
// a clean (not_detected) verdict.
func (d *EndpointHallucinationDetector) cleanResult() *EnhancedHallucinationResult {
	return &EnhancedHallucinationResult{
		HallucinationDetected: false,
		Confidence:            1.0,
		Spans:                 []EnhancedHallucinationSpan{},
	}
}

func (d *EndpointHallucinationDetector) Detect(reqContext, question, answer string) (*HallucinationResult, error) {
	enhanced, err := d.DetectWithNLI(reqContext, question, answer)
	if err != nil {
		return nil, err
	}

	res := &HallucinationResult{
		HallucinationDetected: enhanced.HallucinationDetected,
		Confidence:            enhanced.Confidence,
	}
	for _, s := range enhanced.Spans {
		res.UnsupportedSpans = append(res.UnsupportedSpans, s.Text)
	}
	return res, nil
}
