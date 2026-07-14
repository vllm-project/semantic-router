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

// IsNLIInitialized returns true because the endpoint natively provides explanation when requested
func (d *EndpointHallucinationDetector) IsNLIInitialized() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.initialized
}

func (d *EndpointHallucinationDetector) buildRequestPayload(reqContext, question, answer string) ([]byte, error) {
	prompt := fmt.Sprintf("User request: %s\n\nContext (trusted evidence):\n%s\n\nAnswer to verify:\n%s", question, reqContext, answer)

	systemPrompt := systemPromptBase
	if d.config.IncludeExplanation {
		systemPrompt = systemPromptExpl
	}

	spanProps := map[string]interface{}{
		"text":        map[string]interface{}{"type": "string"},
		"category":    map[string]interface{}{"type": "string", "enum": []string{"contradiction", "fabricated_reference", "unsupported_addition"}},
		"subcategory": map[string]interface{}{"type": "string", "enum": []string{"entity", "temporal", "numerical", "value", "relational", "identifier", "section", "attribute", "claim", "behavior", "elaboration", "subjective", "unspecified"}},
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

func (d *EndpointHallucinationDetector) parseOpenAIResponse(respBytes []byte) ([]EnhancedHallucinationSpan, error) {
	var openaiResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}

	if err := json.Unmarshal(respBytes, &openaiResp); err != nil || len(openaiResp.Choices) == 0 {
		return nil, fmt.Errorf("response parsing failed")
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

	var spans []EnhancedHallucinationSpan
	for _, s := range parsed.HallucinatedSpans {
		explanation := s.Explanation
		if explanation == "" {
			explanation = fmt.Sprintf("Unsupported %s (%s) detected", s.Subcategory, s.Category)
		}

		spans = append(spans, EnhancedHallucinationSpan{
			Text:                    s.Text,
			HallucinationConfidence: 1.0,
			NLILabel:                NLIContradiction,
			NLILabelStr:             s.Category,
			NLIConfidence:           1.0,
			Severity:                4,
			Explanation:             explanation,
		})
	}
	return spans, nil
}

func (d *EndpointHallucinationDetector) DetectWithNLI(reqContext, question, answer string) (*EnhancedHallucinationResult, error) {
	if answer == "" {
		return d.fallbackResult(), nil
	}
	if reqContext == "" {
		logging.Warnf("Endpoint hallucination detection requested with empty context")
		return d.fallbackResult(), nil
	}

	bodyBytes, err := d.buildRequestPayload(reqContext, question, answer)
	if err != nil {
		return d.fallbackResult(), nil
	}
	req, err := http.NewRequestWithContext(context.Background(), "POST", d.endpoint+"/chat/completions", bytes.NewReader(bodyBytes))
	if err != nil {
		return d.fallbackResult(), nil
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := d.client.Do(req)
	if err != nil {
		logging.Warnf("Endpoint hallucination detection failed (fallback to safe): %v", err)
		return d.fallbackResult(), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		logging.Warnf("Endpoint hallucination detection non-200 status: %d", resp.StatusCode)
		return d.fallbackResult(), nil
	}

	// Limit response size to 10MB to prevent OOM
	respBytes, err := io.ReadAll(io.LimitReader(resp.Body, 10*1024*1024))
	if err != nil {
		logging.Warnf("Endpoint hallucination detection response read failed: %v", err)
		return d.fallbackResult(), nil
	}

	spans, err := d.parseOpenAIResponse(respBytes)
	if err != nil {
		logging.Warnf("Endpoint hallucination detection %v", err)
		return d.fallbackResult(), nil
	}

	return &EnhancedHallucinationResult{
		HallucinationDetected: len(spans) > 0,
		Confidence:            1.0,
		Spans:                 spans,
	}, nil
}

func (d *EndpointHallucinationDetector) fallbackResult() *EnhancedHallucinationResult {
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
