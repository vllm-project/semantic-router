package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	systemPromptBase = `You are an expert annotator who identifies hallucinated spans in a generated answer with respect to a given context (the only trusted evidence). A hallucinated span is a substring of the answer that is not supported by the context. Spans consistent with the context are not hallucinations.

Quote each hallucinated span verbatim from the answer and classify it into exactly one category and one subcategory.

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

Reply with ONLY a JSON object (no markdown, no code fences): {"hallucinated_spans": [{"text": "...", "category": "...", "subcategory": "..."}]}. If nothing is unsupported, reply {"hallucinated_spans": []}.`

	systemPromptExpl = `You are an expert annotator who identifies hallucinated spans in a generated answer with respect to a given context (the only trusted evidence). A hallucinated span is a substring of the answer that is not supported by the context. Spans consistent with the context are not hallucinations.

Quote each hallucinated span verbatim from the answer and classify it into exactly one category and one subcategory, and give a short explanation of why it is unsupported.

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

Reply with ONLY a JSON object (no markdown, no code fences): {"hallucinated_spans": [{"text": "...", "category": "...", "subcategory": "...", "explanation": "..."}]}. If nothing is unsupported, reply {"hallucinated_spans": []}.`
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
	client      *connector.Client
	handle      *binding.Resolved[tasks.GroundedTextRequest, tasks.TokenClassificationResult]
	spec        config.ResolvedModelBinding
	endpoint    string
}

func NewEndpointHallucinationDetector(cfg *config.HallucinationModelConfig, models ...*classifierModelRuntime) (*EndpointHallucinationDetector, error) {
	if cfg == nil {
		return nil, fmt.Errorf("hallucination model config is required")
	}
	runtime := consumerModelRuntime(models)
	endpoint := strings.TrimRight(strings.TrimSpace(cfg.Endpoint), "/")
	external := &config.ExternalModelConfig{Name: "hallucination_endpoint", ModelName: cfg.ModelID, ModelEndpoint: config.ClassifierVLLMEndpoint{Address: endpoint}, TimeoutSeconds: 10}
	spec := config.ResolvedModelBinding{Recipe: runtime.recipe, Name: "hallucination_detector", Binding: config.ModelBinding{Deployment: "hallucination_detector", Contract: config.RemoteClassifierContractTokenSpans, Adapter: config.RemoteClassifierProtocolHTTPChat}, Deployment: config.ModelDeployment{Provider: "http", ExternalModel: external.Name}, Admission: runtime.cfg.ModelAdmission["hallucination_detector"]}
	if declared, ok := runtime.plan.Lookup(runtime.recipe, "hallucination_detector"); ok {
		spec = declared
		if declared.Deployment.Provider != "http" {
			return nil, fmt.Errorf("endpoint hallucination detector requires HTTP deployment")
		}
		backend := &config.RemoteClassifierBackend{Model: declared.Deployment.ExternalModel, Protocol: declared.Binding.Adapter, Contract: declared.Binding.Contract}
		var err error
		external, err = config.ResolveRemoteClassifierBackend(runtime.cfg, backend, config.ModelRoleClassification, config.RemoteClassifierContractTokenSpans)
		if err != nil {
			return nil, err
		}
		scheme := external.ModelEndpoint.Protocol
		if scheme == "" {
			scheme = "http"
		}
		endpoint = fmt.Sprintf("%s://%s:%d/v1", scheme, external.ModelEndpoint.Address, external.ModelEndpoint.Port)
		copied := *cfg
		copied.ModelID = external.ModelName
		copied.Endpoint = endpoint
		cfg = &copied
	}
	if endpoint == "" {
		return nil, fmt.Errorf("hallucination endpoint is required when backend is endpoint")
	}
	if cfg.ModelID == "" {
		return nil, fmt.Errorf("hallucination model_id is required")
	}
	client, err := connector.New(endpoint, bearerAuthorizer(external.AccessKey), connector.Options{AttemptTimeout: external.GetTimeout(), MaxRequestBytes: external.GetMaxRequestBytes(), MaxResponseBytes: external.GetMaxResponseBytes(), MaxErrorBytes: 4096})
	if err != nil {
		return nil, err
	}
	detector := &EndpointHallucinationDetector{config: cfg, client: client, endpoint: endpoint, spec: spec}
	handle, err := remoteTaskBinding(context.Background(), runtime, spec, external, client, detector.classifyGrounded, func(input tasks.GroundedTextRequest, out tasks.TokenClassificationResult) error {
		for _, span := range out.Entities {
			if span.Start < 0 || span.End > len(input.Answer) || span.Start >= span.End || input.Answer[span.Start:span.End] != span.Text {
				return fmt.Errorf("invalid grounding span")
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	detector.handle = handle
	return detector, nil
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
	prompt := fmt.Sprintf("User request: %s\n\nExcerpt 1:\n%s\n\nAnswer to verify:\n%s", question, reqContext, answer)

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
// Start/End offsets are populated. The NLI label is set to the NLIUnknown sentinel
// (not 0, which is NLIEntailment) because the endpoint backend does not produce NLI
// labels, keeping the numeric and string forms consistent.
func (d *EndpointHallucinationDetector) parseOpenAIResponse(respBytes []byte, answer string) ([]tasks.TokenEntity, error) {
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

	// HallucinatedSpans is a pointer so we can distinguish an explicitly present
	// array (including []) from a missing field or JSON null. A decodable body
	// that omits the array is a schema violation, not a clean verdict.
	var parsed struct {
		HallucinatedSpans *[]struct {
			Text        string `json:"text"`
			Category    string `json:"category"`
			Subcategory string `json:"subcategory"`
			Explanation string `json:"explanation"`
		} `json:"hallucinated_spans"`
	}

	if err := json.Unmarshal([]byte(openaiResp.Choices[0].Message.Content), &parsed); err != nil {
		return nil, fmt.Errorf("JSON schema parsing failed: %w", err)
	}
	if parsed.HallucinatedSpans == nil {
		return nil, fmt.Errorf("endpoint response missing required hallucinated_spans array")
	}

	rawSpans := *parsed.HallucinatedSpans
	spans := make([]tasks.TokenEntity, 0, len(rawSpans))
	invalidCount := 0
	for _, s := range rawSpans {
		if s.Text == "" {
			invalidCount++
			continue
		}
		// The span must be quoted verbatim from the answer; anything not actually
		// present is invalid so we never emit fabricated offsets.
		start := strings.Index(answer, s.Text)
		if start < 0 {
			logging.Debugf("Endpoint hallucination span not found in answer, skipping: %q", s.Text)
			invalidCount++
			continue
		}

		category := normalizeTaxonomyValue(s.Category, endpointCategories)
		subcategory := normalizeTaxonomyValue(s.Subcategory, endpointSubcategories)
		if category == "" || subcategory == "" {
			logging.Debugf("Endpoint hallucination span has invalid taxonomy, skipping: category=%q subcategory=%q", s.Category, s.Subcategory)
			invalidCount++
			continue
		}

		spans = append(spans, tasks.TokenEntity{Text: s.Text, Start: start, End: start + len(s.Text), EntityType: category, Subtype: subcategory, Explanation: endpointSpanExplanation(s.Explanation, category, subcategory)})
	}

	// A response that returned spans but where every one failed validation is a
	// malformed detector result, not a clean verdict. Fail open via an error
	// rather than silently reporting hallucination_detected=false.
	if len(spans) == 0 && invalidCount > 0 {
		return nil, fmt.Errorf("endpoint returned %d span(s) but none were valid", invalidCount)
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
func (d *EndpointHallucinationDetector) classifyGrounded(ctx context.Context, input tasks.GroundedTextRequest) (tasks.TokenClassificationResult, error) {
	if input.Context == "" {
		return tasks.TokenClassificationResult{}, fmt.Errorf("context is required for hallucination detection")
	}
	body, err := d.buildRequestPayload(input.Context, input.Question, input.Answer)
	if err != nil {
		return tasks.TokenClassificationResult{}, err
	}
	response, err := d.client.Do(ctx, connector.Operation{Name: "hallucination", Method: http.MethodPost, Path: "/chat/completions", SuccessStatusCode: http.StatusOK}, body)
	if err != nil {
		return tasks.TokenClassificationResult{}, err
	}
	spans, err := d.parseOpenAIResponse(response, input.Answer)
	available := false
	return tasks.TokenClassificationResult{Entities: spans, ScoresAvailable: &available}, err
}

func (d *EndpointHallucinationDetector) ClassifyGrounded(ctx context.Context, input tasks.GroundedTextRequest) (tasks.TokenClassificationResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()
	if !d.initialized {
		return tasks.TokenClassificationResult{}, binding.ErrClosed
	}
	return d.handle.Call(ctx, string(d.spec.Recipe), input)
}

func (d *EndpointHallucinationDetector) DetectWithNLI(ctx context.Context, reqContext, question, answer string) (*EnhancedHallucinationResult, error) {
	if answer == "" {
		return d.cleanResult(), nil
	}
	result, err := d.ClassifyGrounded(ctx, tasks.GroundedTextRequest{Context: reqContext, Question: question, Answer: answer})
	if err != nil {
		return nil, err
	}
	enhanced := &EnhancedHallucinationResult{HallucinationDetected: len(result.Entities) > 0, Spans: make([]EnhancedHallucinationSpan, 0, len(result.Entities))}
	for _, span := range result.Entities {
		enhanced.Spans = append(enhanced.Spans, EnhancedHallucinationSpan{Text: span.Text, Start: span.Start, End: span.End, NLILabel: NLIUnknown, NLILabelStr: NLIUnknown.String(), Severity: 2, Explanation: span.Explanation})
	}
	return enhanced, nil
}

// cleanResult is the "nothing to verify" verdict, used only when the answer is
// empty. Endpoint failures return an error instead so they are never recorded as
// a clean (not_detected) verdict.
func (d *EndpointHallucinationDetector) cleanResult() *EnhancedHallucinationResult {
	return &EnhancedHallucinationResult{
		HallucinationDetected: false,
		Spans:                 []EnhancedHallucinationSpan{},
	}
}

func (d *EndpointHallucinationDetector) Detect(ctx context.Context, reqContext, question, answer string) (*HallucinationResult, error) {
	enhanced, err := d.DetectWithNLI(ctx, reqContext, question, answer)
	if err != nil {
		return nil, err
	}

	res := &HallucinationResult{
		HallucinationDetected: enhanced.HallucinationDetected,
		Confidence:            enhanced.Confidence,
		ScoreAvailable:        enhanced.ScoreAvailable,
		ScoreKind:             enhanced.ScoreKind,
	}
	for _, s := range enhanced.Spans {
		res.UnsupportedSpans = append(res.UnsupportedSpans, s.Text)
	}
	return res, nil
}

func (d *EndpointHallucinationDetector) Close() error {
	if d == nil {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.initialized = false
	if d.handle != nil {
		return d.handle.Close()
	}
	return nil
}
