package classification

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// newTestEndpointDetector builds an endpoint detector pointed at the given base
// URL. The base URL is treated as the OpenAI-compatible "/v1" root; the detector
// appends "/chat/completions".
func newTestEndpointDetector(t *testing.T, endpoint string, includeExplanation bool) *EndpointHallucinationDetector {
	t.Helper()
	detector, err := NewEndpointHallucinationDetector(&config.HallucinationModelConfig{
		Backend:            config.HallucinationBackendEndpoint,
		Endpoint:           endpoint,
		ModelID:            "test-detector",
		IncludeExplanation: includeExplanation,
	})
	if err != nil {
		t.Fatalf("NewEndpointHallucinationDetector: %v", err)
	}
	return detector
}

// openAIResponse wraps content into an OpenAI-compatible chat completion body.
func openAIResponse(content string) string {
	body, _ := json.Marshal(map[string]interface{}{
		"choices": []map[string]interface{}{
			{"message": map[string]interface{}{"content": content}},
		},
	})
	return string(body)
}

func verifyRequestShape(t *testing.T, captured map[string]interface{}) {
	t.Helper()
	if captured["model"] != "test-detector" {
		t.Errorf("model = %v, want test-detector", captured["model"])
	}
	if captured["stream"] != false {
		t.Errorf("stream = %v, want false", captured["stream"])
	}
	rf, ok := captured["response_format"].(map[string]interface{})
	if !ok || rf["type"] != "json_schema" {
		t.Errorf("response_format = %v, want type json_schema", captured["response_format"])
	}
	messages, ok := captured["messages"].([]interface{})
	if !ok || len(messages) != 2 {
		t.Fatalf("messages = %v, want 2 entries", captured["messages"])
	}
	sys := messages[0].(map[string]interface{})
	if sys["role"] != "system" {
		t.Errorf("messages[0].role = %v, want system", sys["role"])
	}
	if !strings.Contains(sys["content"].(string), "explanation") {
		t.Errorf("include_explanation=true should request an explanation in the system prompt")
	}
	if messages[1].(map[string]interface{})["role"] != "user" {
		t.Errorf("messages[1].role = %v, want user", messages[1].(map[string]interface{})["role"])
	}
}

func TestEndpointDetector_RequestShape(t *testing.T) {
	var captured map[string]interface{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &captured); err != nil {
			t.Errorf("request body not JSON: %v", err)
		}
		_, _ = io.WriteString(w, openAIResponse(`{"hallucinated_spans": []}`))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, true)
	if _, err := detector.DetectWithNLI("the sky is blue", "what color is the sky?", "the sky is green"); err != nil {
		t.Fatalf("DetectWithNLI: %v", err)
	}

	verifyRequestShape(t, captured)
}

func TestEndpointDetector_CleanResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(`{"hallucinated_spans": []}`))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	result, err := detector.DetectWithNLI("ctx", "q", "answer text")
	if err != nil {
		t.Fatalf("DetectWithNLI: %v", err)
	}
	if result.HallucinationDetected {
		t.Errorf("HallucinationDetected = true, want false for empty span list")
	}
	if len(result.Spans) != 0 {
		t.Errorf("Spans = %d, want 0", len(result.Spans))
	}
}

func assertSpanShape(t *testing.T, first EnhancedHallucinationSpan, answer, expectedText, expectedExplanation string) {
	t.Helper()
	if first.Text != expectedText {
		t.Errorf("Span Text = %q, want %s", first.Text, expectedText)
	}
	wantStart := strings.Index(answer, expectedText)
	if first.Start != wantStart || first.End != wantStart+len(expectedText) {
		t.Errorf("Span offsets = [%d,%d), want [%d,%d)", first.Start, first.End, wantStart, wantStart+len(expectedText))
	}
	// NLI fields must be backend-neutral, not fabricated. The numeric label must
	// be the NLIUnknown sentinel (not 0, which is NLIEntailment) and stay
	// consistent with the string form.
	if first.NLILabelStr != "UNKNOWN" || first.NLIConfidence != 0 {
		t.Errorf("Span NLI = (%q, %v), want (UNKNOWN, 0)", first.NLILabelStr, first.NLIConfidence)
	}
	if first.NLILabel != NLIUnknown {
		t.Errorf("Span.NLILabel = %v, want NLIUnknown", first.NLILabel)
	}
	if first.NLILabel == NLIEntailment {
		t.Errorf("Span.NLILabel must not serialize as entailment (0)")
	}
	if first.Severity == 4 {
		t.Errorf("Span.Severity = 4 (critical); endpoint spans must not synthesize critical severity")
	}
	if first.Explanation != expectedExplanation {
		t.Errorf("Span.Explanation = %q, want model-supplied explanation", first.Explanation)
	}
}

func TestEndpointDetector_DetectedResponse_TaxonomyAndOffsets(t *testing.T) {
	answer := "The capital of France is Berlin and it was founded in 1850."
	content := `{"hallucinated_spans": [` +
		`{"text": "Berlin", "category": "contradiction", "subcategory": "entity", "explanation": "France's capital is Paris"},` +
		`{"text": "1850", "category": "unsupported_addition", "subcategory": "temporal"}` +
		`]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(content))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, true)
	result, err := detector.DetectWithNLI("France's capital is Paris.", "capital of France?", answer)
	if err != nil {
		t.Fatalf("DetectWithNLI: %v", err)
	}
	if !result.HallucinationDetected {
		t.Fatalf("HallucinationDetected = false, want true")
	}
	if len(result.Spans) != 2 {
		t.Fatalf("Spans = %d, want 2", len(result.Spans))
	}

	assertSpanShape(t, result.Spans[0], answer, "Berlin", "France's capital is Paris")
}

func TestEndpointDetector_SpanNotInAnswerIsDropped(t *testing.T) {
	// A single span that is not quoted from the answer is invalid, so the whole
	// response is all-invalid and must fail open via an error rather than a clean
	// verdict. A valid span alongside it is still returned.
	content := `{"hallucinated_spans": [` +
		`{"text": "not present anywhere", "category": "contradiction", "subcategory": "entity"},` +
		`{"text": "Berlin", "category": "contradiction", "subcategory": "entity"}` +
		`]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(content))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	result, err := detector.DetectWithNLI("ctx", "q", "Berlin is the answer")
	if err != nil {
		t.Fatalf("DetectWithNLI: %v", err)
	}
	if !result.HallucinationDetected || len(result.Spans) != 1 || result.Spans[0].Text != "Berlin" {
		t.Errorf("only the answer-quoted span should be kept, got detected=%v spans=%d", result.HallucinationDetected, len(result.Spans))
	}
}

func TestEndpointDetector_AllInvalidSpansReturnsError(t *testing.T) {
	// Every returned span is invalid (not quoted from the answer). This must be
	// treated as a malformed detector result, not hallucination_detected=false.
	content := `{"hallucinated_spans": [{"text": "not present anywhere", "category": "contradiction", "subcategory": "entity"}]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(content))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("ctx", "q", "a completely different answer"); err == nil {
		t.Fatalf("expected error when every returned span is invalid")
	}
}

func TestEndpointDetector_InvalidTaxonomyReturnsError(t *testing.T) {
	// A span whose taxonomy is outside the allowed enum is invalid; when it is the
	// only span, the response is all-invalid and must error rather than report a
	// clean verdict.
	answer := "value X is wrong"
	content := `{"hallucinated_spans": [{"text": "value X", "category": "bogus", "subcategory": "also_bogus"}]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(content))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("ctx", "q", answer); err == nil {
		t.Fatalf("expected error when the only span has invalid taxonomy")
	}
}

func TestEndpointDetector_MissingSpansArrayReturnsError(t *testing.T) {
	// A decodable body that omits hallucinated_spans is a schema violation, not a
	// clean verdict.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(`{}`))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("ctx", "q", "answer text"); err == nil {
		t.Fatalf("expected error when hallucinated_spans is missing")
	}
}

func TestEndpointDetector_NullSpansArrayReturnsError(t *testing.T) {
	// An explicit null is indistinguishable from missing at the value level and
	// must not be treated as an empty (clean) list.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(`{"hallucinated_spans": null}`))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("ctx", "q", "answer text"); err == nil {
		t.Fatalf("expected error when hallucinated_spans is null")
	}
}

func TestEndpointDetector_EmptyTextSpanReturnsError(t *testing.T) {
	// A span with empty text carries no verifiable evidence; as the only span it
	// makes the response all-invalid and must error.
	content := `{"hallucinated_spans": [{"text": "", "category": "contradiction", "subcategory": "entity"}]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(content))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("ctx", "q", "answer text"); err == nil {
		t.Fatalf("expected error when the only span has empty text")
	}
}

func TestEndpointDetector_NetworkErrorReturnsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {}))
	url := server.URL
	server.Close() // force connection refused

	detector := newTestEndpointDetector(t, url, false)
	result, err := detector.DetectWithNLI("ctx", "q", "answer")
	if err == nil {
		t.Fatalf("expected error on network failure, got result=%v", result)
	}
	if result != nil {
		t.Errorf("result must be nil on error (never a clean verdict), got %v", result)
	}
}

func TestEndpointDetector_Non200ReturnsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("ctx", "q", "answer"); err == nil {
		t.Fatalf("expected error on non-200 status")
	}
}

func TestEndpointDetector_MalformedResponseReturnsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{not valid json`)
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("ctx", "q", "answer"); err == nil {
		t.Fatalf("expected error on malformed outer response")
	}
}

func TestEndpointDetector_MalformedContentReturnsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse("this is not json schema output"))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("ctx", "q", "answer"); err == nil {
		t.Fatalf("expected error on malformed structured content")
	}
}

func TestEndpointDetector_OversizedResponseCappedAndErrors(t *testing.T) {
	// Emit more than the 10MB read cap of non-JSON bytes: the read is bounded and
	// the truncated body fails to parse, yielding an error rather than hanging or
	// a false clean verdict.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		junk := strings.Repeat("x", 11*1024*1024)
		_, _ = io.WriteString(w, junk)
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("ctx", "q", "answer"); err == nil {
		t.Fatalf("expected error on oversized/malformed response")
	}
}

func TestEndpointDetector_EmptyAnswerIsCleanNoCall(t *testing.T) {
	called := false
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		called = true
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	result, err := detector.DetectWithNLI("ctx", "q", "")
	if err != nil {
		t.Fatalf("DetectWithNLI empty answer: %v", err)
	}
	if result.HallucinationDetected {
		t.Errorf("empty answer must be clean")
	}
	if called {
		t.Errorf("empty answer must not call the endpoint")
	}
}

func TestEndpointDetector_EmptyContextReturnsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(`{"hallucinated_spans": []}`))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	if _, err := detector.DetectWithNLI("", "q", "answer"); err == nil {
		t.Fatalf("expected error when context is empty")
	}
}

func TestEndpointDetector_NLINotAdvertised(t *testing.T) {
	detector := newTestEndpointDetector(t, "http://127.0.0.1:65535/v1", true)
	if err := detector.Initialize(); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	if detector.IsNLIInitialized() {
		t.Errorf("endpoint backend must not advertise NLI readiness")
	}
}

func TestEndpointDetector_DetectMapsSpansToText(t *testing.T) {
	answer := "Berlin is the capital of France"
	content := `{"hallucinated_spans": [{"text": "Berlin", "category": "contradiction", "subcategory": "entity"}]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(content))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	result, err := detector.Detect("France's capital is Paris", "q", answer)
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}
	if !result.HallucinationDetected || len(result.UnsupportedSpans) != 1 || result.UnsupportedSpans[0] != "Berlin" {
		t.Errorf("Detect mapping = %+v, want single unsupported span 'Berlin'", result)
	}
}

func TestSystemPromptHashGuard(t *testing.T) {
	// The generative span detector was fine-tuned on a very specific prompt.
	// Any formatting drift degrades quality.
	baseHash := fmt.Sprintf("%x", sha256.Sum256([]byte(systemPromptBase)))
	explHash := fmt.Sprintf("%x", sha256.Sum256([]byte(systemPromptExpl)))

	// If these hashes change, you MUST verify the new prompts against the training data
	// (https://github.com/KRLabsOrg/LettuceDetect/blob/main/lettucedetect/prompts/generative.py)
	if baseHash != "05939b718dc3b737541666c3debf1ea426316bef7b410057cf1ee6c64a635e6f" {
		t.Errorf("systemPromptBase drift detected (hash: %s)", baseHash)
	}
	if explHash != "f6e3da772b0ed904a5785c422a81a0127f30f16285492ce74ef0c91fa1bda601" {
		t.Errorf("systemPromptExpl drift detected (hash: %s)", explHash)
	}
}
