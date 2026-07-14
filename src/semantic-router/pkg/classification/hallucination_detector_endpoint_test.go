package classification

import (
	"encoding/json"
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

	first := result.Spans[0]
	if first.Text != "Berlin" {
		t.Errorf("Spans[0].Text = %q, want Berlin", first.Text)
	}
	wantStart := strings.Index(answer, "Berlin")
	if first.Start != wantStart || first.End != wantStart+len("Berlin") {
		t.Errorf("Spans[0] offsets = [%d,%d), want [%d,%d)", first.Start, first.End, wantStart, wantStart+len("Berlin"))
	}
	// NLI fields must be backend-neutral, not fabricated.
	if first.NLILabelStr != "UNKNOWN" || first.NLIConfidence != 0 {
		t.Errorf("Spans[0] NLI = (%q, %v), want (UNKNOWN, 0)", first.NLILabelStr, first.NLIConfidence)
	}
	if first.NLILabel != 0 {
		t.Errorf("Spans[0].NLILabel = %v, want 0", first.NLILabel)
	}
	if first.Severity == 4 {
		t.Errorf("Spans[0].Severity = 4 (critical); endpoint spans must not synthesize critical severity")
	}
	if first.Explanation != "France's capital is Paris" {
		t.Errorf("Spans[0].Explanation = %q, want model-supplied explanation", first.Explanation)
	}
}

func TestEndpointDetector_SpanNotInAnswerIsSkipped(t *testing.T) {
	content := `{"hallucinated_spans": [{"text": "not present anywhere", "category": "contradiction", "subcategory": "entity"}]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(content))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	result, err := detector.DetectWithNLI("ctx", "q", "a completely different answer")
	if err != nil {
		t.Fatalf("DetectWithNLI: %v", err)
	}
	if result.HallucinationDetected || len(result.Spans) != 0 {
		t.Errorf("span not quoted from answer must be skipped, got detected=%v spans=%d", result.HallucinationDetected, len(result.Spans))
	}
}

func TestEndpointDetector_InvalidTaxonomyIsSanitized(t *testing.T) {
	answer := "value X is wrong"
	content := `{"hallucinated_spans": [{"text": "value X", "category": "bogus", "subcategory": "also_bogus"}]}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, openAIResponse(content))
	}))
	defer server.Close()

	detector := newTestEndpointDetector(t, server.URL, false)
	result, err := detector.DetectWithNLI("ctx", "q", answer)
	if err != nil {
		t.Fatalf("DetectWithNLI: %v", err)
	}
	if len(result.Spans) != 1 {
		t.Fatalf("Spans = %d, want 1", len(result.Spans))
	}
	// Invalid taxonomy values must not leak into the explanation verbatim.
	if strings.Contains(result.Spans[0].Explanation, "bogus") {
		t.Errorf("invalid taxonomy leaked into explanation: %q", result.Spans[0].Explanation)
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
