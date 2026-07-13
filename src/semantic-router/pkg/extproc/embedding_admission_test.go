package extproc

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func localSelectionAdmissionRouter(admission *embedding.ProcessAdmission) *OpenAIRouter {
	return &OpenAIRouter{
		Config: &config.RouterConfig{
			IntelligentRouting: config.IntelligentRouting{
				Decisions: []config.Decision{{
					Name:      "local-selection",
					ModelRefs: []config.ModelRef{{Model: "small"}, {Model: "large"}},
					Algorithm: &config.AlgorithmConfig{Type: "router_dc"},
				}},
			},
		},
		embeddingAdmission: admission,
	}
}

func TestExtProcFallbackUsesSharedProcessAdmission(t *testing.T) {
	router := &OpenAIRouter{}
	if got := router.embeddingProcessAdmission(); got != embedding.DefaultProcessAdmission {
		t.Fatalf("ExtProc fallback admission = %p, want shared process admission %p", got, embedding.DefaultProcessAdmission)
	}
}

func TestDecisionAdmissionFailsFastForTextAndImageAndReleases(t *testing.T) {
	admission := embedding.NewProcessAdmission(1)
	heldRelease, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("occupy admission: %v", err)
	}

	router := localSelectionAdmissionRouter(admission)
	if _, imageErr := router.admitDecisionEvaluation(context.Background(), "MoM", "data:image/png;base64,AA=="); !errors.Is(imageErr, embedding.ErrOverloaded) {
		t.Fatalf("image decision admission error = %v, want overload", imageErr)
	}
	if _, textErr := router.admitDecisionEvaluation(context.Background(), "MoM", ""); !errors.Is(textErr, embedding.ErrOverloaded) {
		t.Fatalf("text decision admission error = %v, want overload", textErr)
	}

	heldRelease()
	imageRelease, err := router.admitDecisionEvaluation(context.Background(), "MoM", "data:image/png;base64,AA==")
	if err != nil {
		t.Fatalf("acquire released image slot: %v", err)
	}
	imageRelease()
}

func TestDecisionAdmissionSkipsPureStaticConfiguration(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	release, err := router.admitDecisionEvaluation(context.Background(), "explicit-model", "")
	if err != nil {
		t.Fatalf("pure static admission returned error: %v", err)
	}
	release()
}

func TestExtProcRemoteConfigWithAmbiguousLocalOverrideFailsBuild(t *testing.T) {
	t.Setenv("EMBEDDING_BACKEND_OVERRIDE", config.EmbeddingBackendOpenVINO)
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
			EmbeddingConfig: config.HNSWConfig{
				Backend:   config.EmbeddingBackendOpenAICompatible,
				ModelType: config.EmbeddingModelTypeRemote,
			},
			Endpoint: config.EmbeddingEndpointConfig{
				BaseURL: "https://embedding.invalid/v1",
				Model:   "remote-embedding",
			},
		}},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{EmbeddingRules: []config.EmbeddingRule{{
				Name:       "remote-config-local-override",
				Candidates: []string{"billing"},
			}}},
			Decisions: []config.Decision{{
				Name:      "billing-route",
				Rules:     config.RuleCombination{Type: config.SignalTypeEmbedding, Name: "remote-config-local-override"},
				ModelRefs: []config.ModelRef{{Model: "default-model"}},
			}},
		},
		BackendModels: config.BackendModels{DefaultModel: "default-model"},
	}
	if _, err := classification.BuildClassifier(cfg, nil, nil, nil); err == nil || !strings.Contains(err.Error(), "requires an explicit local model") {
		t.Fatalf("local override build error = %v", err)
	}
}

func TestClassificationOverloadResponseIsStableAndNonCacheable(t *testing.T) {
	router := &OpenAIRouter{}
	response := router.classificationEvaluationErrorResponse(embedding.ErrOverloaded)
	immediate := response.GetImmediateResponse()
	if immediate == nil {
		t.Fatal("expected immediate overload response")
	}
	if got := immediate.GetStatus().GetCode(); got != typev3.StatusCode_ServiceUnavailable {
		t.Fatalf("status = %v, want service unavailable", got)
	}
	headers := headerValuesByName(immediate.GetHeaders().GetSetHeaders())
	if got := headers["cache-control"]; got != "no-store" {
		t.Fatalf("cache-control = %q, want no-store", got)
	}
	if got := headers["retry-after"]; got != "1" {
		t.Fatalf("retry-after = %q, want 1", got)
	}
	if body := string(immediate.GetBody()); !containsAll(body, embeddingOverloadMessage, `"code":503`) {
		t.Fatalf("overload body = %q", body)
	}
}

func TestImageEvaluationErrorsAreContentSafeAndNonCacheable(t *testing.T) {
	router := &OpenAIRouter{}
	tests := []struct {
		name       string
		err        error
		wantStatus typev3.StatusCode
		wantBody   string
	}{
		{
			name:       "invalid input",
			err:        errors.Join(classification.ErrImageSignalEvaluation, classification.ErrInvalidImageSignalInput),
			wantStatus: typev3.StatusCode_BadRequest,
			wantBody:   "image input must contain a decodable JPEG or PNG image within the supported limits",
		},
		{
			name:       "internal failure",
			err:        fmt.Errorf("%w: private model path", classification.ErrImageSignalEvaluation),
			wantStatus: typev3.StatusCode_InternalServerError,
			wantBody:   "router image evaluation failed",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response := router.classificationEvaluationErrorResponse(tt.err)
			immediate := response.GetImmediateResponse()
			if got := immediate.GetStatus().GetCode(); got != tt.wantStatus {
				t.Fatalf("status = %v, want %v", got, tt.wantStatus)
			}
			if body := string(immediate.GetBody()); !strings.Contains(body, tt.wantBody) || strings.Contains(body, "private model path") {
				t.Fatalf("unsafe image evaluation body = %q", body)
			}
			if got := headerValuesByName(immediate.GetHeaders().GetSetHeaders())["cache-control"]; got != "no-store" {
				t.Fatalf("cache-control = %q, want no-store", got)
			}
		})
	}
}

func TestTextEvaluationErrorIsContentSafeAndNonCacheable(t *testing.T) {
	router := &OpenAIRouter{}
	response := router.classificationEvaluationErrorResponse(
		fmt.Errorf("%w: private provider detail", classification.ErrTextSignalEvaluation),
	)
	immediate := response.GetImmediateResponse()
	if got := immediate.GetStatus().GetCode(); got != typev3.StatusCode_ServiceUnavailable {
		t.Fatalf("status = %v, want service unavailable", got)
	}
	body := string(immediate.GetBody())
	if !strings.Contains(body, "router signal evaluation temporarily unavailable") || strings.Contains(body, "private provider detail") {
		t.Fatalf("unsafe text evaluation body = %q", body)
	}
	if got := headerValuesByName(immediate.GetHeaders().GetSetHeaders())["cache-control"]; got != "no-store" {
		t.Fatalf("cache-control = %q, want no-store", got)
	}
	if got := headerValuesByName(immediate.GetHeaders().GetSetHeaders())["retry-after"]; got != "1" {
		t.Fatalf("retry-after = %q, want 1", got)
	}
}

func TestHandleRequestBodyRemoteEmbeddingFailureCannotDefaultRoute(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "private upstream failure", http.StatusInternalServerError)
	}))
	defer server.Close()

	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{EmbeddingModels: config.EmbeddingModels{
			EmbeddingConfig: config.HNSWConfig{
				Backend:   config.EmbeddingBackendOpenAICompatible,
				ModelType: config.EmbeddingModelTypeRemote,
			},
			Endpoint: config.EmbeddingEndpointConfig{
				BaseURL: server.URL + "/v1",
				Model:   "remote-embedding",
			},
		}},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{EmbeddingRules: []config.EmbeddingRule{{
				Name:       "protected-route",
				Candidates: []string{"protected request"},
			}}},
			Decisions: []config.Decision{{
				Name:  "protected",
				Rules: config.RuleCombination{Type: config.SignalTypeEmbedding, Name: "protected-route"},
			}},
		},
		BackendModels: config.BackendModels{DefaultModel: "must-not-default"},
	}
	classifier, err := classification.BuildClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildClassifier: %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	response, err := router.handleRequestBody(
		&ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{Body: []byte(
			`{"model":"MoM","messages":[{"role":"user","content":"protected request"}]}`,
		)}},
		&RequestContext{Headers: map[string]string{":path": "/v1/chat/completions"}, TraceContext: context.Background()},
	)
	if err != nil {
		t.Fatalf("handleRequestBody: %v", err)
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil || immediate.GetStatus().GetCode() != typev3.StatusCode_ServiceUnavailable {
		t.Fatalf("remote failure response = %+v, want fail-closed 503", response)
	}
	body := string(immediate.GetBody())
	if strings.Contains(body, "private upstream failure") || strings.Contains(body, "must-not-default") {
		t.Fatalf("remote failure leaked detail or default-routed: %q", body)
	}
	headers := headerValuesByName(immediate.GetHeaders().GetSetHeaders())
	if headers["cache-control"] != "no-store" || headers["retry-after"] != "1" {
		t.Fatalf("remote failure headers = %v", headers)
	}
}

func TestUnknownEvaluationErrorDoesNotLeak(t *testing.T) {
	response := (&OpenAIRouter{}).classificationEvaluationErrorResponse(errors.New("private evaluator detail"))
	immediate := response.GetImmediateResponse()
	if immediate.GetStatus().GetCode() != typev3.StatusCode_InternalServerError || strings.Contains(string(immediate.GetBody()), "private evaluator detail") {
		t.Fatalf("unknown evaluation response = %+v", immediate)
	}
	if headerValuesByName(immediate.GetHeaders().GetSetHeaders())["cache-control"] != "no-store" {
		t.Fatal("unknown evaluation error was cacheable")
	}
}

func TestImageOnlyRequestIsEligibleForSignalEvaluation(t *testing.T) {
	ctx := &RequestContext{RequestImageURL: "data:image/png;base64,AA=="}
	if !hasDecisionSignalInput(signalConversationHistory{}, ctx.RequestImageURL) {
		t.Fatal("image-only request was classified as contentless")
	}
	if hasDecisionSignalInput(signalConversationHistory{}, "") {
		t.Fatal("contentless request was classified as signal-bearing")
	}
}

func TestImageOnlyDecisionEvaluationReachesSharedAdmission(t *testing.T) {
	admission := embedding.NewProcessAdmission(1)
	heldRelease, err := admission.TryAcquire(context.Background())
	if err != nil {
		t.Fatalf("occupy admission: %v", err)
	}
	defer heldRelease()

	router := localSelectionAdmissionRouter(admission)
	ctx := &RequestContext{
		RequestImageURL: "data:image/png;base64,AA==",
		TraceContext:    context.Background(),
	}

	_, _, _, _, err = router.performDecisionEvaluation(
		"MoM",
		signalConversationHistory{},
		ctx,
	)
	if !errors.Is(err, embedding.ErrOverloaded) {
		t.Fatalf("image-only evaluation error = %v, want shared admission overload", err)
	}
}

func TestHandleRequestBodyRejectsInvalidImageOnlySignalFailClosed(t *testing.T) {
	cfg := &config.RouterConfig{
		InlineModels: config.InlineModels{
			EmbeddingModels: config.EmbeddingModels{
				EmbeddingConfig: config.HNSWConfig{ModelType: config.EmbeddingModelTypeQwen3},
			},
		},
		IntelligentRouting: config.IntelligentRouting{
			Signals: config.Signals{
				EmbeddingRules: []config.EmbeddingRule{{
					Name:                "image-rule",
					Candidates:          []string{"safe image"},
					SimilarityThreshold: 0.5,
					QueryModality:       config.QueryModalityImage,
				}},
			},
			Decisions: []config.Decision{{
				Name:  "image-route",
				Rules: config.RuleCombination{Type: config.SignalTypeEmbedding, Name: "image-rule"},
			}},
		},
	}
	classifier, err := classification.BuildClassifier(cfg, nil, nil, nil)
	if err != nil {
		t.Fatalf("build classifier: %v", err)
	}
	router := &OpenAIRouter{Config: cfg, Classifier: classifier}
	body := []byte(`{"model":"MoM","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]}`)
	response, err := router.handleRequestBody(
		&ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{Body: body}},
		&RequestContext{
			Headers:      map[string]string{":path": "/v1/chat/completions"},
			TraceContext: context.Background(),
		},
	)
	if err != nil {
		t.Fatalf("handle request body: %v", err)
	}
	immediate := response.GetImmediateResponse()
	if immediate == nil || immediate.GetStatus().GetCode() != typev3.StatusCode_BadRequest {
		t.Fatalf("expected fail-closed 400, got %+v", response)
	}
	if body := string(immediate.GetBody()); !strings.Contains(body, "image input must contain") || strings.Contains(body, "AA==") {
		t.Fatalf("unexpected image failure body: %q", body)
	}
}

func TestHandleRequestBodyRejectsInvalidInlineImageBeforeDefaultRoute(t *testing.T) {
	tests := []struct {
		name     string
		protocol string
		path     string
		body     string
	}{
		{
			name: "openai empty base64",
			path: "/v1/chat/completions",
			body: `{"model":"MoM","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,"}}]}]}`,
		},
		{
			name: "openai invalid image bytes",
			path: "/v1/chat/completions",
			body: `{"model":"MoM","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,cHJpdmF0ZQ=="}}]}]}`,
		},
		{
			name: "openai unsupported inline format",
			path: "/v1/chat/completions",
			body: `{"model":"MoM","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="}}]}]}`,
		},
		{
			name:     "anthropic empty base64",
			protocol: config.ClientProtocolAnthropic,
			path:     "/v1/messages",
			body:     `{"model":"MoM","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":""}}]}]}`,
		},
		{
			name:     "anthropic invalid image bytes",
			protocol: config.ClientProtocolAnthropic,
			path:     "/v1/messages",
			body:     `{"model":"MoM","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"cHJpdmF0ZQ=="}}]}]}`,
		},
		{
			name:     "anthropic unsupported inline format",
			protocol: config.ClientProtocolAnthropic,
			path:     "/v1/messages",
			body:     `{"model":"MoM","messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/gif","data":"R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="}}]}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{
				BackendModels: config.BackendModels{DefaultModel: "must-not-be-selected"},
			}}
			response, err := router.handleRequestBody(
				&ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{Body: []byte(tt.body)}},
				&RequestContext{
					Headers:        map[string]string{":path": tt.path},
					TraceContext:   context.Background(),
					ClientProtocol: tt.protocol,
				},
			)
			if err != nil {
				t.Fatalf("handle request body: %v", err)
			}
			immediate := response.GetImmediateResponse()
			if immediate == nil || immediate.GetStatus().GetCode() != typev3.StatusCode_BadRequest {
				t.Fatalf("invalid inline image continued to default route: %+v", response)
			}
			headers := headerValuesByName(immediate.GetHeaders().GetSetHeaders())
			if headers["cache-control"] != "no-store" {
				t.Fatalf("cache-control = %q, want no-store", headers["cache-control"])
			}
			body := string(immediate.GetBody())
			if !strings.Contains(body, invalidRequestImageMessage) ||
				strings.Contains(body, "cHJpdmF0ZQ==") || strings.Contains(body, "must-not-be-selected") {
				t.Fatalf("unsafe invalid-image response body = %q", body)
			}
		})
	}
}

func TestHandleRequestBodyPreservesRemoteImageAndTextCompatibility(t *testing.T) {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "default-model",
			ModelConfig: map[string]config.ModelParams{
				"default-model": {PreferredEndpoints: []string{"default-endpoint"}},
			},
			VLLMEndpoints: []config.VLLMEndpoint{{
				Name: "default-endpoint", Address: "127.0.0.1", Port: 8000, Type: "vllm", Weight: 1,
			}},
		},
	}
	semanticCache, err := newTestSemanticCache(cfg)
	if err != nil {
		t.Fatalf("create disabled cache: %v", err)
	}
	router := &OpenAIRouter{
		Config:             cfg,
		Cache:              semanticCache,
		CredentialResolver: newTestCredentialResolver(cfg),
	}
	tests := []struct {
		name     string
		protocol string
		path     string
		body     string
	}{
		{
			name: "pure text",
			path: "/v1/chat/completions",
			body: `{"model":"MoM","messages":[{"role":"user","content":"hello"}]}`,
		},
		{
			name: "openai remote image",
			path: "/v1/chat/completions",
			body: `{"model":"MoM","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}]}]}`,
		},
		{
			name:     "anthropic remote image",
			protocol: config.ClientProtocolAnthropic,
			path:     "/v1/messages",
			body:     `{"model":"MoM","max_tokens":64,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"url","url":"https://example.com/image.png"}}]}]}`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			response, err := router.handleRequestBody(
				&ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{Body: []byte(tt.body)}},
				&RequestContext{
					Headers:        map[string]string{":path": tt.path},
					TraceContext:   context.Background(),
					ClientProtocol: tt.protocol,
				},
			)
			if err != nil {
				t.Fatalf("handle request body: %v", err)
			}
			if response.GetImmediateResponse() != nil || response.GetRequestBody() == nil {
				t.Fatalf("compatible request was rejected: %+v", response)
			}
		})
	}
}

func containsAll(value string, fragments ...string) bool {
	for _, fragment := range fragments {
		if !strings.Contains(value, fragment) {
			return false
		}
	}
	return true
}
