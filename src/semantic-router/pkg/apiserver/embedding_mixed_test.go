//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

type mixedEncoderObservation struct {
	legacyTextCalls int
	textCalls       int
	imageCalls      int
	textDimension   int
	imageDimension  int
	encodedImage    string
}

func installSuccessfulMixedEmbeddingEncoders(t *testing.T) *mixedEncoderObservation {
	t.Helper()
	originalLegacyText := embeddingOutputForRequest
	originalMultimodalText := multiModalEncodeText
	originalMultimodalImage := multiModalEncodeImage
	t.Cleanup(func() {
		embeddingOutputForRequest = originalLegacyText
		multiModalEncodeText = originalMultimodalText
		multiModalEncodeImage = originalMultimodalImage
	})

	observation := &mixedEncoderObservation{}
	embeddingOutputForRequest = func(EmbeddingRequest, string) (*candle_binding.EmbeddingOutput, error) {
		observation.legacyTextCalls++
		return nil, errors.New("legacy text encoder must not serve a mixed request")
	}
	multiModalEncodeText = func(_ string, dimension int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		observation.textCalls++
		observation.textDimension = dimension
		return &candle_binding.MultiModalEmbeddingOutput{
			Embedding:        make([]float32, dimension),
			Modality:         "text",
			ProcessingTimeMs: 2,
		}, nil
	}
	multiModalEncodeImage = func(image string, dimension int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		observation.imageCalls++
		observation.imageDimension = dimension
		observation.encodedImage = image
		return &candle_binding.MultiModalEmbeddingOutput{
			Embedding:        make([]float32, dimension),
			Modality:         "image",
			ProcessingTimeMs: 4,
		}, nil
	}
	return observation
}

func TestEmbeddingHandlerRoutesMixedInputsThroughSharedMultimodalSpace(t *testing.T) {
	observation := installSuccessfulMixedEmbeddingEncoders(t)

	imageURI := mustEmbeddingImageDataURI(t, "image/png")
	body, err := json.Marshal(EmbeddingRequest{Texts: []string{"hello"}, Images: []string{imageURI}})
	if err != nil {
		t.Fatalf("marshal mixed request: %v", err)
	}
	server := &ClassificationAPIServer{embeddingAdmission: newEmbeddingProcessAdmission(1)}
	request := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings", strings.NewReader(string(body)))
	request.Header.Set("Content-Type", "application/json")
	responseRecorder := httptest.NewRecorder()

	server.handleEmbeddings(responseRecorder, request)

	if responseRecorder.Code != http.StatusOK {
		t.Fatalf("mixed embedding status = %d: %s", responseRecorder.Code, responseRecorder.Body.String())
	}
	var response EmbeddingResponse
	if err := json.Unmarshal(responseRecorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode mixed response: %v", err)
	}
	assertMixedEncoderObservation(t, observation)
	assertMixedEmbeddingResponse(t, response)
}

func assertMixedEncoderObservation(t *testing.T, observation *mixedEncoderObservation) {
	t.Helper()
	if observation.legacyTextCalls != 0 || observation.textCalls != 1 || observation.imageCalls != 1 {
		t.Fatalf(
			"encoder calls = legacy:%d text:%d image:%d, want 0/1/1",
			observation.legacyTextCalls,
			observation.textCalls,
			observation.imageCalls,
		)
	}
	if observation.textDimension != defaultMixedEmbeddingDimension ||
		observation.imageDimension != defaultMixedEmbeddingDimension {
		t.Fatalf(
			"mixed dimensions = text:%d image:%d, want %d",
			observation.textDimension,
			observation.imageDimension,
			defaultMixedEmbeddingDimension,
		)
	}
	if !strings.HasPrefix(observation.encodedImage, "data:image/png;base64,") {
		t.Fatalf("image input was not canonicalized: %q", observation.encodedImage)
	}
}

func assertMixedEmbeddingResponse(t *testing.T, response EmbeddingResponse) {
	t.Helper()
	if len(response.Embeddings) != 2 || response.TotalCount != 2 {
		t.Fatalf("mixed response counts = len:%d total:%d", len(response.Embeddings), response.TotalCount)
	}
	assertMixedEmbeddingResult(t, response.Embeddings[0], "hello", "text")
	assertMixedEmbeddingResult(t, response.Embeddings[1], "", "image")
	if response.TotalProcessingTimeMs != 6 || response.AvgProcessingTimeMs != 3 {
		t.Fatalf("mixed processing metadata = total:%d avg:%v", response.TotalProcessingTimeMs, response.AvgProcessingTimeMs)
	}
}

func assertMixedEmbeddingResult(t *testing.T, result EmbeddingResult, text, modality string) {
	t.Helper()
	if result.Text != text {
		t.Fatalf("%s result text = %q, want %q", modality, result.Text, text)
	}
	if result.Modality != modality {
		t.Fatalf("result modality = %q, want %q", result.Modality, modality)
	}
	if result.ModelUsed != multimodalEmbeddingModel {
		t.Fatalf("%s result model = %q", modality, result.ModelUsed)
	}
	if result.Dimension != defaultMixedEmbeddingDimension {
		t.Fatalf("%s result dimension = %d", modality, result.Dimension)
	}
}

func TestApplyEmbeddingDefaultsLeavesMixedTextControlsUnset(t *testing.T) {
	req := EmbeddingRequest{Texts: []string{"hello"}, Images: []string{"placeholder"}}

	applyEmbeddingDefaults(&req)

	if req.Model != "auto" || req.Dimension != defaultMixedEmbeddingDimension {
		t.Fatalf("mixed defaults = model:%q dimension:%d", req.Model, req.Dimension)
	}
	if req.TargetLayer != 0 || req.QualityPriority != 0 || req.LatencyPriority != 0 {
		t.Fatalf("mixed defaults injected unsupported text controls: %#v", req)
	}
}

func TestValidateMixedEmbeddingRequestRejectsTextOnlyControls(t *testing.T) {
	base := EmbeddingRequest{
		Texts:     []string{"hello"},
		Images:    []string{"placeholder"},
		Model:     "auto",
		Dimension: defaultMixedEmbeddingDimension,
	}
	if code, message, ok := validateEmbeddingRequestControls(base, nil); !ok {
		t.Fatalf("valid mixed controls rejected: code=%q message=%q", code, message)
	}

	tests := []struct {
		name     string
		mutate   func(*EmbeddingRequest)
		wantCode string
	}{
		{name: "explicit model", mutate: func(req *EmbeddingRequest) { req.Model = "mmbert" }, wantCode: "INVALID_MODEL"},
		{name: "target layer", mutate: func(req *EmbeddingRequest) { req.TargetLayer = 6 }, wantCode: "INVALID_PARAMETER"},
		{name: "quality priority", mutate: func(req *EmbeddingRequest) { req.QualityPriority = 0.5 }, wantCode: "INVALID_PARAMETER"},
		{name: "latency priority", mutate: func(req *EmbeddingRequest) { req.LatencyPriority = 0.5 }, wantCode: "INVALID_PARAMETER"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req := base
			test.mutate(&req)
			code, message, ok := validateEmbeddingRequestControls(req, nil)
			if ok || code != test.wantCode || !strings.Contains(message, "mixed") {
				t.Fatalf("mixed controls = ok:%v code:%q message:%q, want %q", ok, code, message, test.wantCode)
			}
		})
	}
}

func TestValidateMixedEmbeddingCapabilitiesFailsClosed(t *testing.T) {
	baseCapabilities := embeddingBackendCapabilities{
		name:                    "test",
		supportsMultimodalText:  true,
		supportsMultimodalImage: true,
		multimodalDimensions:    []int{defaultMixedEmbeddingDimension},
	}
	baseRequest := EmbeddingRequest{
		Texts:     []string{"hello"},
		Images:    []string{"placeholder"},
		Model:     "auto",
		Dimension: defaultMixedEmbeddingDimension,
	}
	tests := []struct {
		name       string
		capability func(*embeddingBackendCapabilities)
		request    func(*EmbeddingRequest)
		wantCode   string
		wantOK     bool
	}{
		{name: "supported", wantOK: true},
		{
			name: "missing multimodal text",
			capability: func(capabilities *embeddingBackendCapabilities) {
				capabilities.supportsMultimodalText = false
			},
			wantCode: unsupportedEmbeddingCapabilityCode,
		},
		{
			name: "missing multimodal image",
			capability: func(capabilities *embeddingBackendCapabilities) {
				capabilities.supportsMultimodalImage = false
			},
			wantCode: unsupportedEmbeddingCapabilityCode,
		},
		{
			name: "unsupported shared dimension",
			request: func(req *EmbeddingRequest) {
				req.Dimension = defaultImageEmbeddingDimension
			},
			wantCode: "INVALID_DIMENSION",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			capabilities := baseCapabilities
			req := baseRequest
			if test.capability != nil {
				test.capability(&capabilities)
			}
			if test.request != nil {
				test.request(&req)
			}
			code, message, ok := validateMixedEmbeddingCapabilities(capabilities, req)
			if ok != test.wantOK || code != test.wantCode {
				t.Fatalf("mixed capability = ok:%v code:%q message:%q, want ok:%v code:%q", ok, code, message, test.wantOK, test.wantCode)
			}
		})
	}
}

func TestBuildEmbeddingResultsRejectsMixedTextContractMismatch(t *testing.T) {
	originalLegacyText := embeddingOutputForRequest
	originalMultimodalText := multiModalEncodeText
	t.Cleanup(func() {
		embeddingOutputForRequest = originalLegacyText
		multiModalEncodeText = originalMultimodalText
	})
	embeddingOutputForRequest = func(EmbeddingRequest, string) (*candle_binding.EmbeddingOutput, error) {
		return nil, errors.New("legacy text encoder must not serve a mixed request")
	}

	tests := []struct {
		name   string
		output *candle_binding.MultiModalEmbeddingOutput
	}{
		{name: "nil output"},
		{name: "wrong modality", output: &candle_binding.MultiModalEmbeddingOutput{
			Embedding: make([]float32, defaultMixedEmbeddingDimension), Modality: "image",
		}},
		{name: "wrong dimension", output: &candle_binding.MultiModalEmbeddingOutput{
			Embedding: make([]float32, defaultMixedEmbeddingDimension-1), Modality: "text",
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			multiModalEncodeText = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
				return test.output, nil
			}
			_, _, err := buildEmbeddingResults(EmbeddingRequest{
				Texts:     []string{"hello"},
				Images:    []string{"placeholder"},
				Model:     "auto",
				Dimension: defaultMixedEmbeddingDimension,
			})
			var contractErr *embeddingOutputContractError
			if !errors.As(err, &contractErr) {
				t.Fatalf("mixed text contract error = %T %v, want embeddingOutputContractError", err, err)
			}
			if contractErr.input != "texts[0]" {
				t.Fatalf("mixed text contract input = %q", contractErr.input)
			}
			status, code, message := classifyEmbeddingError(err)
			if status != http.StatusInternalServerError || code != "EMBEDDING_GENERATION_FAILED" ||
				message != "failed to generate embedding" {
				t.Fatalf("mixed contract classification = %d %q %q", status, code, message)
			}
		})
	}
}

func TestMixedEmbeddingHandlerMapsNativeTextLimitToStable413(t *testing.T) {
	originalLegacyText := embeddingOutputForRequest
	originalMultimodalText := multiModalEncodeText
	originalMultimodalImage := multiModalEncodeImage
	t.Cleanup(func() {
		embeddingOutputForRequest = originalLegacyText
		multiModalEncodeText = originalMultimodalText
		multiModalEncodeImage = originalMultimodalImage
	})
	embeddingOutputForRequest = func(EmbeddingRequest, string) (*candle_binding.EmbeddingOutput, error) {
		return nil, errors.New("legacy text encoder must not serve a mixed request")
	}
	multiModalEncodeText = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		return nil, fmt.Errorf("%w: private mixed token counts", candle_binding.ErrEmbeddingInputTooLong)
	}
	imageCalls := 0
	multiModalEncodeImage = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		imageCalls++
		return nil, errors.New("image encoder must not run after a text failure")
	}

	imageURI := mustEmbeddingImageDataURI(t, "image/png")
	body, err := json.Marshal(EmbeddingRequest{Texts: []string{"hello"}, Images: []string{imageURI}})
	if err != nil {
		t.Fatalf("marshal mixed request: %v", err)
	}
	server := &ClassificationAPIServer{embeddingAdmission: newEmbeddingProcessAdmission(1)}
	request := httptest.NewRequest(http.MethodPost, "/api/v1/embeddings", strings.NewReader(string(body)))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()

	server.handleEmbeddings(response, request)

	if response.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("mixed text limit status = %d: %s", response.Code, response.Body.String())
	}
	assertJSONErrorCode(t, response.Body.Bytes(), embeddingInputTooLargeCode)
	assertJSONErrorMessage(t, response.Body.Bytes(), "embedding input exceeds the selected model context")
	if imageCalls != 0 || strings.Contains(response.Body.String(), "private") {
		t.Fatalf("mixed text limit leaked detail or continued inference: calls=%d body=%s", imageCalls, response.Body.String())
	}
}

func TestBuildEmbeddingResultsPreservesSingleModalityContracts(t *testing.T) {
	originalLegacyText := embeddingOutputForRequest
	originalMultimodalText := multiModalEncodeText
	originalMultimodalImage := multiModalEncodeImage
	t.Cleanup(func() {
		embeddingOutputForRequest = originalLegacyText
		multiModalEncodeText = originalMultimodalText
		multiModalEncodeImage = originalMultimodalImage
	})

	legacyTextCalls := 0
	multimodalTextCalls := 0
	imageCalls := 0
	embeddingOutputForRequest = func(req EmbeddingRequest, _ string) (*candle_binding.EmbeddingOutput, error) {
		legacyTextCalls++
		return &candle_binding.EmbeddingOutput{
			Embedding: make([]float32, req.Dimension),
			ModelType: "mmbert",
		}, nil
	}
	multiModalEncodeText = func(string, int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		multimodalTextCalls++
		return nil, errors.New("multimodal text encoder must not serve text-only requests")
	}
	multiModalEncodeImage = func(_ string, dimension int) (*candle_binding.MultiModalEmbeddingOutput, error) {
		imageCalls++
		return &candle_binding.MultiModalEmbeddingOutput{
			Embedding: make([]float32, dimension),
			Modality:  "image",
		}, nil
	}

	textResults, _, err := buildEmbeddingResults(EmbeddingRequest{
		Texts: []string{"hello"}, Model: "mmbert", Dimension: defaultEmbeddingDimension,
	})
	if err != nil {
		t.Fatalf("text-only result: %v", err)
	}
	if len(textResults) != 1 || textResults[0].ModelUsed != "mmbert" || textResults[0].Modality != "" {
		t.Fatalf("text-only contract changed: %#v", textResults)
	}

	imageResults, _, err := buildEmbeddingResults(EmbeddingRequest{
		Images: []string{"placeholder"}, Model: "auto", Dimension: defaultImageEmbeddingDimension,
	})
	if err != nil {
		t.Fatalf("image-only result: %v", err)
	}
	if len(imageResults) != 1 || imageResults[0].ModelUsed != multimodalEmbeddingModel || imageResults[0].Modality != "image" {
		t.Fatalf("image-only contract changed: %#v", imageResults)
	}
	if legacyTextCalls != 1 || multimodalTextCalls != 0 || imageCalls != 1 {
		t.Fatalf("single-modality encoder calls = legacy:%d multimodal-text:%d image:%d", legacyTextCalls, multimodalTextCalls, imageCalls)
	}
}
