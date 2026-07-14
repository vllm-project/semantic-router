//go:build !windows && cgo

package apiserver

import (
	"errors"
	"fmt"
	"net/http"
	"strings"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

// Native entry points are package variables so admission and error-path tests
// can remain deterministic without loading models.
var (
	multiModalEncodeText      = candle_binding.MultiModalEncodeText
	multiModalEncodeImage     = candle_binding.MultiModalEncodeImageFromBase64
	embeddingOutputForRequest = embeddingOutput
)

const multimodalEmbeddingModel = "multi-modal-embed"

// imageEncodeError marks an image-encode failure driven by the request input.
// It is constructed only when the binding returns ErrInvalidImageInput.
// The handler maps it to 400 INVALID_IMAGE rather than 500, so a client-supplied
// bad image is reported as a client error, not an internal one.
type imageEncodeError struct {
	index int
	err   error
}

func (e *imageEncodeError) Error() string {
	return fmt.Sprintf("images[%d]: %v", e.index, e.err)
}

func (e *imageEncodeError) Unwrap() error { return e.err }

// embeddingOutputContractError marks a successful native call whose metadata
// or vector shape violates the request contract. This is an internal/backend
// failure, never a client error: admission has already proven the request is
// supported by the selected backend.
type embeddingOutputContractError struct {
	input    string
	expected string
	actual   string
}

func (e *embeddingOutputContractError) Error() string {
	return fmt.Sprintf("%s embedding contract mismatch: expected %s, got %s", e.input, e.expected, e.actual)
}

// classifyEmbeddingError maps a buildEmbeddingResults error to the HTTP status,
// error code, and client message. Input-caused image-encode failures are 400
// INVALID_IMAGE; every other failure is a genuine 500.
func classifyEmbeddingError(err error) (int, string, string) {
	if errors.Is(err, candle_binding.ErrEmbeddingInputTooLong) {
		return http.StatusRequestEntityTooLarge, embeddingInputTooLargeCode,
			"embedding input exceeds the selected model context"
	}
	var imgErr *imageEncodeError
	if errors.As(err, &imgErr) && errors.Is(imgErr, candle_binding.ErrInvalidImageInput) {
		return http.StatusBadRequest, "INVALID_IMAGE",
			fmt.Sprintf("images[%d] could not be decoded as an image", imgErr.index)
	}
	return http.StatusInternalServerError, "EMBEDDING_GENERATION_FAILED",
		"failed to generate embedding"
}

func buildEmbeddingResults(req EmbeddingRequest) ([]EmbeddingResult, int64, error) {
	results := make([]EmbeddingResult, 0, len(req.Texts)+len(req.Images))
	var totalProcessingTime int64
	useMultimodalText := isMixedEmbeddingRequest(req)

	for i, text := range req.Texts {
		result, processingTime, err := buildTextEmbeddingResult(req, i, text, useMultimodalText)
		if err != nil {
			return nil, 0, err
		}
		results = append(results, result)
		totalProcessingTime += processingTime
	}

	for i, image := range req.Images {
		result, processingTime, err := buildImageEmbeddingResult(req.Dimension, i, image)
		if err != nil {
			return nil, 0, err
		}
		results = append(results, result)
		totalProcessingTime += processingTime
	}

	return results, totalProcessingTime, nil
}

func buildTextEmbeddingResult(
	req EmbeddingRequest,
	index int,
	text string,
	useMultimodal bool,
) (EmbeddingResult, int64, error) {
	if useMultimodal {
		output, err := multiModalEncodeText(text, req.Dimension)
		if err != nil {
			return EmbeddingResult{}, 0, fmt.Errorf("texts[%d] embedding failed: %w", index, err)
		}
		if err := validateMultimodalEmbeddingOutput(
			fmt.Sprintf("texts[%d]", index), "text", req.Dimension, output,
		); err != nil {
			return EmbeddingResult{}, 0, err
		}

		processingTime := int64(output.ProcessingTimeMs)
		return EmbeddingResult{
			Text:             text,
			Modality:         output.Modality,
			Embedding:        output.Embedding,
			Dimension:        len(output.Embedding),
			ModelUsed:        multimodalEmbeddingModel,
			ProcessingTimeMs: processingTime,
		}, processingTime, nil
	}

	output, err := embeddingOutputForRequest(req, text)
	if err != nil {
		return EmbeddingResult{}, 0, err
	}
	if err := validateTextEmbeddingOutput(req, output); err != nil {
		return EmbeddingResult{}, 0, err
	}

	processingTime := int64(output.ProcessingTimeMs)
	return EmbeddingResult{
		Text:             text,
		Embedding:        output.Embedding,
		Dimension:        len(output.Embedding),
		ModelUsed:        output.ModelType,
		ProcessingTimeMs: processingTime,
	}, processingTime, nil
}

func buildImageEmbeddingResult(dimension, index int, image string) (EmbeddingResult, int64, error) {
	// Canonicalize so the FFI's case-sensitive ";base64," scan finds the
	// payload boundary (validation already guaranteed a safe data URI).
	encodeInput := image
	if canonical, ok := imageurl.CanonicalDataURL(image); ok {
		encodeInput = canonical
	}
	output, err := multiModalEncodeImage(encodeInput, dimension)
	if err != nil {
		if errors.Is(err, candle_binding.ErrInvalidImageInput) {
			return EmbeddingResult{}, 0, &imageEncodeError{index: index, err: err}
		}
		return EmbeddingResult{}, 0, fmt.Errorf("images[%d] embedding failed: %w", index, err)
	}
	if err := validateImageEmbeddingOutput(index, dimension, output); err != nil {
		return EmbeddingResult{}, 0, err
	}

	processingTime := int64(output.ProcessingTimeMs)
	return EmbeddingResult{
		Modality:         output.Modality,
		Embedding:        output.Embedding,
		Dimension:        len(output.Embedding),
		ModelUsed:        multimodalEmbeddingModel,
		ProcessingTimeMs: processingTime,
	}, processingTime, nil
}

func validateTextEmbeddingOutput(req EmbeddingRequest, output *candle_binding.EmbeddingOutput) error {
	if output == nil {
		return &embeddingOutputContractError{
			input:    "text",
			expected: fmt.Sprintf("a non-nil %d-dimensional result", req.Dimension),
			actual:   "nil",
		}
	}
	if len(output.Embedding) != req.Dimension {
		return &embeddingOutputContractError{
			input:    "text",
			expected: fmt.Sprintf("dimension %d", req.Dimension),
			actual:   fmt.Sprintf("dimension %d", len(output.Embedding)),
		}
	}

	capabilities := nativeEmbeddingBackendCapabilities()
	if req.Model == "auto" || req.Model == "" {
		for _, model := range capabilities.autoModels {
			if output.ModelType == model {
				return nil
			}
		}
		return &embeddingOutputContractError{
			input:    "text",
			expected: fmt.Sprintf("an auto-routed model in [%s]", strings.Join(capabilities.autoModels, ", ")),
			actual:   fmt.Sprintf("model %q", output.ModelType),
		}
	}
	if output.ModelType != req.Model {
		return &embeddingOutputContractError{
			input:    "text",
			expected: fmt.Sprintf("model %q", req.Model),
			actual:   fmt.Sprintf("model %q", output.ModelType),
		}
	}
	return nil
}

func validateImageEmbeddingOutput(index, dimension int, output *candle_binding.MultiModalEmbeddingOutput) error {
	return validateMultimodalEmbeddingOutput(fmt.Sprintf("images[%d]", index), "image", dimension, output)
}

func validateMultimodalEmbeddingOutput(
	input string,
	modality string,
	dimension int,
	output *candle_binding.MultiModalEmbeddingOutput,
) error {
	if output == nil {
		return &embeddingOutputContractError{
			input:    input,
			expected: fmt.Sprintf("a non-nil %d-dimensional %s result", dimension, modality),
			actual:   "nil",
		}
	}
	if output.Modality != modality {
		return &embeddingOutputContractError{
			input:    input,
			expected: fmt.Sprintf("modality %s", modality),
			actual:   fmt.Sprintf("modality %q", output.Modality),
		}
	}
	if len(output.Embedding) != dimension {
		return &embeddingOutputContractError{
			input:    input,
			expected: fmt.Sprintf("dimension %d", dimension),
			actual:   fmt.Sprintf("dimension %d", len(output.Embedding)),
		}
	}
	return nil
}

func embeddingOutput(req EmbeddingRequest, text string) (*candle_binding.EmbeddingOutput, error) {
	switch req.Model {
	case "auto", "":
		// ONNX auto resolves to mmBERT. Its metadata API has no target-layer
		// parameter, so use the explicit 2D entry point when admission allows
		// auto + target_layer (Candle admission rejects that combination).
		if req.TargetLayer != 0 {
			return candle_binding.GetEmbedding2DMatryoshka(text, "mmbert", req.TargetLayer, req.Dimension)
		}
		return candle_binding.GetEmbeddingWithMetadata(text, req.QualityPriority, req.LatencyPriority, req.Dimension)
	case "mmbert":
		return candle_binding.GetEmbedding2DMatryoshka(text, req.Model, req.TargetLayer, req.Dimension)
	default:
		return candle_binding.GetEmbeddingWithModelType(text, req.Model, req.Dimension)
	}
}
