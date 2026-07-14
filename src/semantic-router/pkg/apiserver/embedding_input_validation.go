//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"math"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

const (
	maxEmbeddingInputs           = 100
	maxEmbeddingTextBytes        = 128 * 1024
	maxEmbeddingTextRunes        = 131_072
	maxEmbeddingTotalTextBytes   = 1024 * 1024
	maxBatchSimilarityCandidates = 100
	maxImagesPerRequest          = imageurl.MaxImagePartsPerRequest
	embeddingInputTooLargeCode   = "EMBEDDING_INPUT_TOO_LARGE"
)

func embeddingValidationStatus(code string) int {
	if code == embeddingInputTooLargeCode {
		return http.StatusRequestEntityTooLarge
	}
	return http.StatusBadRequest
}

func validateEmbeddingRequest(req EmbeddingRequest, mmbertLayers []int) (string, string, bool) {
	if code, message, ok := validateEmbeddingRequestShape(req, mmbertLayers); !ok {
		return code, message, false
	}
	return validateEmbeddingImages(req.Images)
}

func validateEmbeddingRequestShape(req EmbeddingRequest, mmbertLayers []int) (string, string, bool) {
	if code, message, ok := validateEmbeddingInputShape(req); !ok {
		return code, message, false
	}
	if code, message, ok := validateEmbeddingRequestControls(req, mmbertLayers); !ok {
		return code, message, false
	}
	if isMixedEmbeddingRequest(req) {
		// Mixed capability validation already checked the shared multimodal
		// dimension; do not reinterpret it as an image-only request here.
		return "", "", true
	}
	if len(req.Images) > 0 {
		return validateImageEmbeddingDimension(req.Dimension)
	}
	return "", "", true
}

func validateEmbeddingInputShape(req EmbeddingRequest) (string, string, bool) {
	if len(req.Texts) == 0 && len(req.Images) == 0 {
		return "INVALID_INPUT", "at least one of texts or images must be provided", false
	}
	if len(req.Texts)+len(req.Images) > maxEmbeddingInputs {
		return embeddingInputTooLargeCode, "embedding request exceeds the maximum number of inputs", false
	}
	if code, message, ok := validateEmbeddingTexts(req.Texts); !ok {
		return code, message, false
	}
	if code, message, ok := validateEmbeddingImageShapes(req.Images); !ok {
		return code, message, false
	}
	return "", "", true
}

func validateEmbeddingRequestControls(req EmbeddingRequest, mmbertLayers []int) (string, string, bool) {
	if isMixedEmbeddingRequest(req) {
		return validateMixedEmbeddingCapabilities(nativeEmbeddingBackendCapabilities(), req)
	}
	if len(req.Texts) > 0 {
		return validateTextEmbeddingCapabilities(
			req.Model,
			req.Dimension,
			req.TargetLayer,
			req.QualityPriority,
			req.LatencyPriority,
			mmbertLayers,
		)
	}
	// Image inference always uses the multimodal model. Reject text-only
	// controls instead of accepting values that the native call will ignore.
	if req.Model != "" && req.Model != "auto" {
		return "INVALID_MODEL", "image-only embedding requests use the multimodal model; model must be omitted or 'auto'", false
	}
	if req.TargetLayer != 0 || req.QualityPriority != 0 || req.LatencyPriority != 0 {
		return "INVALID_PARAMETER", "target_layer and routing priorities are not supported for image-only embedding requests", false
	}
	return "", "", true
}

func isMixedEmbeddingRequest(req EmbeddingRequest) bool {
	return len(req.Texts) > 0 && len(req.Images) > 0
}

func validateEmbeddingImageShapes(images []string) (string, string, bool) {
	if len(images) > maxImagesPerRequest {
		return embeddingInputTooLargeCode, fmt.Sprintf("at most %d images may be provided per request", maxImagesPerRequest), false
	}
	for i, image := range images {
		if !imageurl.IsSafeImageDataURL(image) {
			return "INVALID_IMAGE", fmt.Sprintf("images[%d] must be an inline base64 image data URI (data:image/<type>;base64,...)", i), false
		}
		if !imageurl.IsJPEGOrPNGDataURLShape(image) {
			return "INVALID_IMAGE", fmt.Sprintf("images[%d] must use JPEG or PNG image data", i), false
		}
	}
	return "", "", true
}

func validateEmbeddingTexts(texts []string) (string, string, bool) {
	totalBytes := 0
	for _, text := range texts {
		if len(text) > maxEmbeddingTextBytes || utf8.RuneCountInString(text) > maxEmbeddingTextRunes {
			return embeddingInputTooLargeCode, "text input exceeds the per-input size limit", false
		}
		if !validEmbeddingText(text) {
			return "INVALID_INPUT", "text inputs must contain valid non-empty Unicode without NUL bytes", false
		}
		totalBytes += len(text)
		if totalBytes > maxEmbeddingTotalTextBytes {
			return embeddingInputTooLargeCode, "combined text input exceeds the total size limit", false
		}
	}
	return "", "", true
}

func validEmbeddingText(text string) bool {
	return strings.TrimSpace(text) != "" &&
		!strings.ContainsRune(text, '\x00') &&
		utf8.ValidString(text)
}

func validateEmbeddingImages(images []string) (string, string, bool) {
	var imageBudget imageurl.RequestImageBudget
	for i, image := range images {
		if _, ok := imageurl.InspectJPEGOrPNGDataURL(image, &imageBudget); !ok {
			return "INVALID_IMAGE", fmt.Sprintf("images[%d] must contain a decodable JPEG or PNG image within the supported limits", i), false
		}
	}
	return "", "", true
}

func validateSimilarityTexts(req SimilarityRequest) (string, string, bool) {
	return validateEmbeddingTexts([]string{req.Text1, req.Text2})
}

func validateSimilarityRequest(req SimilarityRequest, mmbertLayers []int) (string, string, bool) {
	if code, message, ok := validateSimilarityTexts(req); !ok {
		return code, message, false
	}
	return validateSimilarityOptions(
		req.Model,
		req.Dimension,
		req.TargetLayer,
		req.QualityPriority,
		req.LatencyPriority,
		mmbertLayers,
	)
}

func validateBatchSimilarityRequest(req BatchSimilarityRequest, mmbertLayers []int) (string, string, bool) {
	if req.Query == "" {
		return "INVALID_INPUT", "query must be provided", false
	}
	if len(req.Candidates) == 0 {
		return "INVALID_INPUT", "candidates array cannot be empty", false
	}
	if len(req.Candidates) > maxBatchSimilarityCandidates {
		return embeddingInputTooLargeCode, "candidates array exceeds the maximum size", false
	}
	texts := make([]string, 0, len(req.Candidates)+1)
	texts = append(texts, req.Query)
	texts = append(texts, req.Candidates...)
	if code, message, ok := validateEmbeddingTexts(texts); !ok {
		return code, message, false
	}
	if req.TopK < 0 {
		return "INVALID_INPUT", "top_k cannot be negative", false
	}
	return validateSimilarityOptions(
		req.Model,
		req.Dimension,
		req.TargetLayer,
		req.QualityPriority,
		req.LatencyPriority,
		mmbertLayers,
	)
}

func validateSimilarityOptions(
	model string,
	dimension int,
	targetLayer int,
	qualityPriority float32,
	latencyPriority float32,
	mmbertLayers []int,
) (string, string, bool) {
	return validateTextEmbeddingCapabilities(
		model,
		dimension,
		targetLayer,
		qualityPriority,
		latencyPriority,
		mmbertLayers,
	)
}

func validSimilarityPriority(priority float32) bool {
	value := float64(priority)
	return !math.IsNaN(value) && !math.IsInf(value, 0) && priority >= 0 && priority <= 1
}
