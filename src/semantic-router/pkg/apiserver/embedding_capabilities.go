//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
)

const (
	defaultImageEmbeddingDimension     = 384
	defaultMixedEmbeddingDimension     = 256
	unsupportedEmbeddingCapabilityCode = "UNSUPPORTED_CAPABILITY"
)

// embeddingModelCapability is the API-visible subset of a native embedding
// model's contract. Keeping this small and local prevents request admission
// from depending on backend-specific Go types while go.onnx.mod replaces the
// Candle module with the ONNX module at build time.
type embeddingModelCapability struct {
	dimensions          []int
	supportsTargetLayer bool
}

// embeddingBackendCapabilities describes only behavior that the selected
// native backend can honor. The backend-specific constructors live in files
// selected by the same `onnx` build tag used by the router binary.
type embeddingBackendCapabilities struct {
	name                    string
	models                  map[string]embeddingModelCapability
	autoModels              []string
	autoSupportsPriorities  bool
	autoSupportsLayer       bool
	supportsMultimodalText  bool
	supportsMultimodalImage bool
	multimodalDimensions    []int
}

func (c embeddingBackendCapabilities) supportedModels() []string {
	models := make([]string, 0, len(c.models)+1)
	models = append(models, "auto")
	for model := range c.models {
		models = append(models, model)
	}
	sort.Strings(models[1:])
	return models
}

func (c embeddingBackendCapabilities) supportsImageDimension(dimension int) bool {
	return c.supportsMultimodalImage && containsEmbeddingDimension(c.multimodalDimensions, dimension)
}

func (c embeddingBackendCapabilities) supportsAutoDimension(dimension int) bool {
	if len(c.autoModels) == 0 {
		return false
	}
	for _, model := range c.autoModels {
		capability, ok := c.models[model]
		if !ok || !containsEmbeddingDimension(capability.dimensions, dimension) {
			return false
		}
	}
	return true
}

func (c embeddingBackendCapabilities) autoDimensions() []int {
	if len(c.autoModels) == 0 {
		return nil
	}
	first, ok := c.models[c.autoModels[0]]
	if !ok {
		return nil
	}
	dimensions := make([]int, 0, len(first.dimensions))
	for _, dimension := range first.dimensions {
		if c.supportsAutoDimension(dimension) {
			dimensions = append(dimensions, dimension)
		}
	}
	sort.Ints(dimensions)
	return dimensions
}

func containsEmbeddingDimension(dimensions []int, target int) bool {
	for _, dimension := range dimensions {
		if dimension == target {
			return true
		}
	}
	return false
}

func formatDimensionList(dimensions []int) string {
	ordered := append([]int(nil), dimensions...)
	sort.Ints(ordered)
	parts := make([]string, len(ordered))
	for i, dimension := range ordered {
		parts[i] = strconv.Itoa(dimension)
	}
	return strings.Join(parts, ", ")
}

func validateTextEmbeddingCapabilities(
	model string,
	dimension int,
	targetLayer int,
	qualityPriority float32,
	latencyPriority float32,
	mmbertLayers []int,
) (string, string, bool) {
	capabilities := nativeEmbeddingBackendCapabilities()
	if model == "" {
		model = "auto"
	}

	if !validSimilarityPriority(qualityPriority) || !validSimilarityPriority(latencyPriority) {
		return "INVALID_PARAMETER", "quality_priority and latency_priority must be finite values between 0 and 1", false
	}
	if model == "auto" {
		return validateAutoEmbeddingCapabilities(
			capabilities, dimension, targetLayer, qualityPriority, latencyPriority, mmbertLayers,
		)
	}
	return validateExplicitEmbeddingCapabilities(
		capabilities, model, dimension, targetLayer, qualityPriority, latencyPriority, mmbertLayers,
	)
}

func validateAutoEmbeddingCapabilities(
	capabilities embeddingBackendCapabilities,
	dimension int,
	targetLayer int,
	qualityPriority float32,
	latencyPriority float32,
	mmbertLayers []int,
) (string, string, bool) {
	if !capabilities.supportsAutoDimension(dimension) {
		return "INVALID_DIMENSION", fmt.Sprintf(
			"dimension %d is not safe for auto routing on the %s backend; dimensions supported by every auto candidate: %s",
			dimension,
			capabilities.name,
			formatDimensionList(capabilities.autoDimensions()),
		), false
	}
	if (qualityPriority != 0 || latencyPriority != 0) && !capabilities.autoSupportsPriorities {
		return "INVALID_PARAMETER", fmt.Sprintf(
			"quality_priority and latency_priority are not supported by auto routing on the %s backend",
			capabilities.name,
		), false
	}
	if targetLayer == 0 {
		return "", "", true
	}
	if !capabilities.autoSupportsLayer {
		return "INVALID_PARAMETER", "target_layer is only supported for model='mmbert'", false
	}
	return validateMmBertTargetLayer(targetLayer, mmbertLayers)
}

func validateExplicitEmbeddingCapabilities(
	capabilities embeddingBackendCapabilities,
	model string,
	dimension int,
	targetLayer int,
	qualityPriority float32,
	latencyPriority float32,
	mmbertLayers []int,
) (string, string, bool) {
	capability, ok := capabilities.models[model]
	if !ok {
		return "INVALID_MODEL", fmt.Sprintf(
			"model must be one of: %s for the %s backend",
			strings.Join(capabilities.supportedModels(), ", "),
			capabilities.name,
		), false
	}
	if !containsEmbeddingDimension(capability.dimensions, dimension) {
		return "INVALID_DIMENSION", fmt.Sprintf(
			"dimension %d is not supported for model='%s' on the %s backend; supported dimensions: %s",
			dimension,
			model,
			capabilities.name,
			formatDimensionList(capability.dimensions),
		), false
	}
	if qualityPriority != 0 || latencyPriority != 0 {
		return "INVALID_PARAMETER", "quality_priority and latency_priority are only supported for model='auto'", false
	}
	if targetLayer == 0 {
		return "", "", true
	}
	if !capability.supportsTargetLayer {
		return "INVALID_PARAMETER", "target_layer is only supported for model='mmbert'", false
	}
	return validateMmBertTargetLayer(targetLayer, mmbertLayers)
}

func validateMmBertTargetLayer(targetLayer int, mmbertLayers []int) (string, string, bool) {
	if targetLayer < 0 || !containsEmbeddingDimension(mmbertLayers, targetLayer) {
		return "INVALID_LAYER", fmt.Sprintf(
			"target_layer must be one of: %s (got %d)",
			formatLayerList(mmbertLayers),
			targetLayer,
		), false
	}
	return "", "", true
}

// validateMixedEmbeddingCapabilities keeps mixed text+image requests in one
// backend-declared multimodal space. It deliberately rejects every text-only
// routing control because accepting one would make the API imply behavior the
// multimodal encoder cannot honor.
func validateMixedEmbeddingCapabilities(
	capabilities embeddingBackendCapabilities,
	req EmbeddingRequest,
) (string, string, bool) {
	if req.Model != "" && req.Model != "auto" {
		return "INVALID_MODEL", "mixed text and image embedding requests use the multimodal model; model must be omitted or 'auto'", false
	}
	if req.TargetLayer != 0 || req.QualityPriority != 0 || req.LatencyPriority != 0 {
		return "INVALID_PARAMETER", "target_layer and routing priorities are not supported for mixed text and image embedding requests", false
	}
	if !capabilities.supportsMultimodalText || !capabilities.supportsMultimodalImage {
		return unsupportedEmbeddingCapabilityCode, fmt.Sprintf(
			"mixed text and image embeddings are not supported by the %s backend",
			capabilities.name,
		), false
	}
	if !containsEmbeddingDimension(capabilities.multimodalDimensions, req.Dimension) {
		return "INVALID_DIMENSION", fmt.Sprintf(
			"dimension %d is not supported for mixed text and image embeddings on the %s backend; supported dimensions: %s (default: %d)",
			req.Dimension,
			capabilities.name,
			formatDimensionList(capabilities.multimodalDimensions),
			defaultMixedEmbeddingDimension,
		), false
	}
	return "", "", true
}

func validateImageEmbeddingDimension(dimension int) (string, string, bool) {
	capabilities := nativeEmbeddingBackendCapabilities()
	if !capabilities.supportsMultimodalImage {
		return unsupportedEmbeddingCapabilityCode, fmt.Sprintf(
			"image embeddings are not supported by the %s backend",
			capabilities.name,
		), false
	}
	if capabilities.supportsImageDimension(dimension) {
		return "", "", true
	}
	return "INVALID_DIMENSION", fmt.Sprintf(
		"dimension %d is not supported for image embeddings on the %s backend; supported dimensions: %s (default: %d)",
		dimension,
		capabilities.name,
		formatDimensionList(capabilities.multimodalDimensions),
		defaultImageEmbeddingDimension,
	), false
}
