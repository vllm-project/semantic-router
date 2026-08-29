package llmprotocol

import (
	"encoding/base64"
	"io"
	"strings"
)

func validateImageGenerationOptions(options *ImageGenerationOptions, limits Limits) error {
	if options == nil {
		return nil
	}
	if err := validateImageGenerationEnum("quality", options.Quality, "", "low", "medium", "high", "auto"); err != nil {
		return err
	}
	if err := validateImageGenerationEnum("output_format", options.OutputFormat, "", "png", "webp", "jpeg"); err != nil {
		return err
	}
	if err := validateImageGenerationEnum("moderation", options.Moderation, "", "auto", "low"); err != nil {
		return err
	}
	if err := validateImageGenerationEnum("background", options.Background, "", "transparent", "opaque", "auto"); err != nil {
		return err
	}
	if err := validateImageGenerationEnum("input_fidelity", options.InputFidelity, "", "low", "high"); err != nil {
		return err
	}
	if err := validateImageGenerationEnum("action", options.Action, "", "generate", "edit", "auto"); err != nil {
		return err
	}
	if options.OutputCompression != nil && (*options.OutputCompression < 0 || *options.OutputCompression > 100) {
		return NewError(ErrorInvalidRequest, "invalid_image_output_compression", "image output compression must be between 0 and 100", nil)
	}
	if options.PartialImages != nil && (*options.PartialImages < 0 || *options.PartialImages > 3) {
		return NewError(ErrorInvalidRequest, "invalid_partial_images", "partial image count must be between 0 and 3", nil)
	}
	for _, value := range []string{options.Model, options.Size, options.Quality, options.OutputFormat, options.Moderation, options.Background, options.InputFidelity, options.Action} {
		if exceeds(value, limits.MediaReferenceBytes) {
			return NewError(ErrorInvalidRequest, "image_generation_field_limit", "image generation option exceeds the configured limit", nil)
		}
	}
	if options.InputImageMask != nil {
		if exceeds(options.InputImageMask.EncodedImage, limits.MediaDataBytes) || exceeds(options.InputImageMask.FileID, limits.MediaReferenceBytes) {
			return NewError(ErrorInvalidRequest, "image_generation_mask_limit", "image generation mask exceeds the configured limit", nil)
		}
		if err := validateImageGenerationMaskData(options.InputImageMask.EncodedImage); err != nil {
			return err
		}
	}
	return nil
}

func validateImageGenerationMaskData(value string) error {
	if value == "" {
		return nil
	}
	encoded := value
	if len(value) >= len("data:") && strings.EqualFold(value[:len("data:")], "data:") {
		comma := strings.IndexByte(value, ',')
		if comma < 0 || !strings.HasSuffix(strings.ToLower(value[:comma]), ";base64") {
			return NewError(ErrorInvalidRequest, "invalid_image_generation_mask", "image generation mask must be base64 data", nil)
		}
		encoded = value[comma+1:]
	}
	if err := validateGeneratedImageBase64(encoded); err != nil {
		return NewError(ErrorInvalidRequest, "invalid_image_generation_mask", "image generation mask must be base64 data", err)
	}
	return nil
}

func validateImageGenerationEnum(field, value string, allowed ...string) error {
	for _, candidate := range allowed {
		if value == candidate {
			return nil
		}
	}
	return NewError(ErrorInvalidRequest, "invalid_image_generation_"+field, "image generation "+field+" is invalid", nil)
}

func validateGeneratedImageContent(content Content, limits Limits) error {
	image := content.GeneratedImage
	if content.Text != "" || content.ToolCall != nil || content.ToolResult != nil || content.URL != "" ||
		content.Data != "" || content.FileID != "" || content.Filename != "" || content.MediaType != "" ||
		content.Detail != "" || content.Signature != "" || content.Reasoning != "" || content.Cache != nil ||
		len(content.Citations) != 0 {
		return NewError(ErrorInvalidRequest, "invalid_generated_image", "generated image content contains fields from another content kind", nil)
	}
	if err := ValidateGeneratedImage(image, limits); err != nil {
		return err
	}
	if image.PartialIndex != nil || image.PartialImage != "" || image.Size != "" || image.Quality != "" ||
		image.Background != "" || image.OutputFormat != "" {
		return NewError(ErrorInvalidRequest, "invalid_generated_image", "generated image output contains progress-only fields", nil)
	}
	return nil
}

// ValidateGeneratedImage applies the bounded neutral image-generation
// lifecycle contract to buffered content and stream progress events.
func ValidateGeneratedImage(image *GeneratedImage, limits Limits) error {
	if image == nil {
		return NewError(ErrorInvalidRequest, "generated_image_required", "generated image content requires image-generation state", nil)
	}
	if !validImageGenerationStatus(image.Status) {
		return NewError(ErrorInvalidRequest, "invalid_image_generation_status", "image generation status is invalid", nil)
	}
	if image.PartialIndex != nil && *image.PartialIndex < 0 {
		return NewError(ErrorInvalidRequest, "invalid_partial_image_index", "partial image index cannot be negative", nil)
	}
	if image.PartialImage != "" && image.PartialIndex == nil {
		return NewError(ErrorInvalidRequest, "partial_image_index_required", "partial image data requires an index", nil)
	}
	if limits.MediaDataBytes > 0 && image.Result != nil && len(*image.Result) > limits.MediaDataBytes {
		return NewError(ErrorInvalidRequest, "generated_image_limit", "generated image data exceeds the configured limit", nil)
	}
	if limits.MediaDataBytes > 0 && len(image.PartialImage) > limits.MediaDataBytes {
		return NewError(ErrorInvalidRequest, "generated_image_limit", "generated image data exceeds the configured limit", nil)
	}
	if image.Result != nil {
		if err := validateGeneratedImageBase64(*image.Result); err != nil {
			return err
		}
	}
	if image.PartialIndex != nil {
		if err := validateGeneratedImageBase64(image.PartialImage); err != nil {
			return err
		}
	}
	for _, value := range []string{image.Size, image.Quality, image.Background, image.OutputFormat} {
		if exceeds(value, limits.MediaReferenceBytes) {
			return NewError(ErrorInvalidRequest, "generated_image_field_limit", "generated image metadata exceeds the configured limit", nil)
		}
	}
	return nil
}

func validateGeneratedImageBase64(value string) error {
	decoder := base64.NewDecoder(base64.StdEncoding.Strict(), strings.NewReader(value))
	if _, err := io.Copy(io.Discard, decoder); err != nil {
		return NewError(ErrorInvalidRequest, "invalid_generated_image_data", "generated image data must be valid base64", err)
	}
	return nil
}

func validImageGenerationStatus(status ImageGenerationStatus) bool {
	switch status {
	case ImageGenerationInProgress, ImageGenerationGenerating, ImageGenerationCompleted, ImageGenerationFailed:
		return true
	default:
		return false
	}
}
