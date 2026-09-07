package llmprotocol

import (
	"strings"
	"testing"
)

func TestImageGenerationRequestValidationAndCapabilities(t *testing.T) {
	compression, partialImages := int64(72), int64(3)
	request := validSemanticRequest()
	request.Tools = nil
	request.ImageGeneration = &ImageGenerationOptions{
		Model: "gpt-image-1", Quality: "high", Size: "1536x1024",
		OutputFormat: "webp", OutputCompression: &compression,
		Moderation: "low", Background: "transparent", InputFidelity: "high",
		InputImageMask: &ImageGenerationMask{
			EncodedImage: "data:image/png;base64,bWFzaw==", FileID: "file_mask",
		},
		PartialImages: &partialImages, Action: "edit",
	}
	request.ToolChoice = ToolChoice{Mode: ToolChoiceImageGeneration}
	if err := ValidateRequest(request, DefaultPolicy().Limits); err != nil {
		t.Fatalf("valid image-generation request rejected: %v", err)
	}
	required := RequiredCapabilities(request)
	if !required.Supports(CapabilityImageGeneration) || !required.Supports(CapabilityTools) {
		t.Fatalf("required capabilities = %v", required.Names())
	}

	response := Response{
		Generation: 1, ID: "resp_image", Model: "image-model", StopReason: StopEndTurn,
		Output: []OutputItem{{
			ID: "ig_1", Role: RoleAssistant,
			Content: []Content{{Kind: ContentGeneratedImage, GeneratedImage: &GeneratedImage{
				Status: ImageGenerationCompleted, Result: imageString("ZmluYWw="),
			}}},
		}},
		Usage: Usage{State: UsageUnavailable},
	}
	if err := ValidateResponse(response, DefaultPolicy().Limits); err != nil {
		t.Fatalf("valid generated-image response rejected: %v", err)
	}
	if required := RequiredResponseCapabilities(response); !required.Supports(CapabilityImageGeneration) {
		t.Fatalf("response capabilities = %v", required.Names())
	}
	event := Event{Type: EventImageGenerationProgress, GeneratedImage: &GeneratedImage{Status: ImageGenerationGenerating}}
	if required := RequiredEventCapabilities(event); !required.Supports(CapabilityImageGeneration) {
		t.Fatalf("event capabilities = %v", required.Names())
	}
}

func TestImageGenerationRequestRejectsInvalidOptions(t *testing.T) {
	negative, tooLarge := int64(-1), int64(101)
	tooManyPartials := int64(4)
	tests := []struct {
		name string
		code string
		edit func(*Request)
	}{
		{name: "choice without tool", code: "image_generation_tool_required", edit: func(request *Request) {
			request.ImageGeneration = nil
		}},
		{name: "quality", code: "invalid_image_generation_quality", edit: func(request *Request) {
			request.ImageGeneration.Quality = "ultra"
		}},
		{name: "output format", code: "invalid_image_generation_output_format", edit: func(request *Request) {
			request.ImageGeneration.OutputFormat = "gif"
		}},
		{name: "moderation", code: "invalid_image_generation_moderation", edit: func(request *Request) {
			request.ImageGeneration.Moderation = "strict"
		}},
		{name: "background", code: "invalid_image_generation_background", edit: func(request *Request) {
			request.ImageGeneration.Background = "solid"
		}},
		{name: "input fidelity", code: "invalid_image_generation_input_fidelity", edit: func(request *Request) {
			request.ImageGeneration.InputFidelity = "auto"
		}},
		{name: "action", code: "invalid_image_generation_action", edit: func(request *Request) {
			request.ImageGeneration.Action = "transform"
		}},
		{name: "negative compression", code: "invalid_image_output_compression", edit: func(request *Request) {
			request.ImageGeneration.OutputCompression = &negative
		}},
		{name: "compression over one hundred", code: "invalid_image_output_compression", edit: func(request *Request) {
			request.ImageGeneration.OutputCompression = &tooLarge
		}},
		{name: "negative partial count", code: "invalid_partial_images", edit: func(request *Request) {
			request.ImageGeneration.PartialImages = &negative
		}},
		{name: "too many partial images", code: "invalid_partial_images", edit: func(request *Request) {
			request.ImageGeneration.PartialImages = &tooManyPartials
		}},
		{name: "bounded option", code: "image_generation_field_limit", edit: func(request *Request) {
			request.ImageGeneration.Model = strings.Repeat("m", 9)
		}},
		{name: "bounded mask", code: "image_generation_mask_limit", edit: func(request *Request) {
			request.ImageGeneration.InputImageMask = &ImageGenerationMask{FileID: strings.Repeat("f", 9)}
		}},
		{name: "invalid mask data", code: "invalid_image_generation_mask", edit: func(request *Request) {
			request.ImageGeneration.InputImageMask = &ImageGenerationMask{EncodedImage: "https://example.com/mask.png"}
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := validSemanticRequest()
			request.Tools = nil
			request.ImageGeneration = &ImageGenerationOptions{}
			request.ToolChoice = ToolChoice{Mode: ToolChoiceImageGeneration}
			test.edit(&request)
			limits := DefaultPolicy().Limits
			limits.MediaReferenceBytes = 8
			requireLLMProtocolErrorCode(t, ValidateRequest(request, limits), test.code)
		})
	}
}

func TestValidateGeneratedImageClosesLifecycleAndBounds(t *testing.T) {
	negative := int64(-1)
	zero := int64(0)
	validResult := "ZmluYWw="
	tests := []struct {
		name  string
		image *GeneratedImage
		code  string
	}{
		{name: "missing", code: "generated_image_required"},
		{name: "unknown status", image: &GeneratedImage{Status: "queued"}, code: "invalid_image_generation_status"},
		{name: "negative partial index", image: &GeneratedImage{Status: ImageGenerationGenerating, PartialIndex: &negative}, code: "invalid_partial_image_index"},
		{name: "partial data without index", image: &GeneratedImage{Status: ImageGenerationGenerating, PartialImage: "YQ=="}, code: "partial_image_index_required"},
		{name: "invalid result base64", image: &GeneratedImage{Status: ImageGenerationCompleted, Result: imageString("not-base64")}, code: "invalid_generated_image_data"},
		{name: "invalid partial base64", image: &GeneratedImage{Status: ImageGenerationGenerating, PartialIndex: &zero, PartialImage: "not-base64"}, code: "invalid_generated_image_data"},
		{name: "result limit", image: &GeneratedImage{Status: ImageGenerationCompleted, Result: &validResult}, code: "generated_image_limit"},
		{name: "metadata limit", image: &GeneratedImage{Status: ImageGenerationGenerating, Quality: "quality-too-long"}, code: "generated_image_field_limit"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			limits := DefaultPolicy().Limits
			if test.name == "result limit" {
				limits.MediaDataBytes = len(validResult) - 1
			} else {
				limits.MediaDataBytes = 1 << 20
			}
			limits.MediaReferenceBytes = 8
			requireLLMProtocolErrorCode(t, ValidateGeneratedImage(test.image, limits), test.code)
		})
	}

	for _, image := range []*GeneratedImage{
		{Status: ImageGenerationCompleted, Result: imageString("")},
		{Status: ImageGenerationFailed},
		{Status: ImageGenerationGenerating, PartialIndex: &zero, PartialImage: "YQ=="},
	} {
		if err := ValidateGeneratedImage(image, DefaultPolicy().Limits); err != nil {
			t.Fatalf("valid generated image %+v rejected: %v", image, err)
		}
	}
}

func imageString(value string) *string { return &value }
