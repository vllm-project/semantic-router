package protocolcodec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// ImagesCodec is the DALL-E-compatible image-generation dialect
// (/v1/images/generations) exposed by diffusion backends such as vLLM-Omni.
// It is a sink dialect: it does not decode arbitrary chat traffic nor stream,
// its only purpose is to express the hosted image_generation operation and to
// decode the generated image back into the neutral GeneratedImage output item
// so upstream images and client responses formats share one representation.
type ImagesCodec struct{}

func (ImagesCodec) Format() llmprotocol.WireFormat { return llmprotocol.OpenAIImagesV1 }
func (ImagesCodec) Stateless() bool                { return true }
func (ImagesCodec) Capabilities() llmprotocol.CapabilitySet {
	return llmprotocol.Capabilities(
		llmprotocol.CapabilityText,
		llmprotocol.CapabilityImageGeneration,
		// The hosted image_generation tool is this wire's native operation;
		// tool-bearing requests carry CapabilityTools in the required set
		// (llmprotocol.requestTransportCapabilities), so the dialect must
		// advertise it to be selected for such requests by capability-driven
		// rerouting; mixed-tools requests are rejected in EncodeRequest.
		llmprotocol.CapabilityTools,
		llmprotocol.CapabilityMultipleCandidates,
	)
}

const (
	imagesDefaultSize       = "1024x1024"
	imagesResponseFormatB64 = "b64_json"
)

type imagesRequestWire struct {
	Model             string `json:"model,omitempty"`
	Prompt            string `json:"prompt"`
	Size              string `json:"size,omitempty"`
	N                 *int   `json:"n,omitempty"`
	Quality           string `json:"quality,omitempty"`
	ResponseFormat    string `json:"response_format,omitempty"`
	Background        string `json:"background,omitempty"`
	Moderation        string `json:"moderation,omitempty"`
	OutputFormat      string `json:"output_format,omitempty"`
	OutputCompression *int64 `json:"output_compression,omitempty"`
}

type imagesResponseWire struct {
	Created int64            `json:"created,omitempty"`
	Data    []imagesDataWire `json:"data"`
	Error   *imagesErrorWire `json:"error,omitempty"`
}

type imagesDataWire struct {
	B64JSON string  `json:"b64_json"`
	URL     *string `json:"url"`
}

type imagesErrorWire struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// DecodeRequest decodes a direct images-dialect client request into the
// neutral request IR. A non-empty prompt becomes a user text message, and
// every images request field maps onto the neutral image-generation options
// so the conversion is lossless.
func (ImagesCodec) DecodeRequest(body []byte, _ llmprotocol.Policy) (llmprotocol.Request, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire imagesRequestWire
	if err := json.Unmarshal(body, &wire); err != nil {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, invalidRequestTranslation(err)
	}
	if wire.Prompt == "" {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, invalidRequestTranslation(fmt.Errorf("images prompt is required"))
	}
	if wire.ResponseFormat != "" && wire.ResponseFormat != imagesResponseFormatB64 {
		return llmprotocol.Request{}, llmprotocol.Envelope{}, nil, unsupportedDownstreamTranslation(fmt.Errorf("images wire only supports response_format %q", imagesResponseFormatB64))
	}
	content := llmprotocol.Content{Kind: llmprotocol.ContentText, Text: wire.Prompt}
	request := llmprotocol.Request{
		Model: wire.Model,
		Messages: []llmprotocol.Message{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{content},
		}},
		ImageGeneration: &llmprotocol.ImageGenerationOptions{
			Model:             wire.Model,
			Quality:           wire.Quality,
			Size:              wire.Size,
			OutputFormat:      wire.OutputFormat,
			OutputCompression: wire.OutputCompression,
			Moderation:        wire.Moderation,
			Background:        wire.Background,
		},
	}
	if request.ImageGeneration.Size == "" {
		request.ImageGeneration.Size = imagesDefaultSize
	}
	if wire.N != nil {
		request.CandidateCount = llmprotocol.Int64(int64(*wire.N))
	}
	return request, llmprotocol.Envelope{}, nil, nil
}

// EncodeRequest renders a neutral request as a DALL-E images request. The
// prompt is taken from the last user text block; size comes from the hosted
// image-generation options with a 1024x1024 default. Every other neutral
// option that the images dialect cannot express fails closed instead of being
// silently dropped, and the provider model plus b64_json response format are
// always sent so the upstream reply matches what DecodeResponse accepts.
func (ImagesCodec) EncodeRequest(request llmprotocol.Request, _ llmprotocol.Envelope, _ llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	if request.ImageGeneration == nil {
		return nil, nil, unsupportedDownstreamTranslation(fmt.Errorf("images wire requires a hosted image_generation request"))
	}
	if err := rejectUnsupportedImageOptions(*request.ImageGeneration); err != nil {
		return nil, nil, err
	}
	if err := rejectUnsupportedTools(request); err != nil {
		return nil, nil, err
	}
	prompt, ok := lastUserRequestText(request)
	if !ok {
		return nil, nil, unsupportedDownstreamTranslation(fmt.Errorf("images prompt is unavailable"))
	}
	size := request.ImageGeneration.Size
	if size == "" {
		size = imagesDefaultSize
	}
	wire := imagesRequestWire{
		Model:             request.Model,
		Prompt:            prompt,
		Size:              size,
		Quality:           request.ImageGeneration.Quality,
		ResponseFormat:    imagesResponseFormatB64,
		Background:        request.ImageGeneration.Background,
		Moderation:        request.ImageGeneration.Moderation,
		OutputFormat:      request.ImageGeneration.OutputFormat,
		OutputCompression: request.ImageGeneration.OutputCompression,
	}
	if n := request.CandidateCount; n != nil && *n > 0 {
		value := int(*n)
		wire.N = &value
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, nil, fmt.Errorf("encode images request: %w", err)
	}
	return body, nil, nil
}

// rejectUnsupportedTools fails closed on tool-bearing requests the images
// dialect cannot express: any ordinary function tool, or a tool choice that
// asks for something other than the hosted image_generation operation. The
// codec still advertises CapabilityTools so capability-driven rerouting can
// select it for pure image-generation requests (whose required set carries
// that capability), but mixed-tools requests are rejected here instead of
// being silently rewritten.
func rejectUnsupportedTools(request llmprotocol.Request) error {
	if len(request.Tools) > 0 {
		return unsupportedDownstreamTranslation(fmt.Errorf("images wire does not support function tools"))
	}
	switch request.ToolChoice.Mode {
	case "", llmprotocol.ToolChoiceNone, llmprotocol.ToolChoiceImageGeneration:
		return nil
	default:
		return unsupportedDownstreamTranslation(fmt.Errorf("images wire does not support tool choice %q", request.ToolChoice.Mode))
	}
}

// rejectUnsupportedImageOptions fails closed on neutral image-generation
// options that belong to the Responses tool contract but have no images
// request field, so supported conversions stay lossless instead of silently
// dropping the caller's intent.
func rejectUnsupportedImageOptions(options llmprotocol.ImageGenerationOptions) error {
	if options.InputFidelity != "" {
		return unsupportedDownstreamTranslation(fmt.Errorf("images wire does not support image option \"input_fidelity\""))
	}
	if options.Action != "" {
		return unsupportedDownstreamTranslation(fmt.Errorf("images wire does not support image option \"action\""))
	}
	if options.PartialImages != nil {
		return unsupportedDownstreamTranslation(fmt.Errorf("images wire does not support image option \"partial_images\""))
	}
	if options.InputImageMask != nil {
		return unsupportedDownstreamTranslation(fmt.Errorf("images wire does not support image option \"input_image_mask\""))
	}
	return nil
}

// DecodeResponse converts a DALL-E response ({data:[{b64_json}]}) into the
// neutral response IR as completed GeneratedImage output items.
func (ImagesCodec) DecodeResponse(body []byte, _ llmprotocol.Policy) (llmprotocol.Response, llmprotocol.Envelope, llmprotocol.Diagnostics, error) {
	var wire imagesResponseWire
	if err := json.Unmarshal(body, &wire); err != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, invalidUpstreamResponse(err)
	}
	if wire.Error != nil {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, fmt.Errorf("images upstream error: %s", wire.Error.Message)
	}
	if len(wire.Data) == 0 {
		return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, invalidUpstreamResponse(fmt.Errorf("images response contains no data"))
	}
	output := make([]llmprotocol.OutputItem, 0, len(wire.Data))
	for index := range wire.Data {
		result := wire.Data[index].B64JSON
		if result == "" {
			return llmprotocol.Response{}, llmprotocol.Envelope{}, nil, invalidUpstreamResponse(fmt.Errorf("images response item %d has no b64 payload", index))
		}
		output = append(output, llmprotocol.OutputItem{
			ID:   llmprotocol.StableID("images-output", fmt.Sprint(wire.Created), fmt.Sprint(index)),
			Role: llmprotocol.RoleAssistant,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentGeneratedImage,
				GeneratedImage: &llmprotocol.GeneratedImage{
					Status: llmprotocol.ImageGenerationCompleted,
					Result: &result,
				},
			}},
		})
	}
	response := llmprotocol.Response{
		Generation: 1,
		ID:         llmprotocol.StableID("images-response", fmt.Sprint(wire.Created)),
		CreatedAt:  time.Unix(wire.Created, 0),
		Output:     output,
		Usage:      llmprotocol.Usage{State: llmprotocol.UsageUnavailable},
	}
	return response, llmprotocol.Envelope{}, nil, nil
}

// EncodeResponse renders a neutral response back to the images dialect,
// serving clients that call /v1/images/generations through the Router.
func (ImagesCodec) EncodeResponse(response llmprotocol.Response, _ llmprotocol.Envelope, _ llmprotocol.Policy) ([]byte, llmprotocol.Diagnostics, error) {
	data := make([]imagesDataWire, 0, len(response.Output))
	for index := range response.Output {
		for _, block := range response.Output[index].Content {
			if block.Kind != llmprotocol.ContentGeneratedImage || block.GeneratedImage == nil {
				continue
			}
			if block.GeneratedImage.Result == nil {
				data = append(data, imagesDataWire{})
				continue
			}
			data = append(data, imagesDataWire{B64JSON: *block.GeneratedImage.Result})
		}
	}
	if len(data) == 0 {
		return nil, nil, unsupportedDownstreamTranslation(fmt.Errorf("images response contains no generated image"))
	}
	body, err := json.Marshal(imagesResponseWire{Created: time.Now().Unix(), Data: data})
	if err != nil {
		return nil, nil, fmt.Errorf("encode images response: %w", err)
	}
	return body, nil, nil
}

func (ImagesCodec) DecodeTransportError(body []byte, _ llmprotocol.Policy) (llmprotocol.TransportError, llmprotocol.Diagnostics, error) {
	var wire struct {
		Error *imagesErrorWire `json:"error"`
	}
	if len(bytes.TrimSpace(body)) > 0 {
		if err := json.Unmarshal(body, &wire); err != nil {
			return llmprotocol.TransportError{}, nil, invalidUpstreamResponse(err)
		}
	}
	message := "image service returned an error"
	code := "invalid_upstream_error"
	if wire.Error != nil {
		if wire.Error.Message != "" {
			message = wire.Error.Message
		}
		if wire.Error.Code != "" {
			code = wire.Error.Code
		}
	}
	return llmprotocol.TransportError{
		Error: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, code, message, nil),
	}, nil, nil
}

func (ImagesCodec) EncodeTransportError(err llmprotocol.TransportError) []byte {
	message := "model service returned an invalid error response"
	code := "invalid_upstream_error"
	if err.Error != nil {
		if err.Error.Message != "" {
			message = err.Error.Message
		}
		if err.Error.Code != "" {
			code = err.Error.Code
		}
	}
	body, marshalErr := json.Marshal(map[string]any{
		"error": map[string]any{"message": message, "type": "invalid_request_error", "code": code},
	})
	if marshalErr != nil {
		return []byte(`{"error":{"message":"model service returned an invalid error response","type":"invalid_request_error","code":"invalid_upstream_error"}}`)
	}
	return body
}

// The images dialect never streams; the stubs exist to satisfy the codec
// registry's streaming requirement and fail loudly if ever invoked.
func (ImagesCodec) NewDecoder(_ llmprotocol.StreamContext, _ llmprotocol.Policy) llmprotocol.StreamDecoder {
	return imagesStreamUnsupportedDecoder{}
}

func (ImagesCodec) NewEncoder(_ llmprotocol.StreamContext, _ llmprotocol.Policy) llmprotocol.StreamEncoder {
	return imagesStreamUnsupportedEncoder{}
}

type imagesStreamUnsupportedDecoder struct{}
type imagesStreamUnsupportedEncoder struct{}

func (imagesStreamUnsupportedDecoder) Push([]byte) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("images wire does not support streaming")
}
func (imagesStreamUnsupportedDecoder) Finalize(error) ([]llmprotocol.Event, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("images wire does not support streaming")
}

func (imagesStreamUnsupportedEncoder) Push(llmprotocol.Event) ([][]byte, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("images wire does not support streaming")
}
func (imagesStreamUnsupportedEncoder) Finalize(error) ([][]byte, llmprotocol.Diagnostics, error) {
	return nil, nil, fmt.Errorf("images wire does not support streaming")
}

// lastUserRequestText returns the last non-empty text block of the last user
// message, which is the prompt to render for image generation. Content in
// assistant or tool roles never becomes the prompt, so multi-message
// conversations cannot steer the image request.
func lastUserRequestText(request llmprotocol.Request) (string, bool) {
	for index := len(request.Messages) - 1; index >= 0; index-- {
		message := request.Messages[index]
		if message.Role != llmprotocol.RoleUser {
			continue
		}
		for block := len(message.Content) - 1; block >= 0; block-- {
			content := message.Content[block]
			if content.Kind == llmprotocol.ContentText && content.Text != "" {
				return content.Text, true
			}
		}
		return "", false
	}
	return "", false
}

func invalidRequestTranslation(cause error) error {
	return llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "invalid_images_request", cause.Error(), cause)
}

func invalidUpstreamResponse(cause error) error {
	return llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "invalid_upstream_json", cause.Error(), cause)
}

func unsupportedDownstreamTranslation(cause error) error {
	return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_downstream_translation", cause.Error(), cause)
}
