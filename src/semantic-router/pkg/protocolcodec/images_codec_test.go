package protocolcodec

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestImagesCodecEncodeRequestSinksResponsesImageGeneration(t *testing.T) {
	codec := ImagesCodec{}
	size := "1024x1024"
	request := llmprotocol.Request{
		Model: "openai/gpt-image",
		Messages: []llmprotocol.Message{
			{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "draw a cat"},
			}},
		},
		ImageGeneration: &llmprotocol.ImageGenerationOptions{Size: size},
	}
	body, _, err := codec.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.Policy{})
	if err != nil {
		t.Fatalf("EncodeRequest failed: %v", err)
	}
	var wire imagesRequestWire
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if wire.Prompt != "draw a cat" {
		t.Fatalf("prompt = %q, want %q", wire.Prompt, "draw a cat")
	}
	if wire.Size != size {
		t.Fatalf("size = %q, want %q", wire.Size, size)
	}
}

func TestImagesCodecEncodeRequestDefaultsSizeAndRejectsWithoutTool(t *testing.T) {
	codec := ImagesCodec{}
	request := llmprotocol.Request{
		Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{
			{Kind: llmprotocol.ContentText, Text: "a horse"},
		}}},
		ImageGeneration: &llmprotocol.ImageGenerationOptions{},
	}
	body, _, err := codec.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.Policy{})
	if err != nil {
		t.Fatalf("EncodeRequest failed: %v", err)
	}
	if !strings.Contains(string(body), `"size":"1024x1024"`) {
		t.Fatalf("expected default size in body, got %s", body)
	}

	request.ImageGeneration = nil
	if _, _, err := codec.EncodeRequest(request, llmprotocol.Envelope{}, llmprotocol.Policy{}); err == nil {
		t.Fatalf("EncodeRequest without image_generation must fail")
	}
}

func TestImagesCodecDecodeResponseToGeneratedImage(t *testing.T) {
	codec := ImagesCodec{}
	payload := "aGVsbG8gd29ybGQ=" // "hello world"
	body := []byte(`{"created":1701234567,"data":[{"b64_json":"` + payload + `"}]}`)
	response, _, _, err := codec.DecodeResponse(body, llmprotocol.Policy{})
	if err != nil {
		t.Fatalf("DecodeResponse failed: %v", err)
	}
	if len(response.Output) != 1 {
		t.Fatalf("output len = %d, want 1", len(response.Output))
	}
	item := response.Output[0]
	if item.Role != llmprotocol.RoleAssistant {
		t.Fatalf("role = %q, want assistant", item.Role)
	}
	if len(item.Content) != 1 {
		t.Fatalf("content len = %d, want 1", len(item.Content))
	}
	block := item.Content[0]
	if block.Kind != llmprotocol.ContentGeneratedImage || block.GeneratedImage == nil {
		t.Fatalf("block kind = %q, want generated image", block.Kind)
	}
	if block.GeneratedImage.Status != llmprotocol.ImageGenerationCompleted {
		t.Fatalf("status = %q, want completed", block.GeneratedImage.Status)
	}
	if block.GeneratedImage.Result == nil || *block.GeneratedImage.Result != payload {
		t.Fatalf("result = %v, want %q", block.GeneratedImage.Result, payload)
	}
}

func TestImagesCodecDecodeResponseEmptyData(t *testing.T) {
	codec := ImagesCodec{}
	if _, _, _, err := codec.DecodeResponse([]byte(`{"data":[]}`), llmprotocol.Policy{}); err == nil {
		t.Fatalf("empty data must fail")
	}
}

func TestImagesCodecEncodeResponseRoundTrip(t *testing.T) {
	codec := ImagesCodec{}
	payload := "cHNu" // "psn"
	response := llmprotocol.Response{Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind:           llmprotocol.ContentGeneratedImage,
		GeneratedImage: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationCompleted, Result: &payload},
	}}}}}
	body, _, err := codec.EncodeResponse(response, llmprotocol.Envelope{}, llmprotocol.Policy{})
	if err != nil {
		t.Fatalf("EncodeResponse failed: %v", err)
	}
	var wire imagesResponseWire
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(wire.Data) != 1 || wire.Data[0].B64JSON != payload {
		t.Fatalf("roundtrip data = %+v, want one %q", wire.Data, payload)
	}
}

func TestImagesCodecCapabilitiesEnableCapabilityDrivenSelection(t *testing.T) {
	codec := ImagesCodec{}
	set := codec.Capabilities()
	if !set.Supports(llmprotocol.CapabilityImageGeneration) {
		t.Fatalf("images codec must advertise image_generation")
	}
	if !set.Supports(llmprotocol.CapabilityTools) {
		t.Fatalf("images codec must advertise tools (hosted tool requests carry CapabilityTools)")
	}
	required := llmprotocol.RequiredCapabilities(llmprotocol.Request{
		ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration},
	})
	if !set.Contains(required) {
		t.Fatalf("images capabilities %v must contain required %v", set.Names(), required.Names())
	}
}

func TestRegistryCoordinatesImagesFormat(t *testing.T) {
	registry := NewBuiltinRegistry()
	set, ok := registry.CapabilitiesFor(llmprotocol.OpenAIImagesV1)
	if !ok {
		t.Fatalf("images format did not resolve")
	}
	if !set.Supports(llmprotocol.CapabilityImageGeneration) {
		t.Fatalf("images wire must advertise image_generation capability")
	}

	// A hosted image_generation request must be expressible by the images wire
	// (required set carries CapabilityTools, see requestTransportCapabilities).
	required := llmprotocol.RequiredCapabilities(llmprotocol.Request{
		Tools:      []llmprotocol.Tool{{Name: "image_generation"}},
		ToolChoice: llmprotocol.ToolChoice{Mode: llmprotocol.ToolChoiceImageGeneration},
	})
	if !set.Contains(required) {
		t.Fatalf("images wire capabilities %v must contain required %v", set.Names(), required.Names())
	}
}

// The engine must translate a DALL-E body all the way to the responses wire:
// decode via the images codec, then render image_generation_call for the
// client. Guards the semantic-generation and render path used by the Router's
// response pipe (generation_required regression).
func TestImagesEngineTranslatestoResponsesImageGenerationCall(t *testing.T) {
	engine, err := NewEngine(NewBuiltinRegistry(), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("NewEngine: %v", err)
	}
	body := []byte(`{"created":1701234567,"data":[{"b64_json":"aGVsbG8td29ybGQ="}]}`)
	result, err := engine.TranslateResponse(llmprotocol.OpenAIImagesV1, llmprotocol.OpenAIResponsesV1, body, nil)
	if err != nil {
		t.Fatalf("TranslateResponse: %v", err)
	}
	if len(result.Response.Output) != 1 {
		t.Fatalf("output len = %d, want 1", len(result.Response.Output))
	}
	block := result.Response.Output[0].Content[0]
	if block.Kind != llmprotocol.ContentGeneratedImage || block.GeneratedImage == nil || block.GeneratedImage.Result == nil {
		t.Fatalf("expected generated image with result, got %+v", block)
	}
	if !bytes.Contains(result.Body, []byte(`"type":"image_generation_call"`)) {
		t.Fatalf("client body must render image_generation_call, got: %.400s", result.Body)
	}
}
