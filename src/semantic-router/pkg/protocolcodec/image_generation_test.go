package protocolcodec

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesImageGenerationRequestRoundTripPreservesEveryOfficialField(t *testing.T) {
	body := []byte(`{
		"model":"image-model","input":"draw a red fox","stream":true,
		"tools":[{"type":"image_generation","model":"gpt-image-1","quality":"high","size":"1536x1024",
			"output_format":"webp","output_compression":72,"moderation":"low","background":"transparent",
			"input_fidelity":"high","input_image_mask":{"image_url":"data:image/png;base64,bWFzaw==","file_id":"file_mask"},
			"partial_images":3,"action":"edit"}],
		"tool_choice":{"type":"image_generation"}
	}`)
	engine := NewBuiltinEngine()
	request, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	wantCompression, wantPartials := int64(72), int64(3)
	want := &llmprotocol.ImageGenerationOptions{
		Model: "gpt-image-1", Quality: "high", Size: "1536x1024",
		OutputFormat: "webp", OutputCompression: &wantCompression,
		Moderation: "low", Background: "transparent", InputFidelity: "high",
		InputImageMask: &llmprotocol.ImageGenerationMask{
			EncodedImage: "data:image/png;base64,bWFzaw==", FileID: "file_mask",
		},
		PartialImages: &wantPartials, Action: "edit",
	}
	if !reflect.DeepEqual(request.ImageGeneration, want) {
		t.Fatalf("image-generation options = %+v, want %+v", request.ImageGeneration, want)
	}
	if request.ToolChoice.Mode != llmprotocol.ToolChoiceImageGeneration {
		t.Fatalf("tool choice = %+v", request.ToolChoice)
	}

	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	roundTrip, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, encoded.Body)
	if err != nil {
		t.Fatalf("encoded request does not satisfy its own contract: %v\n%s", err, encoded.Body)
	}
	if !reflect.DeepEqual(roundTrip.ImageGeneration, want) || roundTrip.ToolChoice.Mode != llmprotocol.ToolChoiceImageGeneration {
		t.Fatalf("round trip = %+v, tool choice=%+v", roundTrip.ImageGeneration, roundTrip.ToolChoice)
	}
}

func TestResponsesGeneratedImagePreservesNullAndEmptyResults(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, test := range []struct {
		name       string
		resultJSON string
		wantNil    bool
		want       string
	}{
		{name: "null", resultJSON: "null", wantNil: true},
		{name: "empty string", resultJSON: `""`, want: ""},
		{name: "base64", resultJSON: `"ZmluYWw="`, want: "ZmluYWw="},
	} {
		t.Run(test.name, func(t *testing.T) {
			assertGeneratedImageResultRoundTrip(t, engine, test.resultJSON, test.wantNil, test.want)
		})
	}
}

func assertGeneratedImageResultRoundTrip(
	t *testing.T,
	engine *Engine,
	resultJSON string,
	wantNil bool,
	want string,
) {
	t.Helper()
	body := []byte(`{"id":"resp_image","object":"response","model":"image-model","status":"completed","output":[` +
		`{"type":"image_generation_call","id":"ig_1","status":"completed","result":` + resultJSON + `}]}`)
	response, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	assertDecodedGeneratedImageResult(t, response, wantNil, want)
	assertEncodedGeneratedImageResult(t, engine, response, resultJSON)
}

func assertDecodedGeneratedImageResult(t *testing.T, response llmprotocol.Response, wantNil bool, want string) {
	t.Helper()
	image := response.Output[0].Content[0].GeneratedImage
	if image == nil || (image.Result == nil) != wantNil {
		t.Fatalf("decoded image = %+v, want nil result=%v", image, wantNil)
	}
	if image.Result != nil && *image.Result != want {
		t.Fatalf("decoded result = %q, want %q", *image.Result, want)
	}
}

func assertEncodedGeneratedImageResult(t *testing.T, engine *Engine, response llmprotocol.Response, want string) {
	t.Helper()
	encoded, err := engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, response, llmprotocol.Envelope{})
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Output []map[string]json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(encoded.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire.Output) != 1 || string(wire.Output[0]["result"]) != want {
		t.Fatalf("encoded result = %s, want %s\n%s", wire.Output[0]["result"], want, encoded.Body)
	}
}

func TestResponsesGeneratedImageRejectsMalformedProviderItems(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, test := range []struct {
		name string
		item string
		code string
	}{
		{name: "missing result", item: `{"type":"image_generation_call","id":"ig_1","status":"completed"}`, code: "invalid_response_item"},
		{name: "missing id", item: `{"type":"image_generation_call","status":"completed","result":null}`, code: "invalid_response_item"},
		{name: "unknown status", item: `{"type":"image_generation_call","id":"ig_1","status":"queued","result":null}`, code: "invalid_responses_item_status"},
		{name: "nonterminal status", item: `{"type":"image_generation_call","id":"ig_1","status":"generating","result":null}`, code: "nonterminal_image_generation_output"},
		{name: "foreign field", item: `{"type":"image_generation_call","id":"ig_1","status":"completed","result":null,"arguments":"{}"}`, code: "invalid_response_item"},
		{name: "invalid base64", item: `{"type":"image_generation_call","id":"ig_1","status":"completed","result":"not-base64"}`, code: "invalid_generated_image_data"},
	} {
		t.Run(test.name, func(t *testing.T) {
			body := []byte(`{"id":"resp_image","object":"response","model":"image-model","status":"completed","output":[` + test.item + `]}`)
			_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

func TestImageGenerationStreamLifecycleRejectsInvalidTransitions(t *testing.T) {
	completedResult := "ZmluYWw="
	partialZero, partialTwo := int64(0), int64(2)
	for _, test := range []struct {
		name      string
		setup     []*llmprotocol.GeneratedImage
		candidate *llmprotocol.GeneratedImage
		code      string
	}{
		{name: "progress result", candidate: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationGenerating, Result: &completedResult}, code: "stream_image_generation_result"},
		{name: "failed progress", candidate: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationFailed}, code: "stream_image_generation_status"},
		{name: "backwards status", setup: []*llmprotocol.GeneratedImage{{Status: llmprotocol.ImageGenerationGenerating}}, candidate: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationInProgress}, code: "stream_image_generation_order"},
		{name: "duplicate in progress", setup: []*llmprotocol.GeneratedImage{{Status: llmprotocol.ImageGenerationInProgress}}, candidate: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationInProgress}, code: "duplicate_stream_image_generation_event"},
		{name: "duplicate generating", setup: []*llmprotocol.GeneratedImage{{Status: llmprotocol.ImageGenerationGenerating}}, candidate: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationGenerating}, code: "duplicate_stream_image_generation_event"},
		{name: "partial gap", setup: []*llmprotocol.GeneratedImage{{Status: llmprotocol.ImageGenerationGenerating, PartialIndex: &partialZero, PartialImage: "YQ=="}}, candidate: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationGenerating, PartialIndex: &partialTwo, PartialImage: "Yg=="}, code: "stream_partial_image_index_order"},
		{name: "partial wrong status", candidate: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationInProgress, PartialIndex: &partialZero, PartialImage: "YQ=="}, code: "stream_partial_image_status"},
		{name: "invalid partial data", candidate: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationGenerating, PartialIndex: &partialZero, PartialImage: "not-base64"}, code: "invalid_generated_image_data"},
	} {
		t.Run(test.name, func(t *testing.T) {
			encoder := newStartedImageGenerationEncoder(t)
			for _, image := range test.setup {
				pushImageGenerationProgress(t, encoder, image)
			}
			_, _, err := encoder.Push(imageProgressEvent(test.candidate))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

func TestImageGenerationStreamLifecycleRejectsInvalidStartAndCompletion(t *testing.T) {
	engine := NewBuiltinEngine()
	encoder, err := engine.NewEventStreamEncoder(llmprotocol.OpenAIResponsesV1, imageStreamContext())
	if err != nil {
		t.Fatal(err)
	}
	pushImageStreamEvent(t, encoder, imageResponseStartEvent())
	_, _, err = encoder.Push(imageItemStartEvent(&llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationCompleted}))
	assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, "stream_image_generation_start")

	for _, test := range []struct {
		name      string
		progress  *llmprotocol.GeneratedImage
		completed *llmprotocol.GeneratedImage
		code      string
	}{
		{name: "nonterminal completion", completed: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationGenerating}, code: "stream_image_generation_completion"},
		{name: "completed progress then failed item", progress: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationCompleted}, completed: &llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationFailed}, code: "stream_image_generation_completion_mismatch"},
	} {
		t.Run(test.name, func(t *testing.T) {
			encoder := newStartedImageGenerationEncoder(t)
			if test.progress != nil {
				pushImageGenerationProgress(t, encoder, test.progress)
			}
			_, _, err := encoder.Push(imageItemCompletionEvent(test.completed))
			assertProtocolError(t, err, llmprotocol.ErrorUpstreamUnavailable, test.code)
		})
	}
}

func TestImageGenerationStreamLifecycleAcceptsCompleteOrderedSequence(t *testing.T) {
	encoder := newStartedImageGenerationEncoder(t)
	zero, one := int64(0), int64(1)
	for _, image := range []*llmprotocol.GeneratedImage{
		{Status: llmprotocol.ImageGenerationInProgress},
		{Status: llmprotocol.ImageGenerationGenerating},
		{Status: llmprotocol.ImageGenerationGenerating, PartialIndex: &zero, PartialImage: "YQ=="},
		{Status: llmprotocol.ImageGenerationGenerating, PartialIndex: &one, PartialImage: "Yg=="},
		{Status: llmprotocol.ImageGenerationCompleted},
	} {
		pushImageGenerationProgress(t, encoder, image)
	}
	result := "ZmluYWw="
	pushImageStreamEvent(t, encoder, imageItemCompletionEvent(&llmprotocol.GeneratedImage{
		Status: llmprotocol.ImageGenerationCompleted, Result: &result,
	}))
}

func TestDecodeResponsesImageGenerationStreamAccumulatesTerminalImage(t *testing.T) {
	response := decodeImageGenerationStreamFixture(t)
	assertCompletedGeneratedImageResponse(t, response)
}

func decodeImageGenerationStreamFixture(t *testing.T) llmprotocol.Response {
	t.Helper()
	path := filepath.Join("testdata", "golden", "capability", "049-responses-image-generation-stream-in.json")
	body, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var input goldenCapabilityInput
	if err := json.Unmarshal(body, &input); err != nil {
		t.Fatal(err)
	}
	if input.Stream == nil {
		t.Fatal("image-generation stream fixture is missing its stream payload")
	}
	response, _, err := NewBuiltinEngine().DecodeResponseStream(
		llmprotocol.OpenAIResponsesV1,
		[]byte(strings.Join(input.Stream.Chunks, "")),
		llmprotocol.StreamContext{
			Context: context.Background(), PublicModel: input.Stream.PublicModel,
			ProviderModel: input.Stream.ProviderModel,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	return response
}

func assertCompletedGeneratedImageResponse(t *testing.T, response llmprotocol.Response) {
	t.Helper()
	if len(response.Output) != 1 || len(response.Output[0].Content) != 1 {
		t.Fatalf("accumulated output = %+v", response.Output)
	}
	image := response.Output[0].Content[0].GeneratedImage
	if image == nil || image.Status != llmprotocol.ImageGenerationCompleted || image.Result == nil || *image.Result != "ZmluYWwtaW1hZ2U=" {
		t.Fatalf("accumulated image = %+v", image)
	}
	if image.PartialIndex != nil || image.PartialImage != "" {
		t.Fatalf("partial progress leaked into terminal image: %+v", image)
	}
}

func newStartedImageGenerationEncoder(t *testing.T) *EventStreamEncoder {
	t.Helper()
	encoder, err := NewBuiltinEngine().NewEventStreamEncoder(llmprotocol.OpenAIResponsesV1, imageStreamContext())
	if err != nil {
		t.Fatal(err)
	}
	pushImageStreamEvent(t, encoder, imageResponseStartEvent())
	pushImageStreamEvent(t, encoder, imageItemStartEvent(&llmprotocol.GeneratedImage{Status: llmprotocol.ImageGenerationInProgress}))
	return encoder
}

func imageStreamContext() llmprotocol.StreamContext {
	return llmprotocol.StreamContext{Context: context.Background(), PublicModel: "public-model", ProviderModel: "image-model"}
}

func imageResponseStartEvent() llmprotocol.Event {
	return llmprotocol.Event{Type: llmprotocol.EventResponseStarted, ResponseID: "resp_image", Model: "image-model"}
}

func imageItemStartEvent(image *llmprotocol.GeneratedImage) llmprotocol.Event {
	return llmprotocol.Event{
		Type: llmprotocol.EventOutputItemStarted, ResponseID: "resp_image", ItemIndex: 0, ItemID: "ig_1",
		Content: &llmprotocol.Content{Kind: llmprotocol.ContentGeneratedImage, GeneratedImage: image},
	}
}

func imageProgressEvent(image *llmprotocol.GeneratedImage) llmprotocol.Event {
	return llmprotocol.Event{
		Type: llmprotocol.EventImageGenerationProgress, ResponseID: "resp_image", ItemIndex: 0, ItemID: "ig_1",
		GeneratedImage: image,
	}
}

func imageItemCompletionEvent(image *llmprotocol.GeneratedImage) llmprotocol.Event {
	return llmprotocol.Event{
		Type: llmprotocol.EventOutputItemCompleted, ResponseID: "resp_image", ItemIndex: 0, ItemID: "ig_1",
		Content: &llmprotocol.Content{Kind: llmprotocol.ContentGeneratedImage, GeneratedImage: image},
	}
}

func pushImageGenerationProgress(t *testing.T, encoder *EventStreamEncoder, image *llmprotocol.GeneratedImage) {
	t.Helper()
	pushImageStreamEvent(t, encoder, imageProgressEvent(image))
}

func pushImageStreamEvent(t *testing.T, encoder *EventStreamEncoder, event llmprotocol.Event) {
	t.Helper()
	if _, _, err := encoder.Push(event); err != nil {
		t.Fatalf("push %s: %v", event.Type, err)
	}
}
