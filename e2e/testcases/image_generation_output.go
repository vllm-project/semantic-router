package testcases

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	imageGenerationModel     = "mock/image-generation"
	imageGenerationTextModel = "openai/gpt-oss-20b"
	// imageGenerationProviderProtocol is the provider wire the image-capable
	// backend is reached on; the chat wire cannot express the operation.
	imageGenerationProviderProtocol = "openai.responses.v1"
)

func init() {
	pkgtestcases.Register("response-api-image-generation", pkgtestcases.TestCase{
		Description: "POST /v1/responses declaring the image_generation tool returns the generated image from a Responses-capable backend and fails closed on a chat-wire backend",
		Tags:        []string{"response-api", "functional", "image_generation"},
		Fn:          testResponseAPIImageGeneration,
	})
	pkgtestcases.Register("response-api-image-generation-stream", pkgtestcases.TestCase{
		Description: "POST /v1/responses stream:true declaring the image_generation tool returns the generation as Responses API SSE events ending in a completed image_generation_call item",
		Tags:        []string{"response-api", "streaming", "image_generation"},
		Fn:          testResponseAPIImageGenerationStreaming,
	})
}

// testResponseAPIImageGeneration pins the hosted image_generation contract end
// to end: a Responses request declaring the image_generation tool must reach a
// backend whose wire can express it, and must come back as an
// image_generation_call output item carrying the generated image in a decodable
// base64 result. Where the selected backend speaks the chat wire only, the same
// request must fail closed with unsupported_capability instead of degrading into
// a text call.
func testResponseAPIImageGeneration(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: image_generation tool contract")
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	sessionID := fmt.Sprintf("response-api-image-generation-%d", time.Now().UnixNano())
	if err := assertImageGenerationReturnsImage(ctx, session, sessionID); err != nil {
		return err
	}
	if err := verifyBackendReceivedImageGenerationTool(ctx, client, opts, sessionID); err != nil {
		return err
	}
	// The rejection request carries its own session marker, so the
	// absent-dispatch probe on that marker is a real discriminator instead of a
	// re-read of the positive request's record.
	if err := assertChatWiredBackendRejectsImageGeneration(ctx, client, opts, session, sessionID+"-chat-wire"); err != nil {
		return err
	}
	return nil
}

// assertImageGenerationReturnsImage sends the image-generation tool on the
// Responses wire and asserts the client-visible generated-image item.
func assertImageGenerationReturnsImage(ctx context.Context, session *fixtures.ServiceSession, sessionID string) error {
	response, err := postResponsesWithHeaders(ctx, session, map[string]any{
		"model": imageGenerationModel,
		"store": false,
		"input": []map[string]any{
			{"role": "user", "content": []map[string]any{
				{"type": "input_text", "text": "draw a red cat"},
			}},
		},
		"tools": []map[string]any{
			{"type": "image_generation", "size": "1024x1024", "quality": "high", "output_format": "png"},
		},
	}, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("image generation request returned HTTP %d, want 200: %s",
			response.StatusCode, truncateString(string(response.Body), 500))
	}
	var decoded struct {
		Status string `json:"status"`
		Output []struct {
			Type   string `json:"type"`
			Status string `json:"status"`
			Result string `json:"result"`
		} `json:"output"`
	}
	if decodeErr := json.Unmarshal(response.Body, &decoded); decodeErr != nil {
		return fmt.Errorf("decode image generation response: %w", decodeErr)
	}
	if decoded.Status != "completed" {
		return fmt.Errorf("image generation status = %q, want completed: %s",
			decoded.Status, truncateString(string(response.Body), 500))
	}
	if len(decoded.Output) != 1 {
		return fmt.Errorf("image generation returned %d output items, want 1: %s",
			len(decoded.Output), truncateString(string(response.Body), 500))
	}
	item := decoded.Output[0]
	if item.Type != "image_generation_call" {
		return fmt.Errorf("output item type = %q, want image_generation_call: %s",
			item.Type, truncateString(string(response.Body), 500))
	}
	if item.Status != "completed" {
		return fmt.Errorf("output item status = %q, want completed: %s",
			item.Status, truncateString(string(response.Body), 500))
	}
	if item.Result == "" {
		return fmt.Errorf("output item carries no generated image: %s", truncateString(string(response.Body), 500))
	}
	if err := decedeImageResult(item.Result); err != nil {
		return fmt.Errorf("generated image item result: %w: %s", err, truncateString(string(response.Body), 500))
	}
	return nil
}

// verifyBackendReceivedImageGenerationTool asserts the request the mock backend
// recorded carries the image_generation tool, so the client-visible image came
// from the backend instead of being synthesized on the way out.
func verifyBackendReceivedImageGenerationTool(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	sessionID string,
) error {
	providerSession, err := openProtocolCodecProviderSession(ctx, client, opts, imageGenerationProviderProtocol)
	if err != nil {
		return err
	}
	defer providerSession.Close()

	recorded, err := lastProviderSimulatorRequest(ctx, providerSession, sessionID)
	if err != nil {
		return err
	}
	var debug struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(recorded, &debug); err != nil {
		return fmt.Errorf("decode recorded backend request: %w", err)
	}
	var model string
	if err := json.Unmarshal(debug.Body["model"], &model); err != nil || model != imageGenerationModel {
		return fmt.Errorf("backend request model = %q, want %q: %s",
			model, imageGenerationModel, truncateString(string(recorded), 600))
	}
	var tools []struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(debug.Body["tools"], &tools); err != nil || len(tools) != 1 || tools[0].Type != "image_generation" {
		return fmt.Errorf("backend request lost the image_generation tool: %s", truncateString(string(recorded), 600))
	}
	return nil
}

// assertChatWiredBackendRejectsImageGeneration pins the capability contract of
// the pipeline: the same image-generation request naming a chat-wire backend
// must fail as a client error, and the backend must not have been dispatched to.
func assertChatWiredBackendRejectsImageGeneration(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	session *fixtures.ServiceSession,
	sessionID string,
) error {
	response, err := postResponsesWithHeaders(ctx, session, map[string]any{
		"model": imageGenerationTextModel,
		"store": false,
		"input": []map[string]any{
			{"role": "user", "content": []map[string]any{
				{"type": "input_text", "text": "draw a red cat"},
			}},
		},
		"tools": []map[string]any{
			{"type": "image_generation", "size": "1024x1024"},
		},
	}, map[string]string{"x-vsr-test-session-id": sessionID})
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("chat-wire image generation returned HTTP %d, want 400: %s",
			response.StatusCode, truncateString(string(response.Body), 500))
	}
	var envelope struct {
		Error struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if decodeErr := json.Unmarshal(response.Body, &envelope); decodeErr != nil {
		return fmt.Errorf("decode chat-wire capability error: %w", decodeErr)
	}
	if envelope.Error.Type != "invalid_request_error" ||
		envelope.Error.Code != "unsupported_capability" ||
		!strings.Contains(envelope.Error.Message, "openai.chat.v1") ||
		!strings.Contains(envelope.Error.Message, "image_generation") {
		return fmt.Errorf("chat-wire backend returned the wrong capability error: %s",
			truncateString(string(response.Body), 500))
	}
	providerSession, err := openProtocolCodecProviderSession(ctx, client, opts, "openai.chat.v1")
	if err != nil {
		return err
	}
	defer providerSession.Close()
	dispatched, model, err := lookupShortCircuitDispatch(ctx, providerSession, sessionID)
	if err != nil {
		return err
	}
	if dispatched {
		return fmt.Errorf("rejected chat-wire image generation request reached provider model %q", model)
	}
	return nil
}

// decedeImageResult decodes a generated image payload, tolerating the URL-safe
// alfabet and absent padding: the simulator emits URL-safe base64, and the
// pipeline may relay either form. A payload that does not decode after that
// normalization carries no image, so an alfabet-shaped string is not enough.
func decedeImageResult(result string) error {
	if result == "" {
		return fmt.Errorf("generated image payload is empty")
	}
	if _, err := base64.StdEncoding.DecodeString(result); err == nil {
		return nil
	}
	normalized := strings.ReplaceAll(strings.ReplaceAll(result, "-", "+"), "_", "/")
	for len(normalized)%4 != 0 {
		normalized += "="
	}
	if _, err := base64.StdEncoding.DecodeString(normalized); err != nil {
		return fmt.Errorf("generated image is not decodable base64: %w", err)
	}
	return nil
}

// testResponseAPIImageGenerationStreaming pins the streaming shape of the image
// generation contract: a streaming Responses request declaring the
// image_generation tool must carry the generation progress as Responses API SSE
// events and end in a completed image_generation_call item holding the
// generated image, without leaking upstream chat chunks into the stream.
func testResponseAPIImageGenerationStreaming(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API streaming: image_generation tool contract")
	}

	result, err := requestResponseAPIStreamingSSE(ctx, client, opts, imageGenerationModel,
		"response-api-image-generation-stream", "draw a red cat",
		[]map[string]any{
			{"type": "image_generation", "size": "1024x1024", "quality": "high", "output_format": "png"},
		})
	if err != nil {
		return err
	}
	return validateResponseAPIImageGenerationStream(result)
}

func validateResponseAPIImageGenerationStream(result responseAPIStreamingSSEResult) error {
	stream := string(result.body)

	if result.statusCode != http.StatusOK {
		return fmt.Errorf("streaming image generation returned HTTP %d, want 200: %s",
			result.statusCode, truncateString(stream, 500))
	}
	if !strings.Contains(result.contentType, "text/event-stream") {
		return fmt.Errorf("streaming image generation content-type = %q, want text/event-stream",
			result.contentType)
	}

	requiredEvents := []string{
		"event: response.output_item.added",
		"event: response.image_generation_call.generating",
		"event: response.image_generation_call.partial_image",
		"event: response.image_generation_call.completed",
		"event: response.output_item.done",
		"event: response.completed",
	}
	for _, event := range requiredEvents {
		if !strings.Contains(stream, event) {
			return fmt.Errorf("streaming image generation is missing Responses API SSE event %q: %s",
				event, truncateString(stream, 800))
		}
	}
	if err := validateResponsesStreamEventShapes(stream); err != nil {
		return err
	}
	for _, fragment := range []string{"chat.completion.chunk", "data: [DONE]"} {
		if strings.Contains(stream, fragment) {
			return fmt.Errorf("streaming image generation leaked upstream chat fragment %q: %s",
				fragment, truncateString(stream, 800))
		}
	}

	completed := 0
	for _, item := range imageGenerationStreamItems(stream) {
		if item.Type != "image_generation_call" || item.Status != "completed" {
			continue
		}
		completed++
		if item.Result == "" {
			return fmt.Errorf("streaming image generation completed item carries no image: %s",
				truncateString(stream, 800))
		}
		if err := decedeImageResult(item.Result); err != nil {
			return fmt.Errorf("streaming image generation result: %w: %s", err, truncateString(stream, 800))
		}
	}
	if completed == 0 {
		return fmt.Errorf("streaming image generation never completed an image_generation_call item: %s",
			truncateString(stream, 800))
	}
	return nil
}

type imageGenerationStreamItem struct {
	Type   string `json:"type"`
	Status string `json:"status"`
	Result string `json:"result"`
}

// imageGenerationStreamItems collects every item the stream carries, from both
// the item and the response frames.
func imageGenerationStreamItems(stream string) []imageGenerationStreamItem {
	items := []imageGenerationStreamItem{}
	for _, data := range protocolSSEDataFrames([]byte(stream)) {
		var frame struct {
			Item     *imageGenerationStreamItem `json:"item"`
			Response *struct {
				Output []imageGenerationStreamItem `json:"output"`
			} `json:"response"`
		}
		if err := json.Unmarshal([]byte(data), &frame); err != nil {
			continue
		}
		if frame.Item != nil {
			items = append(items, *frame.Item)
		}
		if frame.Response != nil {
			items = append(items, frame.Response.Output...)
		}
	}
	return items
}
