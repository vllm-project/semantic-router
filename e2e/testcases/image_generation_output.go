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
	if err := assertChatWiredBackendRejectsImageGeneration(ctx, session); err != nil {
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
	if _, decodeErr := base64.StdEncoding.DecodeString(item.Result); decodeErr != nil {
		return fmt.Errorf("generated image is not decodable base64: %w", decodeErr)
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
func assertChatWiredBackendRejectsImageGeneration(ctx context.Context, session *fixtures.ServiceSession) error {
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
	}, nil)
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
	if envelope.Error.Code != "unsupported_capability" ||
		!strings.Contains(envelope.Error.Message, imageGenerationTextModel) ||
		!strings.Contains(envelope.Error.Message, "image_generation") {
		return fmt.Errorf("chat-wire backend returned the wrong capability error: %s",
			truncateString(string(response.Body), 500))
	}
	return nil
}
