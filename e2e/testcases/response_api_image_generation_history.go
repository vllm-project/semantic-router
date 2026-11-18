package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("response-api-image-generation-history", pkgtestcases.TestCase{
		Description: "Stored buffered and streamed generated images retain their payload on GET and reach the provider intact on continuation",
		Tags:        []string{"response-api", "functional", "image_generation"},
		Fn:          testResponseAPIImageGenerationHistory,
	})
}

type retainedGeneratedImage struct {
	ID     string `json:"id"`
	Type   string `json:"type"`
	Status string `json:"status"`
	Result string `json:"result"`
}

type generatedImageHistoryResponse struct {
	ID     string                   `json:"id"`
	Status string                   `json:"status"`
	Output []retainedGeneratedImage `json:"output"`
}

func testResponseAPIImageGenerationHistory(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	router, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer router.Close()
	provider, err := openProtocolCodecProviderSession(ctx, client, opts, imageGenerationProviderProtocol)
	if err != nil {
		return err
	}
	defer provider.Close()
	api := fixtures.NewResponseAPIClient(router, 45*time.Second)
	for _, stream := range []bool{false, true} {
		created, err := sendProtocolMatrixRaw(ctx, router, "/v1/responses", map[string]any{
			"model": imageGenerationModel, "store": true, "stream": stream,
			"input": "Draw a red cat", "tools": []map[string]any{{"type": "image_generation"}},
		}, stream, nil)
		if err != nil {
			return err
		}
		if created.StatusCode != http.StatusOK {
			return fmt.Errorf("stored image stream=%t: HTTP %d: %s", stream, created.StatusCode, truncateString(string(created.Body), 500))
		}
		body := created.Body
		if stream {
			body = nil
			for _, data := range protocolSSEDataFrames(created.Body) {
				var frame struct {
					Type     string          `json:"type"`
					Response json.RawMessage `json:"response"`
				}
				if json.Unmarshal([]byte(data), &frame) == nil && frame.Type == "response.completed" {
					body = frame.Response
				}
			}
		}
		var original generatedImageHistoryResponse
		if decodeErr := json.Unmarshal(body, &original); decodeErr != nil {
			return fmt.Errorf("decode original image stream=%t: %w", stream, decodeErr)
		}
		if original.ID == "" || original.Status != "completed" || len(original.Output) != 1 {
			return fmt.Errorf("original image response is incomplete: %s", truncateString(string(body), 500))
		}
		defer func(id string) { _, _ = api.DeleteRaw(ctx, id) }(original.ID)
		image := original.Output[0]
		if image.ID == "" || image.Type != "image_generation_call" || image.Status != "completed" {
			return fmt.Errorf("original response has no completed image item: %+v", image)
		}
		if imageErr := decedeImageResult(image.Result); imageErr != nil {
			return imageErr
		}
		stored, err := api.GetRaw(ctx, original.ID)
		if err != nil {
			return err
		}
		var retrieved generatedImageHistoryResponse
		if stored.StatusCode != http.StatusOK {
			return fmt.Errorf("GET stored image: HTTP %d: %s", stored.StatusCode, truncateString(string(stored.Body), 500))
		}
		if decodeErr := json.Unmarshal(stored.Body, &retrieved); decodeErr != nil {
			return decodeErr
		}
		if retrieved.ID != original.ID || retrieved.Status != "completed" || len(retrieved.Output) != 1 || retrieved.Output[0] != image {
			return fmt.Errorf("GET changed the generated image stream=%t: %s", stream, truncateString(string(stored.Body), 500))
		}
		sessionID := "image-history-" + uuid.NewString()
		next, err := sendProtocolMatrixRaw(ctx, router, "/v1/responses", map[string]any{
			"model": imageGenerationModel, "store": false, "previous_response_id": original.ID,
			"input": "Make it blue", "tools": []map[string]any{{"type": "image_generation"}},
		}, false, map[string]string{"x-vsr-test-session-id": sessionID})
		if err != nil {
			return err
		}
		if next.StatusCode != http.StatusOK {
			return fmt.Errorf("image continuation stream=%t: HTTP %d: %s", stream, next.StatusCode, truncateString(string(next.Body), 500))
		}
		// A permissive mock returning 200 cannot prove that the stored image
		// survived replay. Inspect the actual provider-bound history as well.
		raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
		if err != nil {
			return err
		}
		var recorded struct {
			Body struct {
				Input []retainedGeneratedImage `json:"input"`
			} `json:"body"`
		}
		if decodeErr := json.Unmarshal(raw, &recorded); decodeErr != nil {
			return decodeErr
		}
		count := 0
		for _, item := range recorded.Body.Input {
			if item.Type == "image_generation_call" {
				count++
				if item != image {
					return fmt.Errorf("provider history changed the stored image stream=%t: %+v", stream, item)
				}
			}
		}
		if count != 1 {
			return fmt.Errorf("provider history contains %d generated images, want 1: %s", count, truncateString(string(raw), 600))
		}
	}
	return nil
}
