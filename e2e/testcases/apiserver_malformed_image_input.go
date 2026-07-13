package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const malformedImageE2EPayload = "private-e2e-image-payload!!!!"

var malformedImageE2ECases = []struct {
	name    string
	mime    string
	payload string
}{
	{name: "invalid-base64", mime: "image/png", payload: malformedImageE2EPayload},
	{name: "valid-base64-non-image", mime: "image/png", payload: "cHJpdmF0ZS1lMmUtaW1hZ2UtYnl0ZXM="},
	{name: "unsupported-gif", mime: "image/gif", payload: "R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="},
	{name: "unsupported-webp", mime: "image/webp", payload: "UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAEAAUAmJaQAA3AA/v89WAAAAA=="},
}

func init() {
	pkgtestcases.Register("apiserver-malformed-image-input", pkgtestcases.TestCase{
		Description: "Reject malformed or unsupported inline image data consistently across the live classify, eval, and embeddings APIs",
		Tags:        []string{"kubernetes", "apiserver", "classification", "embedding", "multimodal", "security"},
		Fn:          testAPIServerMalformedImageInput,
	})
}

type malformedImageAPIError struct {
	Error struct {
		Code string `json:"code"`
	} `json:"error"`
}

func testAPIServerMalformedImageInput(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	httpClient := session.HTTPClient(30 * time.Second)
	checked := make([]string, 0, len(malformedImageE2ECases)*3)
	for _, imageCase := range malformedImageE2ECases {
		imageURL := "data:" + imageCase.mime + ";base64," + imageCase.payload
		messagePayload := map[string]interface{}{
			"messages": []map[string]interface{}{
				{
					"role": "user",
					"content": []map[string]interface{}{
						{"type": "text", "text": "describe this image"},
						{"type": "image_url", "image_url": map[string]string{"url": imageURL}},
					},
				},
			},
		}
		requests := []struct {
			path    string
			payload interface{}
		}{
			{path: "/api/v1/classify/intent", payload: messagePayload},
			{path: "/api/v1/eval", payload: messagePayload},
			{path: "/api/v1/embeddings", payload: map[string]interface{}{"images": []string{imageURL}}},
		}

		for _, request := range requests {
			body, err := json.Marshal(request.payload)
			if err != nil {
				return fmt.Errorf("marshal %s image request for %s: %w", imageCase.name, request.path, err)
			}
			resp, err := postJSON(ctx, httpClient, http.MethodPost, session.URL(request.path), body)
			if err != nil {
				return err
			}
			if err := validateMalformedImageRejection(request.path, resp, imageCase.payload); err != nil {
				return fmt.Errorf("%s: %w", imageCase.name, err)
			}
			checked = append(checked, imageCase.name+":"+request.path)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"validated_endpoints": checked})
	}
	return nil
}

func validateMalformedImageRejection(path string, resp *httpResponse, forbiddenPayload string) error {
	if resp.StatusCode != http.StatusBadRequest {
		return fmt.Errorf("expected %s malformed image status 400, got %d: %s", path, resp.StatusCode, string(resp.Body))
	}
	var document malformedImageAPIError
	if err := json.Unmarshal(resp.Body, &document); err != nil {
		return fmt.Errorf("decode %s malformed-image response: %w", path, err)
	}
	if document.Error.Code != "INVALID_IMAGE" {
		return fmt.Errorf("expected %s error code INVALID_IMAGE, got %q: %s", path, document.Error.Code, string(resp.Body))
	}
	if forbiddenPayload != "" && strings.Contains(string(resp.Body), forbiddenPayload) {
		return fmt.Errorf("%s reflected the rejected image payload", path)
	}
	return nil
}
