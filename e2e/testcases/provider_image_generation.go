package testcases

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image/png"
	"net/http"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("provider-native-image-generation", pkgtestcases.TestCase{
		Description: "Verify the deployed provider fixture returns valid native image payloads and rejects unknown fields",
		Tags:        []string{"provider", "image-generation"},
		Fn:          testProviderNativeImageGeneration,
	})
}

// The image endpoint is a native provider fixture boundary. Image generation is
// tested directly; it does not pretend to exercise a Router image-generation API.
func testProviderNativeImageGeneration(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceEndpointSession(ctx, client, opts, "default", "provider-mocker", "8000")
	if err != nil {
		return err
	}
	defer session.Close()

	const prompt = "provider image contract marker"
	request := map[string]any{"model": "image-fixture", "prompt": prompt, "n": 2, "response_format": "b64_json"}
	response, err := fixtures.DoPOSTRequest(ctx, session.HTTPClient(10*time.Second), session.URL("/v1/images/generations"), request)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("image fixture returned HTTP %d: %s", response.StatusCode, response.Body)
	}
	var body struct {
		Created int64 `json:"created"`
		Data    []struct {
			Base64        string `json:"b64_json"`
			RevisedPrompt string `json:"revised_prompt"`
		} `json:"data"`
	}
	if decodeErr := response.DecodeJSON(&body); decodeErr != nil {
		return decodeErr
	}
	if body.Created <= 0 || len(body.Data) != 2 {
		return fmt.Errorf("image fixture lost created timestamp or requested image count: created=%d, count=%d", body.Created, len(body.Data))
	}
	for index, image := range body.Data {
		if image.RevisedPrompt != prompt {
			return fmt.Errorf("image %d lost the prompt marker", index)
		}
		raw, decodeErr := base64.StdEncoding.DecodeString(image.Base64)
		if decodeErr != nil {
			return fmt.Errorf("image %d is not base64: %w", index, decodeErr)
		}
		decoded, decodeErr := png.Decode(bytes.NewReader(raw))
		if decodeErr != nil {
			return fmt.Errorf("image %d is not a valid PNG: %w", index, decodeErr)
		}
		if decoded.Bounds().Dx() != 1 || decoded.Bounds().Dy() != 1 {
			return fmt.Errorf("image %d differs from the deterministic 1x1 fixture", index)
		}
	}
	request["unknown_provider_field"] = true
	invalid, err := fixtures.DoPOSTRequest(ctx, session.HTTPClient(10*time.Second), session.URL("/v1/images/generations"), request)
	if err != nil {
		return err
	}
	var failure struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if decodeErr := invalid.DecodeJSON(&failure); decodeErr != nil {
		return decodeErr
	}
	if invalid.StatusCode != http.StatusBadRequest || failure.Error.Message == "" {
		return fmt.Errorf("unknown image field did not return the native OpenAI error envelope: HTTP %d: %s", invalid.StatusCode, invalid.Body)
	}
	return nil
}
