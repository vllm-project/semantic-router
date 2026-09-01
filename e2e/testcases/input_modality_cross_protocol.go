package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("input-modality-cross-protocol", pkgtestcases.TestCase{
		Description: "The input_modality signal matches the same way for Responses and Anthropic clients",
		Tags:        []string{"signal-decision", "input-modality", "response-api", "multimodal"},
		Fn:          testInputModalityCrossProtocol,
	})
}

// inputModalityCrossProtocolPNG is the base64 payload of a 1x1 transparent
// PNG, used both as a Responses image_url data URI and as an Anthropic base64
// image source.
const inputModalityCrossProtocolPNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

type inputModalityWireCase struct {
	name           string
	path           string
	payload        map[string]interface{}
	wantDecision   string
	wantMatched    []string
	wantNotMatched []string
}

// testInputModalityCrossProtocol pins the cross-API half of the input_modality
// contract at the wire: a Responses input_image part and an Anthropic image
// block select the vision decision exactly like a Chat Completions image_url
// part does (covered by input-modality-routing), while text-only requests on
// both protocols stay on the text route. Responses officially carries no audio
// content type, and Anthropic has no audio block, so audio coverage lives with
// the Chat Completions and classify/eval cases.
func testInputModalityCrossProtocol(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	imageDataURI := "data:image/png;base64," + inputModalityCrossProtocolPNG
	cases := []inputModalityWireCase{
		{
			name: "responses image request selects the vision decision",
			path: "/v1/responses",
			payload: map[string]interface{}{
				"model": "MoM",
				"store": false,
				"input": []map[string]interface{}{{
					"role": "user",
					"content": []map[string]interface{}{
						{"type": "input_text", "text": "what is shown in this image?"},
						{"type": "input_image", "image_url": imageDataURI},
					},
				}},
			},
			wantDecision:   "input_modality_vision_decision",
			wantMatched:    []string{"image_input", "text_input"},
			wantNotMatched: nil,
		},
		{
			name: "responses text request stays on the text decision",
			path: "/v1/responses",
			payload: map[string]interface{}{
				"model": "MoM",
				"store": false,
				"input": "describe the mona lisa",
			},
			wantDecision:   "input_modality_text_decision",
			wantMatched:    []string{"text_input"},
			wantNotMatched: []string{"image_input"},
		},
		{
			name: "anthropic image request selects the vision decision",
			path: "/v1/messages",
			payload: map[string]interface{}{
				"model":      "MoM",
				"max_tokens": 64,
				"messages": []map[string]interface{}{{
					"role": "user",
					"content": []map[string]interface{}{
						{"type": "text", "text": "what is shown in this image?"},
						{"type": "image", "source": map[string]string{
							"type": "base64", "media_type": "image/png", "data": inputModalityCrossProtocolPNG,
						}},
					},
				}},
			},
			wantDecision:   "input_modality_vision_decision",
			wantMatched:    []string{"image_input", "text_input"},
			wantNotMatched: nil,
		},
		{
			name: "anthropic text request stays on the text decision",
			path: "/v1/messages",
			payload: map[string]interface{}{
				"model":      "MoM",
				"max_tokens": 64,
				"messages": []map[string]interface{}{{
					"role": "user", "content": "describe the mona lisa",
				}},
			},
			wantDecision:   "input_modality_text_decision",
			wantMatched:    []string{"text_input"},
			wantNotMatched: []string{"image_input"},
		},
	}

	for _, tc := range cases {
		if err := runInputModalityWireCase(ctx, session, tc); err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"cases":     len(cases),
			"protocols": "responses,anthropic",
		})
	}
	return nil
}

func runInputModalityWireCase(ctx context.Context, session *fixtures.ServiceSession, tc inputModalityWireCase) error {
	encoded, err := json.Marshal(tc.payload)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, session.BaseURL()+tc.path, bytes.NewReader(encoded))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	if tc.path == "/v1/messages" {
		request.Header.Set("anthropic-version", "2023-06-01")
	}
	// v0.4 demotes the matched-signal headers behind x-vsr-debug (#2205);
	// opt in so the assertions can read x-vsr-matched-input-modality.
	request.Header.Set("x-vsr-debug", "true")
	response, err := session.HTTPClient(45 * time.Second).Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return err
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s returned HTTP %d: %s", tc.path, response.StatusCode, truncateString(string(body), 500))
	}
	if decision := response.Header.Get("x-vsr-selected-decision"); decision != tc.wantDecision {
		return fmt.Errorf("selected decision = %q, want %q (body: %s)",
			decision, tc.wantDecision, truncateString(string(body), 200))
	}
	return assertInputModalityHeader(response.Header.Get("x-vsr-matched-input-modality"), tc.wantMatched, tc.wantNotMatched)
}
