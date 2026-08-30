package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("input-modality-routing", pkgtestcases.TestCase{
		Description: "Route deterministically from structural input-modality presence",
		Tags:        []string{"signal-decision", "input-modality", "routing", "multimodal"},
		Fn:          testInputModalityRouting,
	})
}

// inputModalityProbeImage is a 1x1 transparent PNG data URI.
const inputModalityProbeImage = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

type inputModalityCase struct {
	name           string
	content        interface{}
	wantDecision   string
	wantMatched    []string
	wantNotMatched []string
}

func testInputModalityRouting(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	cases := []inputModalityCase{
		{
			name: "image request selects the vision decision",
			content: []map[string]interface{}{
				{"type": "text", "text": "what is shown in this image?"},
				{"type": "image_url", "image_url": map[string]string{"url": inputModalityProbeImage}},
			},
			wantDecision:   "input_modality_vision_decision",
			wantMatched:    []string{"image_input", "text_input"},
			wantNotMatched: []string{"audio_input"},
		},
		{
			name: "audio request selects the audio decision",
			content: []map[string]interface{}{
				{"type": "input_audio", "input_audio": map[string]string{"data": "aGVsbG8=", "format": "wav"}},
			},
			wantDecision:   "input_modality_audio_decision",
			wantMatched:    []string{"audio_input"},
			wantNotMatched: []string{"image_input", "text_input"},
		},
		{
			name:           "text-only request does not match the media decisions",
			content:        "describe the mona lisa",
			wantDecision:   "input_modality_text_decision",
			wantMatched:    []string{"text_input"},
			wantNotMatched: []string{"image_input", "audio_input"},
		},
	}

	for _, tc := range cases {
		if err := runInputModalityCase(ctx, localPort, tc); err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"cases":  len(cases),
			"recipe": "input-modality-probe",
		})
	}
	return nil
}

func runInputModalityCase(ctx context.Context, localPort string, tc inputModalityCase) error {
	payload := map[string]interface{}{
		"model": "vllm-sr/input-modality-probe",
		"messages": []map[string]interface{}{
			{"role": "user", "content": tc.content},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		fmt.Sprintf("http://localhost:%s/v1/chat/completions", localPort),
		bytes.NewReader(body),
	)
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 30 * time.Second}).Do(request)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		responseBody, _ := io.ReadAll(response.Body)
		return fmt.Errorf("input-modality routing returned HTTP %d: %s", response.StatusCode, responseBody)
	}
	if decision := response.Header.Get("x-vsr-selected-decision"); decision != tc.wantDecision {
		return fmt.Errorf("selected decision = %q, want %q", decision, tc.wantDecision)
	}
	matched := response.Header.Get("x-vsr-matched-input-modality")
	for _, name := range tc.wantMatched {
		if !headerListContains(matched, name) {
			return fmt.Errorf("x-vsr-matched-input-modality = %q, want it to contain %q", matched, name)
		}
	}
	for _, name := range tc.wantNotMatched {
		if headerListContains(matched, name) {
			return fmt.Errorf("x-vsr-matched-input-modality = %q, must not contain %q", matched, name)
		}
	}
	return nil
}

func headerListContains(headerValue string, name string) bool {
	for _, entry := range strings.Split(headerValue, ",") {
		if strings.TrimSpace(entry) == name {
			return true
		}
	}
	return false
}
