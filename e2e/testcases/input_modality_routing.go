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

	apiCases, err := runInputModalityClassifyEvalCases(ctx, client, opts)
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"cases":     len(cases),
			"api_cases": apiCases,
			"recipe":    "input-modality-probe",
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
	// v0.4 demotes the matched-signal headers behind x-vsr-debug (#2205);
	// opt in so the assertions can read x-vsr-matched-input-modality.
	request.Header.Set("x-vsr-debug", "true")
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
	return assertInputModalityHeader(response.Header.Get("x-vsr-matched-input-modality"), tc.wantMatched, tc.wantNotMatched)
}

// assertInputModalityHeader checks the x-vsr-matched-input-modality response
// header against the expected matched and unmatched rule names.
func assertInputModalityHeader(matched string, wantMatched, wantNotMatched []string) error {
	for _, name := range wantMatched {
		if !headerListContains(matched, name) {
			return fmt.Errorf("x-vsr-matched-input-modality = %q, want it to contain %q", matched, name)
		}
	}
	for _, name := range wantNotMatched {
		if headerListContains(matched, name) {
			return fmt.Errorf("x-vsr-matched-input-modality = %q, must not contain %q", matched, name)
		}
	}
	return nil
}

// inputModalityAPICase drives the classify/eval HTTP API boundary with typed
// messages. The recipe is selected through the request model, mirroring how
// the wire cases enter the input-modality-probe recipe.
type inputModalityAPICase struct {
	name           string
	content        interface{}
	wantDecision   string
	wantMatched    []string
	wantNotMatched []string
}

// inputModalitySignalDoc is the slice of an eval or intent response the
// classify/eval assertions read: the selected decision and the matched
// input_modality rule names.
type inputModalitySignalDoc struct {
	RoutingDecision string `json:"routing_decision"`
	MatchedSignals  *struct {
		InputModality []string `json:"input_modality"`
	} `json:"matched_signals"`
	DecisionResult *struct {
		DecisionName   string `json:"decision_name"`
		MatchedSignals *struct {
			InputModality []string `json:"input_modality"`
		} `json:"matched_signals"`
	} `json:"decision_result"`
}

// runInputModalityClassifyEvalCases asserts the input_modality signal at the
// classify/eval API boundary: /api/v1/eval evaluates every configured rule
// (including video, which no wire protocol can carry yet) and
// /api/v1/classify/intent reports the matched rules and selected decision.
func runInputModalityClassifyEvalCases(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (int, error) {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return 0, err
	}
	defer session.Close()

	evalCases := []inputModalityAPICase{
		{
			name: "eval reports image and text for an image request",
			content: []map[string]interface{}{
				{"type": "text", "text": "what is shown in this image?"},
				{"type": "image_url", "image_url": map[string]string{"url": inputModalityProbeImage}},
			},
			wantDecision:   "input_modality_vision_decision",
			wantMatched:    []string{"image_input", "text_input"},
			wantNotMatched: []string{"audio_input", "video_input"},
		},
		{
			name: "eval reports audio for an audio request",
			content: []map[string]interface{}{
				{"type": "input_audio", "input_audio": map[string]string{"data": "aGVsbG8=", "format": "wav"}},
			},
			wantDecision:   "input_modality_audio_decision",
			wantMatched:    []string{"audio_input"},
			wantNotMatched: []string{"image_input", "text_input", "video_input"},
		},
		{
			name: "eval reports video for a video request",
			content: []map[string]interface{}{
				{"type": "video_url", "video_url": map[string]string{"url": "https://example.com/clip.mp4"}},
			},
			wantDecision:   "input_modality_video_decision",
			wantMatched:    []string{"video_input"},
			wantNotMatched: []string{"image_input", "audio_input", "text_input"},
		},
		{
			name:           "eval reports text only for a text request",
			content:        "describe the mona lisa",
			wantDecision:   "input_modality_text_decision",
			wantMatched:    []string{"text_input"},
			wantNotMatched: []string{"image_input", "audio_input", "video_input"},
		},
	}
	for _, tc := range evalCases {
		document, err := postInputModalityAPI(ctx, session, "/api/v1/eval", tc.content)
		if err != nil {
			return 0, fmt.Errorf("%s: %w", tc.name, err)
		}
		if document.DecisionResult == nil || document.DecisionResult.MatchedSignals == nil {
			return 0, fmt.Errorf("%s: eval response has no decision_result.matched_signals", tc.name)
		}
		if err := assertInputModalityRules(
			document.DecisionResult.MatchedSignals.InputModality, tc.wantMatched, tc.wantNotMatched,
		); err != nil {
			return 0, fmt.Errorf("%s: %w", tc.name, err)
		}
		if document.DecisionResult.DecisionName != tc.wantDecision {
			return 0, fmt.Errorf("%s: decision_name = %q, want %q", tc.name, document.DecisionResult.DecisionName, tc.wantDecision)
		}
	}

	if err := runInputModalityIntentCase(ctx, session); err != nil {
		return 0, err
	}

	return len(evalCases) + 1, nil
}

// runInputModalityIntentCase asserts the classify boundary: the intent
// endpoint reports the matched input_modality rules and the selected decision
// for an image-bearing message.
func runInputModalityIntentCase(ctx context.Context, session *fixtures.ServiceSession) error {
	intentContent := []map[string]interface{}{
		{"type": "text", "text": "what is shown in this image?"},
		{"type": "image_url", "image_url": map[string]string{"url": inputModalityProbeImage}},
	}
	document, err := postInputModalityAPI(ctx, session, "/api/v1/classify/intent", intentContent)
	if err != nil {
		return fmt.Errorf("classify intent image request: %w", err)
	}
	if document.MatchedSignals == nil {
		return fmt.Errorf("classify intent image request: response has no matched_signals")
	}
	if err := assertInputModalityRules(
		document.MatchedSignals.InputModality, []string{"image_input", "text_input"}, []string{"audio_input", "video_input"},
	); err != nil {
		return fmt.Errorf("classify intent image request: %w", err)
	}
	if document.RoutingDecision != "input_modality_vision_decision" {
		return fmt.Errorf("classify intent image request: routing_decision = %q, want %q",
			document.RoutingDecision, "input_modality_vision_decision")
	}
	return nil
}

func postInputModalityAPI(
	ctx context.Context,
	session *fixtures.ServiceSession,
	path string,
	content interface{},
) (*inputModalitySignalDoc, error) {
	payload := map[string]interface{}{
		"model": "vllm-sr/input-modality-probe",
		"messages": []map[string]interface{}{
			{"role": "user", "content": content},
		},
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, session.BaseURL()+path, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := session.HTTPClient(30 * time.Second).Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s returned HTTP %d: %s", path, response.StatusCode, truncateString(string(responseBody), 500))
	}
	var document inputModalitySignalDoc
	if err := json.Unmarshal(responseBody, &document); err != nil {
		return nil, fmt.Errorf("decode %s response: %w", path, err)
	}
	return &document, nil
}

func assertInputModalityRules(matched, wantMatched, wantNotMatched []string) error {
	for _, name := range wantMatched {
		if !stringSliceContains(matched, name) {
			return fmt.Errorf("matched input_modality rules %v, want them to contain %q", matched, name)
		}
	}
	for _, name := range wantNotMatched {
		if stringSliceContains(matched, name) {
			return fmt.Errorf("matched input_modality rules %v, must not contain %q", matched, name)
		}
	}
	return nil
}

func stringSliceContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
