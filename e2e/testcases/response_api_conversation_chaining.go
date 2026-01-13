package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"slices"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("response-api-conversation-chaining", pkgtestcases.TestCase{
		Description: "Conversation chaining with previous_response_id (3-turn conversation chain)",
		Tags:        []string{"response-api", "functional"},
		Fn:          testResponseAPIConversationChaining,
	})
}

type mockVLLMEcho struct {
	Mock          string   `json:"mock"`
	Model         string   `json:"model"`
	Roles         []string `json:"roles"`
	System        []string `json:"system"`
	User          []string `json:"user"`
	TotalMessages int      `json:"total_messages"`
}

func testResponseAPIConversationChaining(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Response API: conversation chaining")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	httpClient := &http.Client{Timeout: 30 * time.Second}

	model := "openai/gpt-oss-20b"
	instructions := "You are a helpful assistant. Preserve this instruction across turns."
	turn1 := "turn-1: hello"
	turn2 := "turn-2: follow up"
	turn3 := "turn-3: final"

	storeTrue := true

	// 1) Create initial response (turn 1)
	resp1, raw1, err := postResponseAPI(ctx, httpClient, localPort, ResponseAPIRequest{
		Model:        model,
		Input:        turn1,
		Instructions: instructions,
		Store:        &storeTrue,
		Metadata:     map[string]string{"test": "response-api-conversation-chaining", "turn": "1"},
	})
	if err != nil {
		return fmt.Errorf("turn 1 request failed: %w", err)
	}
	echo1, err := parseMockEcho(resp1, raw1)
	if err != nil {
		return fmt.Errorf("turn 1 echo parse failed: %w", err)
	}
	if !slices.Contains(echo1.User, turn1) {
		return fmt.Errorf("turn 1 backend did not receive user input %q: user=%v", turn1, echo1.User)
	}

	// 2) Follow-up with previous_response_id (turn 2)
	resp2, raw2, err := postResponseAPI(ctx, httpClient, localPort, ResponseAPIRequest{
		Model:              model,
		Input:              turn2,
		PreviousResponseID: resp1.ID,
		Store:              &storeTrue,
		Metadata:           map[string]string{"test": "response-api-conversation-chaining", "turn": "2"},
	})
	if err != nil {
		return fmt.Errorf("turn 2 request failed: %w", err)
	}
	if resp2.PreviousResponseID != resp1.ID {
		return fmt.Errorf("turn 2 previous_response_id mismatch: got %q, expected %q", resp2.PreviousResponseID, resp1.ID)
	}
	echo2, err := parseMockEcho(resp2, raw2)
	if err != nil {
		return fmt.Errorf("turn 2 echo parse failed: %w", err)
	}
	if !containsInOrder(echo2.User, []string{turn1, turn2}) {
		return fmt.Errorf("turn 2 backend user messages missing history: user=%v, expected in-order=%v", echo2.User, []string{turn1, turn2})
	}
	if !slices.Contains(echo2.System, instructions) {
		return fmt.Errorf("turn 2 backend did not receive inherited instructions: system=%v, expected=%q", echo2.System, instructions)
	}

	// 3) Chain a third response (turn 3)
	resp3, raw3, err := postResponseAPI(ctx, httpClient, localPort, ResponseAPIRequest{
		Model:              model,
		Input:              turn3,
		PreviousResponseID: resp2.ID,
		Store:              &storeTrue,
		Metadata:           map[string]string{"test": "response-api-conversation-chaining", "turn": "3"},
	})
	if err != nil {
		return fmt.Errorf("turn 3 request failed: %w", err)
	}
	if resp3.PreviousResponseID != resp2.ID {
		return fmt.Errorf("turn 3 previous_response_id mismatch: got %q, expected %q", resp3.PreviousResponseID, resp2.ID)
	}
	echo3, err := parseMockEcho(resp3, raw3)
	if err != nil {
		return fmt.Errorf("turn 3 echo parse failed: %w", err)
	}
	if !containsInOrder(echo3.User, []string{turn1, turn2, turn3}) {
		return fmt.Errorf("turn 3 backend user messages missing history: user=%v, expected in-order=%v", echo3.User, []string{turn1, turn2, turn3})
	}
	if !slices.Contains(echo3.System, instructions) {
		return fmt.Errorf("turn 3 backend did not receive inherited instructions: system=%v, expected=%q", echo3.System, instructions)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"turn1_response_id": resp1.ID,
			"turn2_response_id": resp2.ID,
			"turn3_response_id": resp3.ID,
			"turn3_user_count":  len(echo3.User),
			"turn3_total_msgs":  echo3.TotalMessages,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Response API conversation chaining successful (ids=%s -> %s -> %s)\n", resp1.ID, resp2.ID, resp3.ID)
	}

	return nil
}

func postResponseAPI(ctx context.Context, httpClient *http.Client, localPort string, reqBody ResponseAPIRequest) (*ResponseAPIResponse, []byte, error) {
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/responses", localPort)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, body, fmt.Errorf("expected status 200, got %d: %s", resp.StatusCode, string(body))
	}

	var apiResp ResponseAPIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, body, fmt.Errorf("failed to parse response: %w", err)
	}

	if apiResp.ID == "" {
		return nil, body, fmt.Errorf("missing response id: %s", string(body))
	}

	return &apiResp, body, nil
}

func parseMockEcho(apiResp *ResponseAPIResponse, rawBody []byte) (*mockVLLMEcho, error) {
	if apiResp == nil {
		return nil, fmt.Errorf("nil api response")
	}
	if apiResp.OutputText == "" {
		return nil, fmt.Errorf("missing output_text in response: %s", truncateString(string(rawBody), 500))
	}
	var echo mockVLLMEcho
	if err := json.Unmarshal([]byte(apiResp.OutputText), &echo); err != nil {
		return nil, fmt.Errorf("output_text is not valid mock-vllm JSON echo: %w (output_text=%q)", err, truncateString(apiResp.OutputText, 200))
	}
	if echo.Mock != "mock-vllm" {
		return nil, fmt.Errorf("unexpected mock backend marker: got %q, want %q", echo.Mock, "mock-vllm")
	}
	return &echo, nil
}

func containsInOrder(haystack, needle []string) bool {
	if len(needle) == 0 {
		return true
	}
	i := 0
	for _, item := range haystack {
		if item == needle[i] {
			i++
			if i == len(needle) {
				return true
			}
		}
	}
	return false
}
