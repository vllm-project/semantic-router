package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
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

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	apiClient := fixtures.NewResponseAPIClient(session, 30*time.Second)
	model := "openai/gpt-oss-20b"
	instructions := "You are a helpful assistant. Preserve this instruction across turns."
	turn1 := "turn-1: hello"
	turn2 := "turn-2: follow up"
	turn3 := "turn-3: final"
	storeTrue := true

	resp1, _, err := executeConversationTurn(ctx, apiClient, fixtures.ResponseAPIRequest{
		Model:        model,
		Input:        turn1,
		Instructions: instructions,
		Store:        &storeTrue,
		Metadata:     map[string]string{"test": "response-api-conversation-chaining", "turn": "1"},
	}, "", []string{turn1}, instructions)
	if err != nil {
		return fmt.Errorf("turn 1 request failed: %w", err)
	}

	resp2, _, err := executeConversationTurn(ctx, apiClient, fixtures.ResponseAPIRequest{
		Model:              model,
		Input:              turn2,
		PreviousResponseID: resp1.ID,
		Store:              &storeTrue,
		Metadata:           map[string]string{"test": "response-api-conversation-chaining", "turn": "2"},
	}, resp1.ID, []string{turn1, turn2}, instructions)
	if err != nil {
		return fmt.Errorf("turn 2 request failed: %w", err)
	}

	resp3, echo3, err := executeConversationTurn(ctx, apiClient, fixtures.ResponseAPIRequest{
		Model:              model,
		Input:              turn3,
		PreviousResponseID: resp2.ID,
		Store:              &storeTrue,
		Metadata:           map[string]string{"test": "response-api-conversation-chaining", "turn": "3"},
	}, resp2.ID, []string{turn1, turn2, turn3}, instructions)
	if err != nil {
		return fmt.Errorf("turn 3 request failed: %w", err)
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

	return nil
}

func executeConversationTurn(
	ctx context.Context,
	apiClient *fixtures.ResponseAPIClient,
	request fixtures.ResponseAPIRequest,
	expectedPreviousID string,
	expectedHistory []string,
	expectedInstructions string,
) (*fixtures.ResponseAPIResponse, *mockVLLMEcho, error) {
	response, raw, err := apiClient.Create(ctx, request)
	if err != nil {
		return nil, nil, err
	}
	if expectedPreviousID != "" && response.PreviousResponseID != expectedPreviousID {
		return nil, nil, fmt.Errorf("previous_response_id mismatch: got %q, expected %q", response.PreviousResponseID, expectedPreviousID)
	}

	echo, err := parseMockEcho(response, raw.Body)
	if err != nil {
		return nil, nil, err
	}
	if !containsInOrder(echo.User, expectedHistory) {
		return nil, nil, fmt.Errorf("backend user messages missing history: user=%v, expected in-order=%v", echo.User, expectedHistory)
	}
	if expectedInstructions != "" && !slices.Contains(echo.System, expectedInstructions) {
		return nil, nil, fmt.Errorf("backend did not receive inherited instructions: system=%v, expected=%q", echo.System, expectedInstructions)
	}

	return response, echo, nil
}

func parseMockEcho(apiResp *fixtures.ResponseAPIResponse, rawBody []byte) (*mockVLLMEcho, error) {
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
