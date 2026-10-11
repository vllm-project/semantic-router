package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("sticky-tool-selection", pkgtestcases.TestCase{
		Description: "Session-scoped sticky tool selection keeps an ordered, bounded, principal-isolated tool set across turns",
		Tags:        []string{"kubernetes", "plugin", "tool-selection"},
		Fn:          testStickyToolSelectionE2E,
	})
}

const stickyToolSelectionMarker = "__TOOL_SELECTION_STICKY__"

// stickyE2ETurn is one request to the sticky filter-mode decision and the
// ordered tool array the provider must receive for it.
type stickyE2ETurn struct {
	name      string
	prompt    string
	principal string
	want      []string
}

func stickyE2ETools() []fixtures.ChatTool {
	params := json.RawMessage(`{"type":"object","properties":{}}`)
	return []fixtures.ChatTool{
		{Type: "function", Function: fixtures.ChatToolFunc{Name: "get_weather", Description: "Get current weather information for a location", Parameters: params}},
		{Type: "function", Function: fixtures.ChatToolFunc{Name: "calculate", Description: "Perform mathematical calculations", Parameters: params}},
		{Type: "function", Function: fixtures.ChatToolFunc{Name: "contract_noise_alpha", Description: "Unrelated tool for cataloguing antique spoons", Parameters: params}},
	}
}

// The decision keeps the single most relevant offered tool per turn. A
// trusted session retains its earlier tool in order and adds the new one; the
// same session value under another principal starts fresh, and a request
// without an authenticated principal uses ordinary stateless selection.
func stickyE2ETurns() []stickyE2ETurn {
	weather := stickyToolSelectionMarker + " Will it rain in Seattle this weekend?"
	math := stickyToolSelectionMarker + " Calculate 17 times 23 for me."
	return []stickyE2ETurn{
		{name: "first_turn_seeds_session", prompt: weather, principal: "sticky-user-a", want: []string{"get_weather"}},
		{name: "second_turn_keeps_prefix_and_grows", prompt: math, principal: "sticky-user-a", want: []string{"get_weather", "calculate"}},
		{name: "other_principal_is_isolated", prompt: math, principal: "sticky-user-b", want: []string{"calculate"}},
		{name: "unauthenticated_request_is_stateless", prompt: math, want: []string{"calculate"}},
	}
}

func testStickyToolSelectionE2E(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	backendOpts := opts
	backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace:   "default",
		Name:        "vllm-llama3-8b-instruct",
		ServicePort: "8000",
	}
	backend, err := fixtures.OpenServiceSession(ctx, client, backendOpts)
	if err != nil {
		return err
	}
	defer backend.Close()

	chat := fixtures.NewChatCompletionsClient(session, 45*time.Second)
	sessionID := fmt.Sprintf("sticky-e2e-%d", time.Now().UnixNano())
	turns := stickyE2ETurns()
	for i, turn := range turns {
		if err := runStickyE2ETurn(ctx, chat, backend, sessionID, i, turn); err != nil {
			return fmt.Errorf("sticky tool selection %s: %w", turn.name, err)
		}
		if opts.Verbose {
			fmt.Printf("[Test] OK   sticky %s\n", turn.name)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"turns":                      len(turns),
			"provider_boundary_verified": true,
			"principal_isolation":        true,
		})
	}
	return nil
}

func runStickyE2ETurn(
	ctx context.Context,
	chat *fixtures.ChatCompletionsClient,
	backend *fixtures.ServiceSession,
	sessionID string,
	index int,
	turn stickyE2ETurn,
) error {
	traceID := fmt.Sprintf("%s-turn-%d", sessionID, index)
	headers := map[string]string{
		"x-session-id":          sessionID,
		"x-vsr-test-session-id": traceID,
	}
	if turn.principal != "" {
		headers["x-authz-user-id"] = turn.principal
	}
	request := fixtures.ChatCompletionsRequest{
		Model:      "e2e-plugins",
		Messages:   []fixtures.ChatMessage{{Role: "user", Content: turn.prompt}},
		Tools:      stickyE2ETools(),
		ToolChoice: json.RawMessage(`"auto"`),
	}
	resp, err := chat.Create(ctx, request, headers)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, truncateString(string(resp.Body), 500))
	}
	if decision := resp.Headers.Get("x-vsr-selected-decision"); decision != "tool_selection_sticky_decision" {
		return fmt.Errorf("decision = %q, want tool_selection_sticky_decision", decision)
	}
	observed, err := lastProviderSimulatorRequest(ctx, backend, traceID)
	if err != nil {
		return err
	}
	got, err := orderedProviderToolNames(observed)
	if err != nil {
		return err
	}
	if strings.Join(got, ",") != strings.Join(turn.want, ",") {
		return fmt.Errorf("provider tools = %v, want %v in order", got, turn.want)
	}
	return nil
}

// orderedProviderToolNames returns the provider-bound tool names in the order
// the provider received them; sticky selection promises a stable prefix.
func orderedProviderToolNames(observed []byte) ([]string, error) {
	var request struct {
		Body struct {
			Tools []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
		} `json:"body"`
	}
	if err := json.Unmarshal(observed, &request); err != nil {
		return nil, fmt.Errorf("decode provider-bound request: %w", err)
	}
	names := make([]string, 0, len(request.Body.Tools))
	for _, tool := range request.Body.Tools {
		names = append(names, tool.Function.Name)
	}
	return names, nil
}
