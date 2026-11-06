package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("tool-selection", pkgtestcases.TestCase{
		Description: "The tools plugin policy: provider-bound mutations, config variants, and PII precedence",
		Tags:        []string{"kubernetes", "plugin", "tool-selection"},
		Fn:          testToolSelectionE2E,
	})
}

type toolSelectionE2ECase struct {
	Name                   string
	Model                  string
	Prompt                 string
	Tools                  []fixtures.ChatTool
	ExpectDecision         string
	ExpectFastResponse     *bool
	ExpectProviderTools    []string
	ExpectNoProviderTools  bool
	ExpectToolChoiceAbsent bool
}

func defaultContractTools(minObjectParams json.RawMessage) []fixtures.ChatTool {
	return []fixtures.ChatTool{
		{Type: "function", Function: fixtures.ChatToolFunc{Name: "get_weather", Description: "Get current weather information for a location", Parameters: minObjectParams}},
		{Type: "function", Function: fixtures.ChatToolFunc{Name: "calculate", Description: "Perform mathematical calculations", Parameters: minObjectParams}},
		{Type: "function", Function: fixtures.ChatToolFunc{Name: "search_web", Description: "Search web resources for recent information", Parameters: minObjectParams}},
	}
}

func toolSelectionContractCases(minObjectParams json.RawMessage) []toolSelectionE2ECase {
	contractTools := defaultContractTools(minObjectParams)
	cases := []toolSelectionE2ECase{
		{
			Name:                "tools_passthrough_preserves_request_tools",
			Model:               "e2e-plugins",
			Prompt:              "__TOOLS_PASSTHROUGH__ What is the weather forecast for Boston tomorrow?",
			Tools:               contractTools,
			ExpectDecision:      "tools_passthrough_decision",
			ExpectProviderTools: []string{"calculate", "get_weather", "search_web"},
		},
		{
			Name:                "tools_filtered_applies_allow_list",
			Model:               "e2e-plugins",
			Prompt:              "__TOOLS_FILTERED__ keep only the weather tool",
			Tools:               contractTools,
			ExpectDecision:      "tools_filtered_decision",
			ExpectProviderTools: []string{"get_weather"},
		},
		{
			Name:                   "tools_none_removes_tools_and_choice",
			Model:                  "e2e-plugins",
			Prompt:                 "__TOOLS_NONE__ What is the weather forecast for Boston tomorrow?",
			Tools:                  contractTools,
			ExpectDecision:         "tools_none_decision",
			ExpectNoProviderTools:  true,
			ExpectToolChoiceAbsent: true,
		},
	}
	frTrue := true
	cases = append(cases, toolSelectionE2ECase{
		Name:               "pii_decision_runs_before_tools",
		Model:              "e2e-pii-precedence",
		Prompt:             "__TOOLS_PASSTHROUGH__ My payment card is 4111111111111111 and I need a weather forecast for Miami. __PII_SPAN__ CREDIT_CARD 4111111111111111",
		ExpectDecision:     "block_pii",
		ExpectFastResponse: &frTrue,
	})
	return cases
}

func testToolSelectionE2E(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] tools plugin provider-boundary contract")
	}

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
	backendSession, err := fixtures.OpenServiceSession(ctx, client, backendOpts)
	if err != nil {
		return err
	}
	defer backendSession.Close()

	chat := fixtures.NewChatCompletionsClient(session, 45*time.Second)
	minObjectParams := json.RawMessage(`{"type":"object","properties":{}}`)
	cases := toolSelectionContractCases(minObjectParams)

	var failed int
	var firstErr error
	for _, tc := range cases {
		if err := runToolSelectionCase(ctx, chat, backendSession, tc, opts.Verbose); err != nil {
			failed++
			if firstErr == nil {
				firstErr = fmt.Errorf("%s: %w", tc.Name, err)
			}
			if opts.Verbose {
				fmt.Printf("[Test] FAIL %s: %v\n", tc.Name, err)
			}
		} else if opts.Verbose {
			fmt.Printf("[Test] OK   %s\n", tc.Name)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"cases_total":                len(cases),
			"cases_failed":               failed,
			"cases_passed":               len(cases) - failed,
			"plugins_verified":           []string{"tools"},
			"provider_boundary_verified": true,
		})
	}

	if failed > 0 {
		if firstErr != nil {
			return fmt.Errorf("tools: %d/%d cases failed: %w", failed, len(cases), firstErr)
		}
		return fmt.Errorf("tools: %d/%d cases failed", failed, len(cases))
	}
	return nil
}

func runToolSelectionCase(
	ctx context.Context,
	chat *fixtures.ChatCompletionsClient,
	backend *fixtures.ServiceSession,
	tc toolSelectionE2ECase,
	verbose bool,
) error {
	req := buildToolSelectionChatRequest(tc)
	sessionID := fmt.Sprintf("tool-plugin-%s-%d", tc.Name, time.Now().UnixNano())
	resp, err := chat.Create(ctx, req, map[string]string{
		"x-vsr-debug":           "true",
		"x-vsr-test-session-id": sessionID,
	})
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(resp.Body))
	}

	decision := resp.Headers.Get("x-vsr-selected-decision")
	if err := assertToolSelectionDecision(tc, decision, resp.Headers); err != nil {
		return err
	}
	if err := assertToolSelectionFastResponse(tc, resp.Headers); err != nil {
		return err
	}
	if err := assertProviderToolContract(ctx, backend, tc, sessionID); err != nil {
		return err
	}
	if verbose {
		logToolSelectionHeaders(decision, resp.Headers)
	}
	return nil
}

func assertProviderToolContract(
	ctx context.Context,
	backend *fixtures.ServiceSession,
	tc toolSelectionE2ECase,
	sessionID string,
) error {
	if tc.ExpectFastResponse != nil && *tc.ExpectFastResponse {
		observed, model, err := lookupShortCircuitDispatch(ctx, backend, sessionID)
		if err != nil {
			return err
		}
		if observed {
			return fmt.Errorf("fast_response request reached provider model %q", model)
		}
		return nil
	}
	if tc.ExpectProviderTools == nil && !tc.ExpectNoProviderTools && !tc.ExpectToolChoiceAbsent {
		return fmt.Errorf("case has no provider-boundary tool expectation")
	}

	observed, err := lastProviderSimulatorRequest(ctx, backend, sessionID)
	if err != nil {
		return err
	}
	var request struct {
		Body map[string]any `json:"body"`
	}
	if decodeErr := json.Unmarshal(observed, &request); decodeErr != nil {
		return fmt.Errorf("decode provider-bound tool request: %w", decodeErr)
	}

	actualTools, toolsPresent, err := providerToolNames(request.Body)
	if err != nil {
		return err
	}
	if tc.ExpectNoProviderTools {
		if toolsPresent && len(actualTools) > 0 {
			return fmt.Errorf("provider request retained tools in mode=none: %v", actualTools)
		}
	} else if strings.Join(actualTools, ",") != strings.Join(tc.ExpectProviderTools, ",") {
		return fmt.Errorf("provider tools = %v, want %v", actualTools, tc.ExpectProviderTools)
	}
	if tc.ExpectToolChoiceAbsent {
		if _, exists := request.Body["tool_choice"]; exists {
			return fmt.Errorf("provider request retained tool_choice without tools: %s", truncateString(string(observed), 800))
		}
	}
	return nil
}

func providerToolNames(body map[string]any) ([]string, bool, error) {
	rawTools, exists := body["tools"]
	if !exists {
		return nil, false, nil
	}
	tools, ok := rawTools.([]any)
	if !ok {
		return nil, true, fmt.Errorf("provider tools field is not an array: %#v", rawTools)
	}
	names := make([]string, 0, len(tools))
	for index, rawTool := range tools {
		tool, ok := rawTool.(map[string]any)
		if !ok {
			return nil, true, fmt.Errorf("provider tool %d is not an object: %#v", index, rawTool)
		}
		function, ok := tool["function"].(map[string]any)
		if !ok {
			return nil, true, fmt.Errorf("provider tool %d has no function object: %#v", index, rawTool)
		}
		name, ok := function["name"].(string)
		if !ok || name == "" {
			return nil, true, fmt.Errorf("provider tool %d has no function name: %#v", index, rawTool)
		}
		names = append(names, name)
	}
	sort.Strings(names)
	return names, true, nil
}

func buildToolSelectionChatRequest(tc toolSelectionE2ECase) fixtures.ChatCompletionsRequest {
	req := fixtures.ChatCompletionsRequest{
		Model: tc.Model,
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: tc.Prompt},
		},
		Tools: tc.Tools,
	}
	if len(tc.Tools) > 0 {
		req.ToolChoice = json.RawMessage(`"auto"`)
	}
	return req
}

func assertToolSelectionDecision(tc toolSelectionE2ECase, decision string, h http.Header) error {
	if tc.ExpectDecision == "" || decision == tc.ExpectDecision {
		return nil
	}
	return fmt.Errorf("decision: want %q got %q (headers=%s)", tc.ExpectDecision, decision, summarizeToolHeaders(h))
}

func assertToolSelectionFastResponse(tc toolSelectionE2ECase, h http.Header) error {
	if tc.ExpectFastResponse == nil {
		return nil
	}
	fr := h.Get("x-vsr-fast-response") == "true"
	if fr == *tc.ExpectFastResponse {
		return nil
	}
	return fmt.Errorf("x-vsr-fast-response: want %v got %v", *tc.ExpectFastResponse, fr)
}

func logToolSelectionHeaders(decision string, h http.Header) {
	fmt.Printf("  decision=%q fast=%q\n", decision, h.Get("x-vsr-fast-response"))
}

func summarizeToolHeaders(h http.Header) string {
	var b strings.Builder
	for _, k := range []string{
		"x-vsr-selected-decision",
		"x-vsr-fast-response",
		"x-vsr-injected-system-prompt",
	} {
		fmt.Fprintf(&b, "%s=%q ", k, h.Get(k))
	}
	return strings.TrimSpace(b.String())
}
