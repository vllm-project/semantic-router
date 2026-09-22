package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	stickyToolSelectionDecision = "sticky_tool_selection_decision"
	stickyToolSelectionMaxTools = 2
	stickyToolSelectionModel    = "e2e-plugins"
)

func init() {
	pkgtestcases.Register("sticky-tool-selection", pkgtestcases.TestCase{
		Description: "Verify trusted-session sticky tool reuse, bounded growth, invalidation, pinning, and provider-prefix stability",
		Tags:        []string{"kubernetes", "plugin", "tool-selection", "sticky"},
		Fn:          testStickyToolSelection,
	})
}

type stickyProviderRequest struct {
	Body struct {
		Tools []json.RawMessage `json:"tools"`
	} `json:"body"`
}

type stickyToolSnapshot struct {
	Tools []json.RawMessage
	Names []string
}

type stickyHTTPResponse struct {
	StatusCode int
	Headers    http.Header
	Body       []byte
}

type stickySessionPair struct {
	gateway *fixtures.ServiceSession
	backend *fixtures.ServiceSession
}

func testStickyToolSelection(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Test] sticky tool-selection contract")
	}

	sessions, err := openStickySessionPair(ctx, client, opts)
	if err != nil {
		return err
	}
	defer sessions.gateway.Close()
	defer sessions.backend.Close()

	sessionID := fmt.Sprintf("sticky-e2e-%d", time.Now().UnixNano())
	trustedHeaders := stickyRequestHeaders(sessionID, true)
	contractTools := stickyContractTools(1, "")

	firstTools, secondTools, err := runStickyTrustedTurns(ctx, sessions, sessionID, trustedHeaders, contractTools)
	if err != nil {
		return err
	}
	calledTools, err := runStickyCalledToolPin(ctx, sessions, sessionID, trustedHeaders, contractTools)
	if err != nil {
		return err
	}
	if err := assertStickyReplacement(secondTools, calledTools); err != nil {
		return err
	}
	invalidationSnapshot, err := runStickyInvalidation(ctx, sessions, sessionID, trustedHeaders, calledTools)
	if err != nil {
		return err
	}
	concurrentTools, err := runStickyConcurrentTurns(ctx, sessions, contractTools)
	if err != nil {
		return err
	}
	trustedControl, untrustedTools, baselineTools, err := runStickyUntrustedComparison(ctx, sessions, contractTools)
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"first_turn_tools":          firstTools.Names,
			"second_turn_tools":         secondTools.Names,
			"replacement_tools":         calledTools.Names,
			"invalidation_tools":        invalidationSnapshot.Names,
			"concurrent_tools":          concurrentTools.Names,
			"trusted_control_tools":     trustedControl.Names,
			"untrusted_tools":           untrustedTools.Names,
			"stateless_baseline_tools":  baselineTools.Names,
			"max_tools_observed":        maxStickyTools(firstTools, secondTools, calledTools, invalidationSnapshot, concurrentTools, trustedControl, untrustedTools, baselineTools),
			"trusted_growth":            len(secondTools.Tools) - len(firstTools.Tools),
			"schema_invalidation_count": len(invalidationSnapshot.Tools),
		})
	}
	return nil
}

func openStickySessionPair(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*stickySessionPair, error) {
	gateway, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	backendOpts := opts
	backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace:   "default",
		Name:        "vllm-llama3-8b-instruct",
		ServicePort: "8000",
	}
	backend, err := fixtures.OpenServiceSession(ctx, client, backendOpts)
	if err != nil {
		gateway.Close()
		return nil, err
	}
	return &stickySessionPair{gateway: gateway, backend: backend}, nil
}

func runStickyTrustedTurns(
	ctx context.Context,
	sessions *stickySessionPair,
	sessionID string,
	headers map[string]string,
	tools []fixtures.ChatTool,
) (stickyToolSnapshot, stickyToolSnapshot, error) {
	first, err := runStickyTurn(ctx, sessions, sessionID, stickyNormalRequest(
		"__STICKY_TOOL_SELECTION__ Find the current weather for Seattle.", tools), headers, "first trusted turn")
	if err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, err
	}
	second, err := runStickyTurn(ctx, sessions, sessionID, stickyNormalRequest(
		"__STICKY_TOOL_SELECTION__ Calculate 17 times 23.", tools), headers, "second trusted turn")
	if err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, err
	}
	if err := assertStickyReuse(first, second); err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, err
	}
	return first, second, nil
}

func runStickyCalledToolPin(
	ctx context.Context,
	sessions *stickySessionPair,
	sessionID string,
	headers map[string]string,
	tools []fixtures.ChatTool,
) (stickyToolSnapshot, error) {
	body := map[string]any{
		"model": stickyToolSelectionModel,
		"messages": []any{
			map[string]any{
				"role":    "assistant",
				"content": nil,
				"tool_calls": []any{map[string]any{
					"id":   "call-sticky-1",
					"type": "function",
					"function": map[string]any{
						"name":      "search_web",
						"arguments": `{"query":"Seattle weather"}`,
					},
				}},
			},
			map[string]any{
				"role":    "user",
				"content": "__STICKY_TOOL_SELECTION__ Calculate 29 times 31.",
			},
		},
		"tools":       tools,
		"tool_choice": "auto",
	}
	called, err := runStickyTurn(ctx, sessions, sessionID, body, headers, "called-tool pin turn")
	if err != nil {
		return stickyToolSnapshot{}, err
	}
	if !containsStickyTool(called, "search_web") {
		return stickyToolSnapshot{}, fmt.Errorf("called-tool pin turn omitted previously called tool %q", "search_web")
	}

	retained, err := runStickyTurn(ctx, sessions, sessionID, stickyNormalRequest(
		"__STICKY_TOOL_SELECTION__ Calculate 29 times 31.", tools), headers, "called-tool retention turn")
	if err != nil {
		return stickyToolSnapshot{}, err
	}
	calledDefinition := toolDefinitionByName(called, "search_web")
	retainedDefinition := toolDefinitionByName(retained, "search_web")
	if calledDefinition == nil || retainedDefinition == nil || !bytes.Equal(calledDefinition, retainedDefinition) {
		return stickyToolSnapshot{}, fmt.Errorf("called tool %q was not retained with stable provider bytes", "search_web")
	}
	return retained, nil
}

func assertStickyReplacement(previous, replacement stickyToolSnapshot) error {
	if len(previous.Tools) != stickyToolSelectionMaxTools {
		return fmt.Errorf("replacement precondition forwarded %d tools, want %d", len(previous.Tools), stickyToolSelectionMaxTools)
	}
	if len(replacement.Tools) > stickyToolSelectionMaxTools {
		return fmt.Errorf("replacement forwarded %d tools, max is %d", len(replacement.Tools), stickyToolSelectionMaxTools)
	}
	if !containsStickyTool(replacement, "search_web") {
		return fmt.Errorf("replacement omitted called tool %q", "search_web")
	}
	for _, name := range previous.Names {
		if !containsStickyTool(replacement, name) {
			return nil
		}
	}
	return fmt.Errorf("called-tool pin did not replace an unpinned retained tool: previous=%v replacement=%v", previous.Names, replacement.Names)
}

func runStickyConcurrentTurns(
	ctx context.Context,
	sessions *stickySessionPair,
	tools []fixtures.ChatTool,
) (stickyToolSnapshot, error) {
	sessionID := fmt.Sprintf("sticky-e2e-concurrent-%d", time.Now().UnixNano())
	headers := stickyRequestHeaders(sessionID, true)
	requests := []fixtures.ChatCompletionsRequest{
		stickyNormalRequest("__STICKY_TOOL_SELECTION__ Find the current weather for Seattle.", tools),
		stickyNormalRequest("__STICKY_TOOL_SELECTION__ Calculate 17 times 23.", tools),
		stickyNormalRequest("__STICKY_TOOL_SELECTION__ Search recent weather reports.", tools),
	}

	errs := make(chan error, len(requests))
	var wait sync.WaitGroup
	for _, request := range requests {
		request := request
		wait.Add(1)
		go func() {
			defer wait.Done()
			response, err := sendStickyChatRequest(ctx, sessions.gateway, request, headers)
			if err != nil {
				errs <- err
				return
			}
			errs <- assertStickyResponse(response)
		}()
	}
	wait.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			return stickyToolSnapshot{}, fmt.Errorf("concurrent sticky turn: %w", err)
		}
	}

	final, err := runStickyTurn(ctx, sessions, sessionID, stickyNormalRequest(
		"__STICKY_TOOL_SELECTION__ Calculate 29 times 31.", tools), headers, "post-concurrency turn")
	if err != nil {
		return stickyToolSnapshot{}, err
	}
	repeated, err := runStickyTurn(ctx, sessions, sessionID, stickyNormalRequest(
		"__STICKY_TOOL_SELECTION__ Calculate 29 times 31.", tools), headers, "repeated post-concurrency turn")
	if err != nil {
		return stickyToolSnapshot{}, err
	}
	if err := assertStickySnapshotsEqual(final, repeated); err != nil {
		return stickyToolSnapshot{}, fmt.Errorf("concurrent state was not deterministic: %w", err)
	}
	if len(final.Tools) != stickyToolSelectionMaxTools {
		return stickyToolSnapshot{}, fmt.Errorf("concurrent state retained %d tools, want %d", len(final.Tools), stickyToolSelectionMaxTools)
	}

	baselineID := fmt.Sprintf("%s-baseline", sessionID)
	baseline, err := runStickyTurn(ctx, sessions, baselineID, stickyNormalRequest(
		"__STICKY_TOOL_SELECTION__ Calculate 29 times 31.", tools),
		stickyRequestHeaders(baselineID, false), "post-concurrency stateless baseline")
	if err != nil {
		return stickyToolSnapshot{}, err
	}
	if err := assertStickySnapshotsDifferent(final, baseline); err != nil {
		return stickyToolSnapshot{}, fmt.Errorf("concurrent turns did not produce retained state: %w", err)
	}
	return final, nil
}

func runStickyInvalidation(
	ctx context.Context,
	sessions *stickySessionPair,
	sessionID string,
	headers map[string]string,
	previous stickyToolSnapshot,
) (stickyToolSnapshot, error) {
	if len(previous.Names) == 0 {
		return stickyToolSnapshot{}, fmt.Errorf("schema invalidation requires retained tools")
	}
	changedName := previous.Names[0]
	if containsStickyTool(previous, "search_web") {
		changedName = "search_web"
	}
	tools := stickyContractTools(2, changedName)
	request := stickyNormalRequest(stickyInvalidationPrompt(changedName), tools)
	updated, err := runStickyTurn(ctx, sessions, sessionID, request, headers, "schema invalidation turn")
	if err != nil {
		return stickyToolSnapshot{}, err
	}
	baselineID := sessionID + "-invalidation-baseline"
	baseline, err := runStickyTurn(
		ctx,
		sessions,
		baselineID,
		request,
		stickyRequestHeaders(baselineID, false),
		"schema invalidation stateless baseline",
	)
	if err != nil {
		return stickyToolSnapshot{}, err
	}
	if err := assertStickyInvalidation(previous, updated, baseline, changedName); err != nil {
		return stickyToolSnapshot{}, err
	}
	return updated, nil
}

func runStickyUntrustedComparison(
	ctx context.Context,
	sessions *stickySessionPair,
	tools []fixtures.ChatTool,
) (stickyToolSnapshot, stickyToolSnapshot, stickyToolSnapshot, error) {
	sessionID := fmt.Sprintf("sticky-e2e-untrusted-%d", time.Now().UnixNano())
	trustedHeaders := stickyRequestHeaders(sessionID, true)
	_, grown, err := runStickyTrustedTurns(ctx, sessions, sessionID, trustedHeaders, tools)
	if err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, stickyToolSnapshot{}, err
	}
	if len(grown.Tools) != stickyToolSelectionMaxTools {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, stickyToolSnapshot{}, fmt.Errorf(
			"untrusted comparison precondition retained %d tools, want %d",
			len(grown.Tools),
			stickyToolSelectionMaxTools,
		)
	}

	request := map[string]any{
		"model": stickyToolSelectionModel,
		"messages": []fixtures.ChatMessage{{
			Role:    "user",
			Content: "__STICKY_TOOL_SELECTION__ Search recent weather reports.",
		}},
		"metadata": map[string]string{
			// Request metadata is client-controlled and must not become the
			// authenticated principal used to scope sticky state.
			"user_id": "sticky-e2e-user",
		},
		"tools":       tools,
		"tool_choice": "auto",
	}
	trusted, err := runStickyTurn(ctx, sessions, sessionID, request, trustedHeaders, "trusted control turn")
	if err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, stickyToolSnapshot{}, err
	}
	// Reuse the trusted session declaration but remove the principal. A
	// correctly scoped identity resolver must fail closed instead of reusing
	// the state written by the authenticated turns above.
	untrusted, err := runStickyTurn(ctx, sessions, sessionID, request,
		stickyRequestHeaders(sessionID, false), "untrusted turn")
	if err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, stickyToolSnapshot{}, err
	}
	baselineID := fmt.Sprintf("%s-baseline", sessionID)
	baseline, err := runStickyTurn(ctx, sessions, baselineID, request,
		stickyRequestHeaders(baselineID, false), "stateless baseline turn")
	if err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, stickyToolSnapshot{}, err
	}
	if err := assertStickySnapshotsEqual(untrusted, baseline); err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, stickyToolSnapshot{}, fmt.Errorf("untrusted turn reused trusted sticky state: %w", err)
	}
	if err := assertStickySnapshotsDifferent(untrusted, trusted); err != nil {
		return stickyToolSnapshot{}, stickyToolSnapshot{}, stickyToolSnapshot{}, fmt.Errorf("trusted control did not reuse session state: %w", err)
	}
	return trusted, untrusted, baseline, nil
}

func stickyNormalRequest(prompt string, tools []fixtures.ChatTool) fixtures.ChatCompletionsRequest {
	return fixtures.ChatCompletionsRequest{
		Model: stickyToolSelectionModel,
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: prompt},
		},
		Tools:      tools,
		ToolChoice: json.RawMessage(`"auto"`),
	}
}

func runStickyTurn(
	ctx context.Context,
	sessions *stickySessionPair,
	sessionID string,
	body any,
	headers map[string]string,
	label string,
) (stickyToolSnapshot, error) {
	response, err := sendStickyChatRequest(ctx, sessions.gateway, body, headers)
	if err != nil {
		return stickyToolSnapshot{}, fmt.Errorf("%s: %w", label, err)
	}
	if err := assertStickyResponse(response); err != nil {
		return stickyToolSnapshot{}, fmt.Errorf("%s: %w", label, err)
	}
	providerBody, err := lastProviderSimulatorRequest(ctx, sessions.backend, sessionID)
	if err != nil {
		return stickyToolSnapshot{}, fmt.Errorf("%s provider request: %w", label, err)
	}
	snapshot, err := decodeStickyProviderTools(providerBody)
	if err != nil {
		return stickyToolSnapshot{}, fmt.Errorf("%s provider request: %w", label, err)
	}
	if err := assertStickyToolBound(snapshot, label); err != nil {
		return stickyToolSnapshot{}, err
	}
	return snapshot, nil
}

func stickyRequestHeaders(sessionID string, trusted bool) map[string]string {
	headers := map[string]string{
		"x-session-id":          sessionID,
		"x-vsr-test-session-id": sessionID,
		"x-vsr-debug":           "true",
	}
	if trusted {
		headers["x-authz-user-id"] = "sticky-e2e-user"
	}
	return headers
}

func stickyContractTools(revision int, changedName string) []fixtures.ChatTool {
	weatherDescription := "Get current weather information for a location"
	weatherParameters := json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`)
	calculateDescription := "Perform mathematical calculations"
	calculateParameters := json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`)
	searchDescription := "Search web resources for recent information"
	searchParameters := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`)
	if revision > 1 {
		switch changedName {
		case "get_weather":
			weatherDescription += fmt.Sprintf(" (schema revision %d)", revision)
			weatherParameters = json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"},"units":{"type":"string"}},"required":["location"]}`)
		case "calculate":
			calculateDescription += fmt.Sprintf(" (schema revision %d)", revision)
			calculateParameters = json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"},"precision":{"type":"integer"}},"required":["expression"]}`)
		case "search_web":
			searchDescription += fmt.Sprintf(" (schema revision %d)", revision)
			searchParameters = json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"freshness":{"type":"string"}},"required":["query"]}`)
		}
	}
	return []fixtures.ChatTool{
		{Type: "function", Function: fixtures.ChatToolFunc{Name: "get_weather", Description: weatherDescription, Parameters: weatherParameters}},
		{Type: "function", Function: fixtures.ChatToolFunc{Name: "calculate", Description: calculateDescription, Parameters: calculateParameters}},
		{Type: "function", Function: fixtures.ChatToolFunc{Name: "search_web", Description: searchDescription, Parameters: searchParameters}},
	}
}

func stickyInvalidationPrompt(toolName string) string {
	switch toolName {
	case "calculate":
		return "__STICKY_TOOL_SELECTION__ Calculate 23 times 31."
	case "search_web":
		return "__STICKY_TOOL_SELECTION__ Search recent weather reports."
	default:
		return "__STICKY_TOOL_SELECTION__ Find the current weather for Seattle."
	}
}

func sendStickyChatRequest(
	ctx context.Context,
	session *fixtures.ServiceSession,
	body any,
	headers map[string]string,
) (*stickyHTTPResponse, error) {
	encoded, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, session.BaseURL()+"/v1/chat/completions", bytes.NewReader(encoded))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	resp, err := session.HTTPClient(60 * time.Second).Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()
	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	return &stickyHTTPResponse{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header.Clone(),
		Body:       responseBody,
	}, nil
}

func assertStickyResponse(response *stickyHTTPResponse) error {
	if response == nil {
		return fmt.Errorf("empty response")
	}
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", response.StatusCode, truncateString(string(response.Body), 600))
	}
	if decision := response.Headers.Get("x-vsr-selected-decision"); decision != stickyToolSelectionDecision {
		return fmt.Errorf("x-vsr-selected-decision = %q, want %q", decision, stickyToolSelectionDecision)
	}
	return nil
}

func decodeStickyProviderTools(body []byte) (stickyToolSnapshot, error) {
	var debug stickyProviderRequest
	if err := json.Unmarshal(body, &debug); err != nil {
		return stickyToolSnapshot{}, fmt.Errorf("decode provider request: %w", err)
	}
	if len(debug.Body.Tools) == 0 {
		return stickyToolSnapshot{}, fmt.Errorf("provider request contained no tools: %s", truncateString(string(body), 700))
	}
	snapshot := stickyToolSnapshot{
		Tools: make([]json.RawMessage, len(debug.Body.Tools)),
		Names: make([]string, len(debug.Body.Tools)),
	}
	for i, raw := range debug.Body.Tools {
		name, err := stickyProviderToolName(raw)
		if err != nil {
			return stickyToolSnapshot{}, fmt.Errorf("tool %d: %w", i, err)
		}
		snapshot.Tools[i] = append(json.RawMessage(nil), raw...)
		snapshot.Names[i] = name
	}
	return snapshot, nil
}

func stickyProviderToolName(raw json.RawMessage) (string, error) {
	var tool struct {
		Name     string `json:"name"`
		Function *struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &tool); err != nil {
		return "", fmt.Errorf("decode tool: %w", err)
	}
	if tool.Function != nil && tool.Function.Name != "" {
		return tool.Function.Name, nil
	}
	if tool.Name != "" {
		return tool.Name, nil
	}
	return "", fmt.Errorf("provider tool has no function name")
}

func assertStickyToolBound(snapshot stickyToolSnapshot, turn string) error {
	if len(snapshot.Tools) > stickyToolSelectionMaxTools {
		return fmt.Errorf("%s forwarded %d tools, max is %d", turn, len(snapshot.Tools), stickyToolSelectionMaxTools)
	}
	return nil
}

func assertStickyReuse(first, second stickyToolSnapshot) error {
	if len(first.Tools) == 0 || len(second.Tools) == 0 {
		return fmt.Errorf("trusted turns must both forward at least one tool")
	}
	if len(second.Tools) < len(first.Tools) {
		return fmt.Errorf("trusted turn dropped retained tools: first=%d second=%d", len(first.Tools), len(second.Tools))
	}
	if len(second.Tools) > len(first.Tools)+1 {
		return fmt.Errorf("trusted turn exceeded one-tool growth allowance: first=%d second=%d", len(first.Tools), len(second.Tools))
	}
	if len(second.Tools) != len(first.Tools)+1 {
		return fmt.Errorf("trusted turn did not append one newly relevant tool: first=%d second=%d", len(first.Tools), len(second.Tools))
	}
	for i := range first.Tools {
		if first.Names[i] != second.Names[i] {
			return fmt.Errorf("retained provider prefix changed order at index %d: first=%q second=%q", i, first.Names[i], second.Names[i])
		}
		if !bytes.Equal(first.Tools[i], second.Tools[i]) {
			return fmt.Errorf("retained provider definition changed bytes for %q", first.Names[i])
		}
	}
	return nil
}

func containsStickyTool(snapshot stickyToolSnapshot, name string) bool {
	for _, candidate := range snapshot.Names {
		if candidate == name {
			return true
		}
	}
	return false
}

func toolDefinitionByName(snapshot stickyToolSnapshot, name string) json.RawMessage {
	for i, candidate := range snapshot.Names {
		if candidate == name {
			return snapshot.Tools[i]
		}
	}
	return nil
}

func assertStickySnapshotsEqual(left, right stickyToolSnapshot) error {
	if len(left.Tools) != len(right.Tools) {
		return fmt.Errorf("tool counts differ: %d vs %d", len(left.Tools), len(right.Tools))
	}
	for i := range left.Tools {
		if !bytes.Equal(left.Tools[i], right.Tools[i]) {
			return fmt.Errorf("tool definition at index %d differs", i)
		}
	}
	return nil
}

func assertStickySnapshotsDifferent(left, right stickyToolSnapshot) error {
	if err := assertStickySnapshotsEqual(left, right); err == nil {
		return fmt.Errorf("tool snapshots are identical: %v", left.Names)
	}
	return nil
}

func assertStickyInvalidation(previous, updated, baseline stickyToolSnapshot, changedName string) error {
	oldDefinition := toolDefinitionByName(previous, changedName)
	if oldDefinition == nil {
		return fmt.Errorf("schema invalidation precondition omitted retained tool %q", changedName)
	}
	if err := assertStickySnapshotsEqual(updated, baseline); err != nil {
		return fmt.Errorf("schema invalidation did not fall back to stateless selection: %w", err)
	}
	if err := assertStickySnapshotsDifferent(previous, updated); err != nil {
		return fmt.Errorf("schema invalidation reused the retained tool set: %w", err)
	}
	if newDefinition := toolDefinitionByName(updated, changedName); newDefinition != nil && bytes.Equal(oldDefinition, newDefinition) {
		return fmt.Errorf("schema invalidation reused stale provider definition for %q", changedName)
	}
	return nil
}

func maxStickyTools(snapshots ...stickyToolSnapshot) int {
	maximum := 0
	for _, snapshot := range snapshots {
		if len(snapshot.Tools) > maximum {
			maximum = len(snapshot.Tools)
		}
	}
	return maximum
}
