package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// A coding agent sends its whole tool loop on every call: the tool calls it
// made, with their arguments, and the tools' output. Personal data usually
// arrives there, in a file or ticket a tool read, not in the user turn the
// cross-protocol case covers. These cases read what provider-mocker recorded
// at /debug/last-request, the dispatched Chat body itself, scoped by the
// session header, so tool messages and tool call arguments are visible.
const (
	maskingAgentSessionHeader = "x-vsr-test-session-id"
	maskingAgentTicketEmail   = "alice@corp.example"
	maskingAgentCopyEmail     = "bob@corp.example"
	maskingAgentTicketCall    = "call_ticket"
	maskingAgentSendCall      = "call_send"
)

var maskingEmailPlaceholder = regexp.MustCompile(`\[EMAIL_ADDRESS_\d+\]`)

func init() {
	pkgtestcases.Register("masking-agent-tool-loop", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "A coding-agent tool loop through Chat (buffered and streamed), Responses and Anthropic reaches the backend with personal data masked in tool output and tool call arguments, the same value under the same placeholder, and every call ID paired",
		Tags:        []string{"masking", "pii", "cross-protocol", "agent", "functional"},
		Fn:          testMaskingAgentToolLoop,
	})
	pkgtestcases.Register("masking-scan-error-fails-closed", pkgtestcases.TestCase{
		Priority:    pkgtestcases.P0,
		Description: "A PII deployment error on a tool result behind a clean user turn, on every protocol, or on a Chat user turn returns 503 and dispatches nothing",
		Tags:        []string{"masking", "pii", "agent", "fail-closed"},
		Fn:          testMaskingScanErrorFailsClosed,
	})
}

// maskingAgentLoop is one agent transcript: the agent fetched a ticket, got it
// back as JSON, sent a reply to its sender, and got the send tool's output.
type maskingAgentLoop struct {
	ticketOutput string
	sendArgs     string
	sendOutput   string
}

func defaultMaskingAgentLoop() maskingAgentLoop {
	return maskingAgentLoop{
		ticketOutput: fmt.Sprintf(`{"from":"%s EMAIL_ADDRESS %s","subject":"Refund"}`, maskingSpanMarker, maskingAgentTicketEmail),
		sendArgs:     fmt.Sprintf(`{"to":"%s EMAIL_ADDRESS %s","body":"Refund approved"}`, maskingSpanMarker, maskingAgentTicketEmail),
		sendOutput:   fmt.Sprintf("Sent. A copy went to %s EMAIL_ADDRESS %s", maskingSpanMarker, maskingAgentCopyEmail),
	}
}

func (l maskingAgentLoop) wireCases() []maskingWireCase {
	const ask = "Answer the newest support ticket."
	chatTool := func(name string) map[string]interface{} {
		return map[string]interface{}{"type": "function", "function": map[string]interface{}{
			"name": name, "parameters": map[string]interface{}{"type": "object"},
		}}
	}
	chatCall := func(id, name, args string) map[string]interface{} {
		return map[string]interface{}{"role": "assistant", "tool_calls": []map[string]interface{}{{
			"id": id, "type": "function", "function": map[string]interface{}{"name": name, "arguments": args},
		}}}
	}
	responsesTool := func(name string) map[string]interface{} {
		return map[string]interface{}{"type": "function", "name": name, "parameters": map[string]interface{}{"type": "object"}}
	}
	anthropicTool := func(name string) map[string]interface{} {
		return map[string]interface{}{"name": name, "input_schema": map[string]interface{}{"type": "object"}}
	}
	anthropicCall := func(id, name, args string) map[string]interface{} {
		var input map[string]interface{}
		_ = json.Unmarshal([]byte(args), &input)
		return map[string]interface{}{"role": "assistant", "content": []map[string]interface{}{{
			"type": "tool_use", "id": id, "name": name, "input": input,
		}}}
	}
	anthropicResult := func(id, output string) map[string]interface{} {
		return map[string]interface{}{"role": "user", "content": []map[string]interface{}{{
			"type": "tool_result", "tool_use_id": id, "content": output,
		}}}
	}
	return []maskingWireCase{
		{
			name: "chat completions",
			path: "/v1/chat/completions",
			body: map[string]interface{}{
				"model": maskingEntrypoint,
				"tools": []map[string]interface{}{chatTool("fetch_ticket"), chatTool("send_email")},
				"messages": []map[string]interface{}{
					{"role": "user", "content": ask},
					chatCall(maskingAgentTicketCall, "fetch_ticket", `{"id":"T-1"}`),
					{"role": "tool", "tool_call_id": maskingAgentTicketCall, "content": l.ticketOutput},
					chatCall(maskingAgentSendCall, "send_email", l.sendArgs),
					{"role": "tool", "tool_call_id": maskingAgentSendCall, "content": l.sendOutput},
				},
			},
		},
		{
			name: "responses",
			path: "/v1/responses",
			body: map[string]interface{}{
				"model": maskingEntrypoint,
				"store": false,
				"tools": []map[string]interface{}{responsesTool("fetch_ticket"), responsesTool("send_email")},
				"input": []map[string]interface{}{
					{"role": "user", "content": []map[string]interface{}{{"type": "input_text", "text": ask}}},
					{"type": "function_call", "call_id": maskingAgentTicketCall, "name": "fetch_ticket", "arguments": `{"id":"T-1"}`},
					{"type": "function_call_output", "call_id": maskingAgentTicketCall, "output": l.ticketOutput},
					{"type": "function_call", "call_id": maskingAgentSendCall, "name": "send_email", "arguments": l.sendArgs},
					{"type": "function_call_output", "call_id": maskingAgentSendCall, "output": l.sendOutput},
				},
			},
		},
		{
			name: "anthropic messages",
			path: "/v1/messages",
			body: map[string]interface{}{
				"model":      maskingEntrypoint,
				"max_tokens": 64,
				"tools":      []map[string]interface{}{anthropicTool("fetch_ticket"), anthropicTool("send_email")},
				"messages": []map[string]interface{}{
					{"role": "user", "content": ask},
					anthropicCall(maskingAgentTicketCall, "fetch_ticket", `{"id":"T-1"}`),
					anthropicResult(maskingAgentTicketCall, l.ticketOutput),
					anthropicCall(maskingAgentSendCall, "send_email", l.sendArgs),
					anthropicResult(maskingAgentSendCall, l.sendOutput),
				},
			},
		},
	}
}

// maskingDispatchedChat is the part of the dispatched Chat body these cases
// read. Content stays raw because a tool message may carry a string or parts.
type maskingDispatchedChat struct {
	Messages []struct {
		Role       string          `json:"role"`
		Content    json.RawMessage `json:"content"`
		ToolCallID string          `json:"tool_call_id"`
		ToolCalls  []struct {
			ID       string `json:"id"`
			Function struct {
				Arguments string `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls"`
	} `json:"messages"`
}

// testMaskingAgentToolLoop checks what a backend receives for one agent loop.
// The ticket's sender is in a JSON tool result and in the send call's
// arguments, so both must carry one placeholder, or the agent's next call
// names a different address than the one it read. The copy address appears
// only in the plain-text output and must get its own placeholder.
func testMaskingAgentToolLoop(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, backend, err := openMaskingAgentSessions(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	defer backend.Close()

	runID := time.Now().UnixNano()
	received := map[string]string{}
	cases := defaultMaskingAgentLoop().wireCases()
	// Agents stream; the streamed request must be masked the same way.
	streamed := map[string]interface{}{"stream": true}
	for key, value := range cases[0].body {
		streamed[key] = value
	}
	cases = append(cases, maskingWireCase{name: "chat completions stream", path: cases[0].path, body: streamed})
	for _, tc := range cases {
		sessionID := fmt.Sprintf("masking-agent-loop-%d-%s", runID, strings.ReplaceAll(tc.name, " ", "-"))
		resp, err := runMaskingRawRequest(ctx, session, tc.path, tc.body, http.Header{maskingAgentSessionHeader: {sessionID}})
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("%s returned HTTP %d: %s", tc.name, resp.StatusCode, truncateString(string(resp.Body), 500))
		}
		if decision := resp.Headers.Get("x-vsr-selected-decision"); decision != maskingDecisionName {
			return fmt.Errorf("%s: selected decision = %q, want %q", tc.name, decision, maskingDecisionName)
		}
		dispatched, found, err := lookupMaskingDispatch(ctx, backend, sessionID)
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if !found {
			return fmt.Errorf("%s: the backend recorded no request for session %s", tc.name, sessionID)
		}
		if err := checkMaskedAgentLoop(dispatched); err != nil {
			return fmt.Errorf("%s: %w (dispatched: %s)", tc.name, err, truncateString(string(dispatched), 1500))
		}
		received[tc.name] = string(dispatched)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"protocols": "chat,chat-stream,responses,anthropic", "dispatched": received})
	}
	return nil
}

func checkMaskedAgentLoop(dispatched []byte) error {
	for _, raw := range []string{maskingAgentTicketEmail, maskingAgentCopyEmail} {
		if strings.Contains(string(dispatched), raw) {
			return fmt.Errorf("raw value %q reached the backend", raw)
		}
	}
	var chat maskingDispatchedChat
	if err := json.Unmarshal(dispatched, &chat); err != nil {
		return fmt.Errorf("dispatched body is not a Chat request: %w", err)
	}
	// The loop must arrive as sent: each call, then its result, in order. A
	// map would hide a duplicated, dropped or reordered entry.
	callArgs := map[string]string{}
	toolOutput := map[string]string{}
	var sequence []string
	for _, message := range chat.Messages {
		for _, call := range message.ToolCalls {
			callArgs[call.ID] = call.Function.Arguments
			sequence = append(sequence, "call:"+call.ID)
		}
		if message.Role == "tool" {
			text, err := maskingMessageText(message.Content)
			if err != nil {
				return fmt.Errorf("tool message %q: %w", message.ToolCallID, err)
			}
			toolOutput[message.ToolCallID] = text
			sequence = append(sequence, "result:"+message.ToolCallID)
		}
	}
	want := []string{
		"call:" + maskingAgentTicketCall, "result:" + maskingAgentTicketCall,
		"call:" + maskingAgentSendCall, "result:" + maskingAgentSendCall,
	}
	if !slices.Equal(sequence, want) {
		return fmt.Errorf("tool loop reached the backend as %v, want %v", sequence, want)
	}

	var ticket struct {
		From    *string `json:"from"`
		Subject *string `json:"subject"`
	}
	if err := json.Unmarshal([]byte(toolOutput[maskingAgentTicketCall]), &ticket); err != nil {
		return fmt.Errorf("JSON tool result is no longer JSON: %w", err)
	}
	if ticket.From == nil || ticket.Subject == nil || *ticket.Subject != "Refund" {
		return fmt.Errorf("JSON tool result lost a member: %q", toolOutput[maskingAgentTicketCall])
	}
	var send struct {
		To   *string `json:"to"`
		Body *string `json:"body"`
	}
	if err := json.Unmarshal([]byte(callArgs[maskingAgentSendCall]), &send); err != nil {
		return fmt.Errorf("tool call arguments are no longer JSON: %w", err)
	}
	if send.To == nil || send.Body == nil || *send.Body != "Refund approved" {
		return fmt.Errorf("tool call arguments lost a member: %q", callArgs[maskingAgentSendCall])
	}

	sender := maskingEmailPlaceholder.FindString(*ticket.From)
	if sender == "" {
		return fmt.Errorf("JSON tool result carries no placeholder: %q", *ticket.From)
	}
	if got := maskingEmailPlaceholder.FindString(*send.To); got != sender {
		return fmt.Errorf("the same address is %s in the tool result and %q in the tool call arguments", sender, got)
	}
	copied := maskingEmailPlaceholder.FindString(toolOutput[maskingAgentSendCall])
	if copied == "" || copied == sender {
		return fmt.Errorf("plain-text tool result placeholder = %q, want one other than %s: %q",
			copied, sender, toolOutput[maskingAgentSendCall])
	}
	return nil
}

// testMaskingScanErrorFailsClosed makes one text fail its scan: the send
// tool's output behind a clean user turn, on each protocol, and then a Chat
// user turn. The request is refused and the backend never sees it. A clean
// control with the same session handling runs first and must be dispatched,
// so a missing record means no dispatch rather than a dropped session header.
func testMaskingScanErrorFailsClosed(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, backend, err := openMaskingAgentSessions(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	defer backend.Close()

	runID := time.Now().UnixNano()
	failing := defaultMaskingAgentLoop()
	failing.sendOutput = fmt.Sprintf("Sent. %s A copy went to %s", maskingErrorMarker, maskingAgentCopyEmail)
	userTurn := func(text string) maskingWireCase {
		return maskingWireCase{name: "chat user turn", path: "/v1/chat/completions", body: map[string]interface{}{
			"model":    maskingEntrypoint,
			"messages": []map[string]interface{}{{"role": "user", "content": text}},
		}}
	}
	cases := append(failing.wireCases(), userTurn(fmt.Sprintf("%s contact %s please", maskingErrorMarker, maskingAgentCopyEmail)))
	controls := append(defaultMaskingAgentLoop().wireCases(), userTurn("contact the billing team please"))
	for i, tc := range cases {
		controlID := fmt.Sprintf("masking-agent-control-%d-%d", runID, i)
		resp, err := runMaskingRawRequest(ctx, session, tc.path, controls[i].body, http.Header{maskingAgentSessionHeader: {controlID}})
		if err != nil {
			return fmt.Errorf("%s control: %w", tc.name, err)
		}
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("%s control returned HTTP %d: %s", tc.name, resp.StatusCode, truncateString(string(resp.Body), 300))
		}
		_, found, err := lookupMaskingDispatch(ctx, backend, controlID)
		if err != nil {
			return fmt.Errorf("%s control: %w", tc.name, err)
		}
		if !found {
			return fmt.Errorf("%s control: the backend recorded no request, so no-dispatch cannot be observed", tc.name)
		}

		sessionID := fmt.Sprintf("masking-agent-scan-error-%d-%d", runID, i)
		resp, err = runMaskingRawRequest(ctx, session, tc.path, tc.body, http.Header{maskingAgentSessionHeader: {sessionID}})
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if resp.StatusCode != http.StatusServiceUnavailable {
			return fmt.Errorf("%s: status = %d, want 503: %s", tc.name, resp.StatusCode, truncateString(string(resp.Body), 300))
		}
		if strings.Contains(string(resp.Body), maskingAgentCopyEmail) {
			return fmt.Errorf("%s: the refusal echoes request content: %s", tc.name, resp.Body)
		}
		_, found, err = lookupMaskingDispatch(ctx, backend, sessionID)
		if err != nil {
			return fmt.Errorf("%s: %w", tc.name, err)
		}
		if found {
			return fmt.Errorf("%s: the refused request still reached the backend", tc.name)
		}
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"refused": len(cases), "status": http.StatusServiceUnavailable})
	}
	return nil
}

func openMaskingAgentSessions(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*fixtures.ServiceSession, *fixtures.ServiceSession, error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return nil, nil, err
	}
	backend, err := fixtures.OpenServiceEndpointSession(ctx, client, opts, "default", "provider-mocker", "8000")
	if err != nil {
		session.Close()
		return nil, nil, fmt.Errorf("failed to reach provider-mocker: %w", err)
	}
	return session, backend, nil
}

// lookupMaskingDispatch returns the body provider-mocker recorded for a
// session, and false when the session never reached it.
func lookupMaskingDispatch(ctx context.Context, backend *fixtures.ServiceSession, sessionID string) ([]byte, bool, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, backend.URL("/debug/last-request"), nil)
	if err != nil {
		return nil, false, err
	}
	req.Header.Set(maskingAgentSessionHeader, sessionID)
	resp, err := backend.HTTPClient(30 * time.Second).Do(req)
	if err != nil {
		return nil, false, fmt.Errorf("provider-mocker observation failed: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, false, err
	}
	switch resp.StatusCode {
	case http.StatusNotFound:
		return nil, false, nil
	case http.StatusOK:
		var observation struct {
			Body json.RawMessage `json:"body"`
		}
		if err := json.Unmarshal(body, &observation); err != nil {
			return nil, false, fmt.Errorf("provider-mocker observation is not JSON: %w", err)
		}
		return observation.Body, true, nil
	default:
		return nil, false, fmt.Errorf("provider-mocker observation status %d: %s", resp.StatusCode, truncateString(string(body), 300))
	}
}

// maskingMessageText reads a Chat message's content as a string or as the
// concatenated text of its parts.
func maskingMessageText(content json.RawMessage) (string, error) {
	var text string
	if err := json.Unmarshal(content, &text); err == nil {
		return text, nil
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(content, &parts); err != nil {
		return "", fmt.Errorf("content is neither a string nor parts: %s", content)
	}
	var joined strings.Builder
	for _, part := range parts {
		joined.WriteString(part.Text)
	}
	return joined.String(), nil
}
