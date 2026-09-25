package testcases

import (
	"bytes"
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"path"
	"sort"
	"strings"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

const (
	agentClientToolMarker  = "__mock_tool_call__"
	agentClientErrorMarker = "__mock_provider_error__"
	agentClientResultText  = "tool result accepted"
)

//go:embed testdata/agent_clients/*.json
var agentClientCaptureFiles embed.FS

// agentClientCapture is one sanitized client tool loop. The first turn asks for
// a tool call; the second carries the client's tool call and its result. Backends
// maps each backend format the capture must pass through to the request fields
// that must reach that provider.
type agentClientCapture struct {
	Name     string              `json:"-"`
	Client   string              `json:"client"`
	Version  string              `json:"version"`
	Path     string              `json:"path"`
	Headers  map[string]string   `json:"headers"`
	Backends map[string][]string `json:"backends"`
	Turns    []json.RawMessage   `json:"turns"`
}

type agentClientProtocol struct {
	userText       func(turn map[string]any) (string, func(string), error)
	toolRefs       agentClientToolRefs
	decodeToolCall func(body []byte, stream bool, anthropicBackend bool) (responsesFunctionCall, error)
	assertText     func(body []byte, stream bool) error
	assertUsage    func(body []byte, stream bool, turn map[string]any) error
}

var agentClientProtocols = map[string]agentClientProtocol{
	"/v1/chat/completions": {
		userText:       lastMessageUserText,
		toolRefs:       agentChatToolRefs,
		decodeToolCall: decodeAgentChatToolCall,
		assertText:     assertAgentChatText,
		assertUsage:    assertAgentChatUsage,
	},
	"/v1/messages": {
		userText:       lastMessageUserText,
		toolRefs:       agentMessagesToolRefs,
		decodeToolCall: decodeAgentMessagesToolCall,
		assertText:     assertAgentMessagesText,
		assertUsage:    assertAgentMessagesUsage,
	},
	"/v1/responses": {
		userText:       lastResponsesUserText,
		toolRefs:       agentResponsesToolRefs,
		decodeToolCall: decodeAgentResponsesToolCall,
		assertText:     assertAgentResponsesText,
		assertUsage:    assertAgentResponsesUsage,
	},
}

// agentClientBackendToolRefs reads a provider request, which arrives in the
// backend's format rather than the client's.
var agentClientBackendToolRefs = map[string]agentClientToolRefs{
	"openai.chat.v1":        agentChatToolRefs,
	"openai.responses.v1":   agentResponsesToolRefs,
	"anthropic.messages.v1": agentMessagesToolRefs,
}

func init() {
	pkgtestcases.Register("protocol-codec-chat-backend-agent-client-replay", pkgtestcases.TestCase{
		Description: "Captured agent-client tool loops, errors and usage survive a native Chat Completions backend",
		Tags:        []string{"protocol-codec", "response-api", "agents", "tools", "streaming", "errors"},
		Fn:          agentClientReplayTest(chatBackendModel, "openai.chat.v1"),
	})
	pkgtestcases.Register("protocol-codec-responses-backend-agent-client-replay", pkgtestcases.TestCase{
		Description: "Captured agent-client tool loops, errors and usage survive a native Responses backend",
		Tags:        []string{"protocol-codec", "response-api", "agents", "tools", "streaming", "errors"},
		Fn:          agentClientReplayTest(nativeResponsesBackendModel, "openai.responses.v1"),
	})
	pkgtestcases.Register("protocol-codec-anthropic-backend-agent-client-replay", pkgtestcases.TestCase{
		Description: "Captured agent-client tool loops, errors and usage survive a native Anthropic Messages backend",
		Tags:        []string{"protocol-codec", "anthropic", "agents", "tools", "streaming", "errors"},
		Fn:          agentClientReplayTest("MoM", "anthropic.messages.v1"),
	})
}

func agentClientReplayTest(
	model string,
	backendFormat string,
) func(context.Context, *kubernetes.Clientset, pkgtestcases.TestCaseOptions) error {
	return func(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
		captures, err := loadAgentClientCaptures()
		if err != nil {
			return err
		}
		session, err := fixtures.OpenServiceSession(ctx, client, opts)
		if err != nil {
			return err
		}
		defer session.Close()
		provider, err := openProtocolCodecProviderSession(ctx, client, opts, backendFormat)
		if err != nil {
			return err
		}
		defer provider.Close()

		results, err := replayAgentClientCaptures(ctx, session, provider, captures, model, backendFormat)
		if opts.SetDetails != nil {
			opts.SetDetails(map[string]interface{}{"backend_format": backendFormat, "captures": results})
		}
		return err
	}
}

func loadAgentClientCaptures() ([]agentClientCapture, error) {
	names, err := fs.Glob(agentClientCaptureFiles, "testdata/agent_clients/*.json")
	if err != nil {
		return nil, err
	}
	sort.Strings(names)
	captures := make([]agentClientCapture, 0, len(names))
	for _, name := range names {
		data, readErr := agentClientCaptureFiles.ReadFile(name)
		if readErr != nil {
			return nil, readErr
		}
		capture, decodeErr := decodeAgentClientCapture(path.Base(name), data)
		if decodeErr != nil {
			return nil, decodeErr
		}
		captures = append(captures, capture)
	}
	return captures, nil
}

func decodeAgentClientCapture(name string, data []byte) (agentClientCapture, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var capture agentClientCapture
	if err := decoder.Decode(&capture); err != nil {
		return capture, fmt.Errorf("%s: %w", name, err)
	}
	capture.Name = strings.TrimSuffix(name, ".json")
	if capture.Client == "" || capture.Version == "" {
		return capture, fmt.Errorf("%s: client and version are required", name)
	}
	if _, ok := agentClientProtocols[capture.Path]; !ok {
		return capture, fmt.Errorf("%s: unsupported client path %q", name, capture.Path)
	}
	if len(capture.Turns) != 2 {
		return capture, fmt.Errorf("%s: want a tool-call turn and a tool-result turn, got %d turns", name, len(capture.Turns))
	}
	if len(capture.Backends) == 0 {
		return capture, fmt.Errorf("%s: declares no backend format", name)
	}
	for backendFormat := range capture.Backends {
		if _, ok := agentClientBackendToolRefs[backendFormat]; !ok {
			return capture, fmt.Errorf("%s: unknown backend format %q", name, backendFormat)
		}
	}
	return capture, nil
}

func replayAgentClientCaptures(
	ctx context.Context,
	session *fixtures.ServiceSession,
	provider *fixtures.ServiceSession,
	captures []agentClientCapture,
	model string,
	backendFormat string,
) (map[string]string, error) {
	results := make(map[string]string)
	var failures []error
	for _, capture := range captures {
		providerFields, ok := capture.Backends[backendFormat]
		if !ok {
			continue
		}
		replay := agentClientReplay{
			session:        session,
			provider:       provider,
			capture:        capture,
			protocol:       agentClientProtocols[capture.Path],
			model:          model,
			backendFormat:  backendFormat,
			providerFields: providerFields,
		}
		if err := replay.run(ctx); err != nil {
			results[capture.Name] = "failed"
			failures = append(failures, fmt.Errorf("%s: %w", capture.Name, err))
			continue
		}
		results[capture.Name] = "passed"
	}
	if len(results) == 0 {
		return results, fmt.Errorf("no agent-client capture declares backend format %s", backendFormat)
	}
	return results, errors.Join(failures...)
}

type agentClientReplay struct {
	session        *fixtures.ServiceSession
	provider       *fixtures.ServiceSession
	capture        agentClientCapture
	protocol       agentClientProtocol
	model          string
	backendFormat  string
	providerFields []string
}

// run replays both turns streamed and buffered, then the error turn in each mode.
func (replay agentClientReplay) run(ctx context.Context) error {
	var streamedCall responsesFunctionCall
	for _, stream := range []bool{true, false} {
		call, err := replay.toolCallTurn(ctx, stream)
		if err != nil {
			return err
		}
		if stream {
			streamedCall = call
		} else if call != streamedCall {
			return fmt.Errorf("buffered and streamed tool calls differ: buffered=%+v streamed=%+v", call, streamedCall)
		}
		if err := replay.toolResultTurn(ctx, stream, call); err != nil {
			return err
		}
		if err := replay.providerErrorTurn(ctx, stream); err != nil {
			return err
		}
	}
	return nil
}

func (replay agentClientReplay) toolCallTurn(ctx context.Context, stream bool) (responsesFunctionCall, error) {
	turn, err := agentClientTurn(replay.protocol, replay.capture.Turns[0], replay.model, stream, agentClientToolMarker)
	if err != nil {
		return responsesFunctionCall{}, err
	}
	body, sessionID, err := replay.send(ctx, "tool-call", turn, stream, http.StatusOK)
	if err != nil {
		return responsesFunctionCall{}, err
	}
	call, err := replay.protocol.decodeToolCall(body, stream, replay.backendFormat == "anthropic.messages.v1")
	if err == nil {
		err = replay.protocol.assertUsage(body, stream, turn)
	}
	if err == nil {
		err = verifyAgentClientProviderFields(ctx, replay.provider, sessionID, replay.providerFields)
	}
	if err != nil {
		return responsesFunctionCall{}, fmt.Errorf("%s: %w", sessionID, err)
	}
	return call, nil
}

func (replay agentClientReplay) toolResultTurn(ctx context.Context, stream bool, call responsesFunctionCall) error {
	turn, err := replay.resultTurn(stream, call)
	if err != nil {
		return err
	}
	body, sessionID, err := replay.send(ctx, "tool-result", turn, stream, http.StatusOK)
	if err != nil {
		return err
	}
	if err := replay.protocol.assertText(body, stream); err != nil {
		return fmt.Errorf("%s: %w", sessionID, err)
	}
	if err := replay.protocol.assertUsage(body, stream, turn); err != nil {
		return fmt.Errorf("%s: %w", sessionID, err)
	}
	if err := verifyAgentClientToolLink(ctx, replay.provider, sessionID, replay.backendFormat, call); err != nil {
		return fmt.Errorf("%s: %w", sessionID, err)
	}
	return nil
}

// resultTurn is the captured follow-up, answering the call decoded on the
// tool-call turn as a client would rather than the call recorded in the capture.
func (replay agentClientReplay) resultTurn(stream bool, call responsesFunctionCall) (map[string]any, error) {
	turn, err := agentClientTurn(replay.protocol, replay.capture.Turns[1], replay.model, stream, "")
	if err != nil {
		return nil, err
	}
	if err := linkAgentClientToolCall(replay.protocol.toolRefs, turn, call); err != nil {
		return nil, err
	}
	return turn, nil
}

func (replay agentClientReplay) providerErrorTurn(ctx context.Context, stream bool) error {
	turn, err := agentClientTurn(replay.protocol, replay.capture.Turns[0], replay.model, stream, agentClientErrorMarker)
	if err != nil {
		return err
	}
	body, sessionID, err := replay.send(ctx, "provider-error", turn, stream, http.StatusTooManyRequests)
	if err != nil {
		return err
	}
	expectedCode := "rate_limit_exceeded"
	if replay.backendFormat == "anthropic.messages.v1" {
		expectedCode = "rate_limit_error"
	}
	if err := assertProtocolMatrixRateLimit(body, replay.capture.Path == "/v1/messages", expectedCode); err != nil {
		return fmt.Errorf("%s: %w", sessionID, err)
	}
	return nil
}

func (replay agentClientReplay) send(
	ctx context.Context,
	step string,
	turn map[string]any,
	stream bool,
	wantStatus int,
) ([]byte, string, error) {
	mode := "buffered"
	if stream {
		mode = "streaming"
	}
	sessionID := strings.Join([]string{"agent-client", replay.capture.Name, replay.backendFormat, step, mode}, "-")
	headers := map[string]string{"x-vsr-test-session-id": sessionID}
	for key, value := range replay.capture.Headers {
		headers[key] = value
	}
	result, err := sendProtocolMatrixRaw(ctx, replay.session, replay.capture.Path, turn, stream, headers)
	if err != nil {
		return nil, sessionID, fmt.Errorf("%s: %w", sessionID, err)
	}
	if result.StatusCode != wantStatus {
		return nil, sessionID, fmt.Errorf("%s: HTTP %d, want %d: %s",
			sessionID, result.StatusCode, wantStatus, truncateString(string(result.Body), 800))
	}
	return result.Body, sessionID, nil
}

// agentClientTurn replays a captured turn against the profile model. Buffered
// replays drop stream_options, which Chat Completions accepts only on streams.
func agentClientTurn(
	protocol agentClientProtocol,
	raw json.RawMessage,
	model string,
	stream bool,
	marker string,
) (map[string]any, error) {
	var turn map[string]any
	if err := json.Unmarshal(raw, &turn); err != nil {
		return nil, err
	}
	turn["model"] = model
	turn["stream"] = stream
	if !stream {
		delete(turn, "stream_options")
	}
	if marker == "" {
		return turn, nil
	}
	text, set, err := protocol.userText(turn)
	if err != nil {
		return nil, err
	}
	set(text + "\n" + marker)
	return turn, nil
}

func verifyAgentClientProviderFields(
	ctx context.Context,
	provider *fixtures.ServiceSession,
	sessionID string,
	fields []string,
) error {
	raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return err
	}
	var observation struct {
		Body map[string]json.RawMessage `json:"body"`
	}
	if err := json.Unmarshal(raw, &observation); err != nil {
		return fmt.Errorf("decode provider observation: %w", err)
	}
	if !strings.Contains(string(raw), agentClientToolMarker) {
		return fmt.Errorf("provider observation does not belong to this turn: %s", truncateString(string(raw), 600))
	}
	for _, field := range fields {
		if value, ok := observation.Body[field]; !ok || string(value) == "null" {
			return fmt.Errorf("provider request lost %q: %s", field, truncateString(string(raw), 800))
		}
	}
	return nil
}

func verifyAgentClientToolLink(
	ctx context.Context,
	provider *fixtures.ServiceSession,
	sessionID string,
	backendFormat string,
	call responsesFunctionCall,
) error {
	raw, err := lastProviderSimulatorRequest(ctx, provider, sessionID)
	if err != nil {
		return err
	}
	var observation struct {
		Body map[string]any `json:"body"`
	}
	if err := json.Unmarshal(raw, &observation); err != nil {
		return fmt.Errorf("decode provider observation: %w", err)
	}
	if err := requireAgentClientToolLink(agentClientBackendToolRefs[backendFormat], observation.Body, call); err != nil {
		return fmt.Errorf("provider request: %w", err)
	}
	return nil
}

// agentClientToolRef points at the object that holds a tool call ID under
// idKey. For a call, name is the object that holds the tool name.
type agentClientToolRef struct {
	fields map[string]any
	idKey  string
	name   map[string]any
}

type agentClientToolRefs func(body map[string]any) (calls, results []agentClientToolRef)

func linkAgentClientToolCall(refs agentClientToolRefs, turn map[string]any, call responsesFunctionCall) error {
	calls, results := refs(turn)
	if len(calls) != 1 || len(results) != 1 {
		return fmt.Errorf("follow-up has %d tool calls and %d results, want one of each", len(calls), len(results))
	}
	captured := calls[0].fields[calls[0].idKey]
	if results[0].fields[results[0].idKey] != captured {
		return fmt.Errorf("captured tool result does not answer the captured call %v", captured)
	}
	calls[0].fields[calls[0].idKey] = call.CallID
	calls[0].name["name"] = call.Name
	results[0].fields[results[0].idKey] = call.CallID
	return nil
}

func requireAgentClientToolLink(refs agentClientToolRefs, body map[string]any, call responsesFunctionCall) error {
	calls, results := refs(body)
	if len(calls) != 1 || len(results) != 1 {
		return fmt.Errorf("want one tool call and one result, got %d and %d", len(calls), len(results))
	}
	callID, name := calls[0].fields[calls[0].idKey], calls[0].name["name"]
	answered := results[0].fields[results[0].idKey]
	if callID != call.CallID || name != call.Name || answered != call.CallID {
		return fmt.Errorf("tool call %v %v is answered by %v; want %s %s from the tool-call turn",
			name, callID, answered, call.Name, call.CallID)
	}
	return nil
}

func agentChatToolRefs(body map[string]any) (calls, results []agentClientToolRef) {
	for _, message := range agentClientObjects(body["messages"]) {
		for _, toolCall := range agentClientObjects(message["tool_calls"]) {
			if function, ok := toolCall["function"].(map[string]any); ok {
				calls = append(calls, agentClientToolRef{fields: toolCall, idKey: "id", name: function})
			}
		}
		if message["role"] == "tool" {
			results = append(results, agentClientToolRef{fields: message, idKey: "tool_call_id"})
		}
	}
	return calls, results
}

func agentMessagesToolRefs(body map[string]any) (calls, results []agentClientToolRef) {
	for _, message := range agentClientObjects(body["messages"]) {
		for _, block := range agentClientObjects(message["content"]) {
			switch block["type"] {
			case "tool_use":
				calls = append(calls, agentClientToolRef{fields: block, idKey: "id", name: block})
			case "tool_result":
				results = append(results, agentClientToolRef{fields: block, idKey: "tool_use_id"})
			}
		}
	}
	return calls, results
}

func agentResponsesToolRefs(body map[string]any) (calls, results []agentClientToolRef) {
	for _, item := range agentClientObjects(body["input"]) {
		switch item["type"] {
		case "function_call":
			calls = append(calls, agentClientToolRef{fields: item, idKey: "call_id", name: item})
		case "function_call_output":
			results = append(results, agentClientToolRef{fields: item, idKey: "call_id"})
		}
	}
	return calls, results
}

func agentClientObjects(value any) []map[string]any {
	items, _ := value.([]any)
	objects := make([]map[string]any, 0, len(items))
	for _, item := range items {
		if object, ok := item.(map[string]any); ok {
			objects = append(objects, object)
		}
	}
	return objects
}

func lastMessageUserText(turn map[string]any) (string, func(string), error) {
	messages, _ := turn["messages"].([]any)
	return lastUserText(messages, "text")
}

func lastResponsesUserText(turn map[string]any) (string, func(string), error) {
	if input, ok := turn["input"].(string); ok {
		return input, func(text string) { turn["input"] = text }, nil
	}
	items, _ := turn["input"].([]any)
	return lastUserText(items, "input_text")
}

// lastUserText returns the text of the last user item, which is either a string
// or the last content part of partType.
func lastUserText(items []any, partType string) (string, func(string), error) {
	for index := len(items) - 1; index >= 0; index-- {
		item, ok := items[index].(map[string]any)
		if !ok || item["role"] != "user" {
			continue
		}
		if text, ok := item["content"].(string); ok {
			return text, func(value string) { item["content"] = value }, nil
		}
		parts, _ := item["content"].([]any)
		for partIndex := len(parts) - 1; partIndex >= 0; partIndex-- {
			part, ok := parts[partIndex].(map[string]any)
			if !ok || part["type"] != partType {
				continue
			}
			text, _ := part["text"].(string)
			return text, func(value string) { part["text"] = value }, nil
		}
	}
	return "", nil, fmt.Errorf("turn has no user %s content", partType)
}

func decodeAgentChatToolCall(body []byte, stream bool, _ bool) (responsesFunctionCall, error) {
	if stream {
		return decodeChatFunctionCallStream(body)
	}
	return decodeChatFunctionCall(body)
}

func decodeAgentMessagesToolCall(body []byte, stream bool, anthropicBackend bool) (responsesFunctionCall, error) {
	if stream {
		return decodeAnthropicToolUseStream(body, anthropicBackend)
	}
	return decodeAnthropicToolUse(body, anthropicBackend)
}

func decodeAgentResponsesToolCall(body []byte, stream bool, _ bool) (responsesFunctionCall, error) {
	if stream {
		return decodeResponsesFunctionCallStream(body)
	}
	return decodeResponsesFunctionCall(body)
}

func assertAgentChatText(body []byte, stream bool) error {
	if stream {
		return assertChatTextStream(body, agentClientResultText)
	}
	return assertChatText(body, agentClientResultText)
}

func assertAgentMessagesText(body []byte, stream bool) error {
	if stream {
		return assertAnthropicTextStream(body, agentClientResultText)
	}
	return assertAnthropicText(body, agentClientResultText)
}

func assertAgentResponsesText(body []byte, stream bool) error {
	if !stream {
		return assertResponsesText(body, agentClientResultText)
	}
	if err := validateResponseAPIStreamingSSEBody(string(body)); err != nil {
		return err
	}
	if !strings.Contains(string(body), agentClientResultText) {
		return fmt.Errorf("responses stream lost the tool-result answer: %s", truncateString(string(body), 1200))
	}
	return nil
}

func assertAgentChatUsage(body []byte, stream bool, turn map[string]any) error {
	if !stream {
		return requireUsageTokens(body, "usage", "prompt_tokens", "completion_tokens", "total_tokens")
	}
	options, _ := turn["stream_options"].(map[string]any)
	if options["include_usage"] != true {
		return nil
	}
	for _, data := range protocolSSEDataFrames(body) {
		if data != "[DONE]" && requireUsageTokens([]byte(data), "usage", "prompt_tokens", "completion_tokens", "total_tokens") == nil {
			return nil
		}
	}
	return fmt.Errorf("chat stream requested usage but carried none: %s", truncateString(string(body), 1200))
}

func assertAgentMessagesUsage(body []byte, stream bool, _ map[string]any) error {
	if !stream {
		return requireUsageTokens(body, "usage", "input_tokens", "output_tokens")
	}
	var started, delivered bool
	for _, data := range protocolSSEDataFrames(body) {
		started = started || requireUsageTokens([]byte(data), "message.usage", "input_tokens") == nil
		delivered = delivered || requireUsageTokens([]byte(data), "usage", "output_tokens") == nil
	}
	if !started || !delivered {
		return fmt.Errorf("messages stream lost input or output usage: %s", truncateString(string(body), 1200))
	}
	return nil
}

func assertAgentResponsesUsage(body []byte, stream bool, _ map[string]any) error {
	if !stream {
		return requireUsageTokens(body, "usage", "input_tokens", "output_tokens", "total_tokens")
	}
	for _, data := range protocolSSEDataFrames(body) {
		if strings.Contains(data, `"response.completed"`) {
			return requireUsageTokens([]byte(data), "response.usage", "input_tokens", "output_tokens", "total_tokens")
		}
	}
	return fmt.Errorf("responses stream has no completed event: %s", truncateString(string(body), 1200))
}

func requireUsageTokens(body []byte, objectPath string, keys ...string) error {
	var value any
	if err := json.Unmarshal(body, &value); err != nil {
		return err
	}
	for _, segment := range strings.Split(objectPath, ".") {
		object, _ := value.(map[string]any)
		value = object[segment]
	}
	usage, ok := value.(map[string]any)
	if !ok {
		return fmt.Errorf("missing %s: %s", objectPath, truncateString(string(body), 600))
	}
	for _, key := range keys {
		if _, ok := usage[key].(float64); !ok {
			return fmt.Errorf("%s has no numeric %s: %s", objectPath, key, truncateString(string(body), 600))
		}
	}
	return nil
}
