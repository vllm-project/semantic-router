package testcases

// Helpers for anthropic_messages_request_preservation.go.
//
// Split out to keep the main file under the 800-line cap. This file holds:
//   - Constants for the shim service location.
//   - Request body types (preservationContentBlock, imageSource, ...).
//   - Shim debug response type.
//   - Infrastructure helpers (port-forwarding, HTTP send/fetch).
//   - Navigation helpers for the shim's recorded JSON body.
//   - Image-block assertion helpers.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

// shimServiceNamespace is the Kubernetes namespace for the anthropic-shim
// service (declared in e2e/profiles/anthropic-shim/gateway-resources/backend.yaml).
const shimServiceNamespace = "anthropic-backend-system"

// shimServiceName is the Kubernetes Service name for the shim. Its port is 9080
// (the shim port; the llama-server port 8080 is cluster-internal only).
const shimServiceName = "anthropic-backend-qwen"

// shimServicePort is the Service port for the anthropic-shim container.
const shimServicePort = "9080"

// tinyPNG1x1 is a 1×1 transparent PNG image, base64-encoded.
// Used as a minimal valid image payload for the image-block preservation test.
const tinyPNG1x1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

// ── Request body types ────────────────────────────────────────────────────────

// preservationContentBlock is a generic Anthropic content block that supports
// type, text, source (for image blocks), tool_use_id, is_error, and content
// (for tool_result blocks). Using interface{} for content and source lets a
// single type cover all block shapes without requiring parallel type families.
type preservationContentBlock struct {
	Type      string      `json:"type"`
	Text      string      `json:"text,omitempty"`
	Source    interface{} `json:"source,omitempty"`
	ToolUseID string      `json:"tool_use_id,omitempty"`
	IsError   bool        `json:"is_error,omitempty"`
	Content   interface{} `json:"content,omitempty"`
}

// imageSource is the Anthropic base64 image source shape.
type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// preservationMessage is a rich Anthropic message that accepts content as
// either a string (simple) or a slice of blocks (complex). We always use
// the block form in this file.
type preservationMessage struct {
	Role    string                     `json:"role"`
	Content []preservationContentBlock `json:"content"`
}

// preservationRequestBody is the POST /v1/messages payload for preservation
// tests. It carries the fields the sibling anthropicMessagesRequestBody type
// does not (top_k, metadata, rich content blocks), so it lives here and is
// not shared with siblings to avoid expanding their scope.
type preservationRequestBody struct {
	Model         string                `json:"model"`
	MaxTokens     int                   `json:"max_tokens"`
	Messages      []preservationMessage `json:"messages"`
	TopK          int                   `json:"top_k,omitempty"`
	Temperature   *float64              `json:"temperature,omitempty"`
	TopP          *float64              `json:"top_p,omitempty"`
	Metadata      interface{}           `json:"metadata,omitempty"`
	StopSequences []string              `json:"stop_sequences,omitempty"`
}

// ── Shim debug response type ──────────────────────────────────────────────────

// shimDebugResponse mirrors the JSON shape returned by the shim's
// GET /debug/last-request endpoint:
//
//	{"session_id": "...", "body": {...}, "headers": {...}}
//
// body is decoded as a raw map so individual field assertions can be
// expressed at test level without coupling this file to every possible shape.
type shimDebugResponse struct {
	SessionID string                 `json:"session_id"`
	Body      map[string]interface{} `json:"body"`
	Headers   map[string]interface{} `json:"headers"`
}

// ── Infrastructure helpers ────────────────────────────────────────────────────

// openShimSession opens a port-forward to the anthropic-shim service and
// returns the local port string and a stop function. Callers must defer stop().
func openShimSession(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) (string, func(), error) {
	localPort, err := allocateLocalPort()
	if err != nil {
		return "", nil, fmt.Errorf("allocate local port for shim: %w", err)
	}

	stop, err := helpers.StartPortForward(
		ctx,
		client,
		opts.RestConfig,
		shimServiceNamespace,
		shimServiceName,
		fmt.Sprintf("%s:%s", localPort, shimServicePort),
		opts.Verbose,
	)
	if err != nil {
		return "", nil, fmt.Errorf("port-forward to shim %s/%s: %w", shimServiceNamespace, shimServiceName, err)
	}

	// Give the port-forward a moment to stabilise before the first request.
	time.Sleep(2 * time.Second)
	return localPort, stop, nil
}

// allocateLocalPort finds a free TCP port on the local machine by briefly
// listening on :0 and immediately closing the listener.
//
// TOCTOU note: between Close and the subsequent StartPortForward bind, another
// process can claim the port. There is no fix available without passing a
// listener socket into the port-forwarder, which the helpers API does not
// expose. In practice the race window is microseconds on an idle test host;
// accept it as a known limitation of the test harness.
func allocateLocalPort() (string, error) {
	l, err := net.Listen("tcp", ":0")
	if err != nil {
		return "", fmt.Errorf("listen :0: %w", err)
	}
	defer func() { _ = l.Close() }()
	addr := l.Addr().(*net.TCPAddr)
	return fmt.Sprintf("%d", addr.Port), nil
}

// postPreservationRequest sends a POST /v1/messages to the router with the
// given session ID header and body, and returns the response.
//
// The caller owns the returned response and MUST call resp.Body.Close() and
// drain the body before discarding, even on the error path, to avoid leaking
// the underlying HTTP connection back to the pool unconsumed.
func postPreservationRequest(
	ctx context.Context,
	body preservationRequestBody,
	sessionID string,
	localPort string,
) (*http.Response, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	url := fmt.Sprintf("http://localhost:%s/v1/messages", localPort)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	// anthropic-version is required by real SDK clients; the router does not
	// gate on it today but sending it matches the production client contract.
	req.Header.Set("anthropic-version", "2023-06-01")
	// x-vsr-test-session-id is the session tracking header used by the shim
	// to key its per-session request store.
	req.Header.Set("x-vsr-test-session-id", sessionID)

	httpClient := &http.Client{Timeout: 120 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do: %w", err)
	}
	return resp, nil
}

// fetchShimDebugBody queries the shim's /debug/last-request endpoint for the
// given session and returns the parsed response. Returns an error if the shim
// responds with a non-200 status (e.g. 404 = session not seen yet).
func fetchShimDebugBody(ctx context.Context, shimPort string, sessionID string) (shimDebugResponse, error) {
	// Query param form: GET /debug/last-request?x-vsr-test-session-id=<id>
	// Both header and query-param are supported; query param avoids the need
	// to thread headers through a separate HTTP client setup.
	url := fmt.Sprintf(
		"http://localhost:%s/debug/last-request?x-vsr-test-session-id=%s",
		shimPort,
		sessionID,
	)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return shimDebugResponse{}, fmt.Errorf("new request: %w", err)
	}

	httpClient := &http.Client{Timeout: 15 * time.Second}
	resp, err := httpClient.Do(req)
	if err != nil {
		return shimDebugResponse{}, fmt.Errorf("do: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return shimDebugResponse{}, fmt.Errorf("read: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return shimDebugResponse{}, fmt.Errorf(
			"shim /debug/last-request returned %d for session %q: %s",
			resp.StatusCode, sessionID, truncateString(string(raw), 200),
		)
	}

	var parsed shimDebugResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return shimDebugResponse{}, fmt.Errorf(
			"unmarshal shim debug response: %w (body=%s)", err, truncateString(string(raw), 200),
		)
	}
	return parsed, nil
}

// ── Navigation helpers ────────────────────────────────────────────────────────

// extractFirstContentBlock navigates body.messages[0].content[0] and returns
// the block as a map. Returns an error if the path does not exist.
func extractFirstContentBlock(body map[string]interface{}) (map[string]interface{}, error) {
	messages, err := messagesSlice(body)
	if err != nil {
		return nil, err
	}
	if len(messages) == 0 {
		return nil, fmt.Errorf("messages array is empty")
	}
	msg0, ok := messages[0].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("messages[0] is not an object")
	}
	content, err := contentSlice(msg0)
	if err != nil {
		return nil, err
	}
	if len(content) == 0 {
		return nil, fmt.Errorf("messages[0].content array is empty")
	}
	block, ok := content[0].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("messages[0].content[0] is not an object")
	}
	return block, nil
}

// extractToolResultIsError navigates body.messages[2].content[0].is_error
// and returns the boolean value. The third message (index 2) is the user
// turn containing the tool_result block in the is_error test scenario.
//
// Returns an error if the is_error field is missing entirely so a maintainer
// can distinguish "router dropped the field" from "field is present and false".
func extractToolResultIsError(body map[string]interface{}) (bool, error) {
	messages, err := messagesSlice(body)
	if err != nil {
		return false, err
	}
	if len(messages) < 3 {
		return false, fmt.Errorf("expected at least 3 messages, got %d", len(messages))
	}
	msg2, ok := messages[2].(map[string]interface{})
	if !ok {
		return false, fmt.Errorf("messages[2] is not an object")
	}
	content, err := contentSlice(msg2)
	if err != nil {
		return false, err
	}
	if len(content) == 0 {
		return false, fmt.Errorf("messages[2].content is empty")
	}
	block, ok := content[0].(map[string]interface{})
	if !ok {
		return false, fmt.Errorf("messages[2].content[0] is not an object")
	}
	raw, present := block["is_error"]
	if !present {
		return false, fmt.Errorf("messages[2].content[0].is_error field missing (router may have stripped it)")
	}
	isError, ok := raw.(bool)
	if !ok {
		return false, fmt.Errorf("messages[2].content[0].is_error is not a bool (got %T)", raw)
	}
	return isError, nil
}

// extractToolResultContent navigates body.messages[2].content[0].content and
// returns it as a string. After join_tool_result_content the shim stores the
// collapsed string, so the return type is always string.
func extractToolResultContent(body map[string]interface{}) (string, error) {
	messages, err := messagesSlice(body)
	if err != nil {
		return "", err
	}
	if len(messages) < 3 {
		return "", fmt.Errorf("expected at least 3 messages, got %d", len(messages))
	}
	msg2, ok := messages[2].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("messages[2] is not an object")
	}
	content, err := contentSlice(msg2)
	if err != nil {
		return "", err
	}
	if len(content) == 0 {
		return "", fmt.Errorf("messages[2].content is empty")
	}
	block, ok := content[0].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("messages[2].content[0] is not an object")
	}
	raw, ok := block["content"]
	if !ok {
		return "", fmt.Errorf("messages[2].content[0].content field missing")
	}
	str, ok := raw.(string)
	if !ok {
		return "", fmt.Errorf("messages[2].content[0].content is not a string (got %T): shim may not have joined array", raw)
	}
	return str, nil
}

func messagesSlice(body map[string]interface{}) ([]interface{}, error) {
	raw, ok := body["messages"]
	if !ok {
		return nil, fmt.Errorf("messages field missing from shim body")
	}
	msgs, ok := raw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("messages is not an array (got %T)", raw)
	}
	return msgs, nil
}

func contentSlice(msg map[string]interface{}) ([]interface{}, error) {
	raw, ok := msg["content"]
	if !ok {
		return nil, fmt.Errorf("content field missing from message")
	}
	content, ok := raw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("content is not an array (got %T)", raw)
	}
	return content, nil
}

// extractStopSequences reads body.stop_sequences and returns it as a []string.
// Returns an error if the field is missing (router may have dropped it in the
// inbound→outbound re-mapping) or is not a JSON array of strings.
func extractStopSequences(body map[string]interface{}) ([]string, error) {
	raw, present := body["stop_sequences"]
	if !present {
		return nil, fmt.Errorf("stop_sequences field missing from shim body (router may have dropped it)")
	}
	arr, ok := raw.([]interface{})
	if !ok {
		return nil, fmt.Errorf("stop_sequences is not an array (got %T)", raw)
	}
	out := make([]string, 0, len(arr))
	for i, v := range arr {
		s, ok := v.(string)
		if !ok {
			return nil, fmt.Errorf("stop_sequences[%d] is not a string (got %T)", i, v)
		}
		out = append(out, s)
	}
	return out, nil
}

// samplingFieldCase parameterises a numeric sampling-field preservation test
// (temperature, top_p). The two cases differ only in the field name, the value
// sent, and which request-body field to populate, so they share one runner.
type samplingFieldCase struct {
	label     string                                  // human label for log/error context
	sessionID string                                  // session-id prefix (a timestamp is appended)
	field     string                                  // JSON key to read back from the shim body
	want      float64                                 // value to send and assert verbatim
	apply     func(*preservationRequestBody, float64) // sets the field on the request body
}

// runSamplingFieldPreservation drives a numeric sampling-field round-trip: it
// POSTs a /v1/messages request carrying the field, then asserts the shim
// received the same numeric value (JSON numbers decode as float64).
func runSamplingFieldPreservation(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	c samplingFieldCase,
) error {
	if opts.Verbose {
		fmt.Printf("[Anthropic] Testing %s preservation through /v1/messages translation\n", c.label)
	}

	routerPort, stopRouter, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopRouter()

	shimPort, stopShim, err := openShimSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopShim()

	sessionID := fmt.Sprintf("%s-%d", c.sessionID, time.Now().UnixNano())

	body := preservationRequestBody{
		Model:     "MoM",
		MaxTokens: 32,
		Messages: []preservationMessage{
			{Role: "user", Content: []preservationContentBlock{{Type: "text", Text: "Say one word."}}},
		},
	}
	c.apply(&body, c.want)

	resp, err := postPreservationRequest(ctx, body, sessionID, routerPort)
	if err != nil {
		return fmt.Errorf("router request: %w", err)
	}
	defer resp.Body.Close()
	rawBody, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("expected 200 from router, got %d: %s", resp.StatusCode, truncateString(string(rawBody), 200))
	}

	debug, err := fetchShimDebugBody(ctx, shimPort, sessionID)
	if err != nil {
		return fmt.Errorf("shim debug: %w", err)
	}

	got, err := extractFloatField(debug.Body, c.field)
	if err != nil {
		return fmt.Errorf("extract %s from shim body: %w", c.field, err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{c.field: got})
	}

	if got != c.want {
		return fmt.Errorf("expected %s=%v in shim body, got %v", c.field, c.want, got)
	}
	return nil
}

// extractFloatField reads a top-level numeric field from the shim body.
// Returns an error if the field is missing (router may have dropped it) or is
// not a JSON number.
func extractFloatField(body map[string]interface{}, field string) (float64, error) {
	raw, present := body[field]
	if !present {
		return 0, fmt.Errorf("%s field missing from shim body (router may have dropped it)", field)
	}
	f, ok := raw.(float64)
	if !ok {
		return 0, fmt.Errorf("%s is not a number (got %T)", field, raw)
	}
	return f, nil
}

// equalStringSlices reports whether two string slices have identical length,
// elements, and order.
func equalStringSlices(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ── Image-block assertion helpers ─────────────────────────────────────────────

// readImageBlock asserts that block is an Anthropic image content block and
// returns its source fields. Returns an error if the block type is wrong or
// the source object is missing/malformed.
func readImageBlock(block map[string]interface{}) (imageSource, error) {
	blockType, _ := block["type"].(string)
	if blockType != "image" {
		return imageSource{}, fmt.Errorf("expected content[0].type=image, got %q", blockType)
	}
	src, ok := block["source"].(map[string]interface{})
	if !ok {
		return imageSource{}, fmt.Errorf("expected content[0].source to be an object, got %T", block["source"])
	}
	srcType, _ := src["type"].(string)
	mediaType, _ := src["media_type"].(string)
	data, _ := src["data"].(string)
	return imageSource{Type: srcType, MediaType: mediaType, Data: data}, nil
}

// assertImageSourceEqual compares the three fields of an Anthropic image
// source struct and returns a descriptive error on the first mismatch.
func assertImageSourceEqual(want, got imageSource) error {
	if got.Type != want.Type {
		return fmt.Errorf("expected source.type=%q, got %q", want.Type, got.Type)
	}
	if got.MediaType != want.MediaType {
		return fmt.Errorf("expected source.media_type=%q, got %q", want.MediaType, got.MediaType)
	}
	if got.Data != want.Data {
		return fmt.Errorf("expected source.data to match tinyPNG1x1, got %q", truncateString(got.Data, 80))
	}
	return nil
}
