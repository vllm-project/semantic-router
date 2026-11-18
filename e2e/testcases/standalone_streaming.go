package testcases

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("standalone-streaming", pkgtestcases.TestCase{
		Description: "The native listener streams a routed chat completion without buffering it, and cancels the backend stream when a client hangs up",
		Tags:        []string{"standalone", "gateway", "streaming", "kubernetes"},
		Fn:          testStandaloneStreaming,
	})
}

// testStandaloneStreaming covers the native listener's own response path,
// which the Envoy streaming profile never reaches. provider-mocker holds the
// stream after its first chunk (`__mock_held_stream__`): the client must get
// that chunk while the backend is still held, so a listener that buffers the
// answer fails, and the rest must arrive in order once released. A client that
// hangs up on a held stream must cancel it at the backend, and the listener
// must serve the next request.
func testStandaloneStreaming(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()
	backend, err := fixtures.OpenServiceEndpointSession(ctx, client, opts, "default", "provider-mocker", "8000")
	if err != nil {
		return fmt.Errorf("reach provider-mocker: %w", err)
	}
	defer backend.Close()
	runID := fmt.Sprintf("%d", time.Now().UnixNano())

	chunks, err := streamThroughHeldBackend(ctx, session, backend, "standalone-stream-"+runID)
	if err != nil {
		return err
	}
	if err = dropHeldStream(ctx, session, backend, "standalone-drop-"+runID); err != nil {
		return err
	}
	reply, err := standaloneChat(ctx, client, opts, "Say hello after a dropped stream.")
	if err != nil {
		return fmt.Errorf("after a client dropped a stream: %w", err)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{"chunks": chunks, "model_after_drop": reply.model})
	}
	return reply.servedBy(standalonePrimaryModel, standaloneDefaultRoute)
}

// streamThroughHeldBackend checks that the first chunk reaches the client while
// the backend holds the rest, then that the released stream reassembles into
// provider-mocker's echo for the routed model and ends with [DONE].
func streamThroughHeldBackend(ctx context.Context, session, backend *fixtures.ServiceSession, sessionID string) (int, error) {
	response, err := postStandaloneStream(ctx, session, sessionID)
	if err != nil {
		return 0, err
	}
	defer response.Body.Close()
	if err = checkStandaloneStreamHeaders(response); err != nil {
		return 0, err
	}
	reader := bufio.NewReader(response.Body)
	first, err := readFirstEvent(reader, 15*time.Second)
	if err != nil {
		return 0, fmt.Errorf("no chunk reached the client while the backend held the stream, so the listener buffered it: %w", err)
	}
	state, err := heldStreamState(ctx, backend, sessionID, "")
	if err != nil {
		return 0, fmt.Errorf("read the backend stream state after the first chunk: %w", err)
	}
	if state != "holding" {
		return 0, fmt.Errorf("the backend stream is %q after the first chunk, want holding", state)
	}
	if _, err = heldStreamState(ctx, backend, sessionID, "/release"); err != nil {
		return 0, err
	}
	content, chunks, err := readStandaloneStream(io.MultiReader(strings.NewReader(first), reader))
	if err != nil {
		return chunks, err
	}
	var echo map[string]any
	if err = json.Unmarshal([]byte(content), &echo); err != nil || echo["mock"] != "provider-mocker" || echo["model"] != standalonePrimaryModel {
		return chunks, fmt.Errorf("the reassembled stream is not provider-mocker's echo for %q: %s", standalonePrimaryModel, truncateString(content, 400))
	}
	return chunks, nil
}

// dropHeldStream hangs up on a held stream after its first chunk and waits for
// the backend to see the cancellation.
func dropHeldStream(ctx context.Context, session, backend *fixtures.ServiceSession, sessionID string) error {
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	response, err := postStandaloneStream(streamCtx, session, sessionID)
	if err != nil {
		return fmt.Errorf("open the stream to drop: %w", err)
	}
	if err = checkStandaloneStreamHeaders(response); err != nil {
		_ = response.Body.Close()
		return err
	}
	if _, err = readFirstEvent(bufio.NewReader(response.Body), 15*time.Second); err != nil {
		_ = response.Body.Close()
		return fmt.Errorf("read the stream to drop: %w", err)
	}
	cancel()
	_ = response.Body.Close()
	deadline := time.Now().Add(15 * time.Second)
	for {
		state, stateErr := heldStreamState(ctx, backend, sessionID, "")
		if stateErr == nil && state == "cancelled" {
			return nil
		}
		if time.Now().After(deadline) {
			if stateErr != nil {
				return fmt.Errorf("read the backend stream state after the client hung up: %w", stateErr)
			}
			return fmt.Errorf("the backend stream is %q after the client hung up, want cancelled", state)
		}
		time.Sleep(200 * time.Millisecond)
	}
}

func checkStandaloneStreamHeaders(response *http.Response) error {
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(response.Body)
		return fmt.Errorf("streamed chat status %d: %s", response.StatusCode, truncateString(string(body), 400))
	}
	if contentType := response.Header.Get("Content-Type"); !strings.HasPrefix(contentType, "text/event-stream") {
		return fmt.Errorf("streamed chat Content-Type %q, want text/event-stream", contentType)
	}
	if model := response.Header.Get("x-vsr-selected-model"); model != standalonePrimaryModel {
		return fmt.Errorf("streamed chat selected model %q, want %q", model, standalonePrimaryModel)
	}
	return nil
}

// readFirstEvent returns the stream's first data line, followed by the blank
// line that ends the event, or fails when none arrives within timeout.
func readFirstEvent(reader *bufio.Reader, timeout time.Duration) (string, error) {
	type line struct {
		text string
		err  error
	}
	got := make(chan line, 1)
	go func() {
		for {
			text, err := reader.ReadString('\n')
			if err != nil || strings.HasPrefix(text, "data: ") {
				got <- line{text, err}
				return
			}
		}
	}()
	select {
	case result := <-got:
		if result.err != nil {
			return "", result.err
		}
		return result.text + "\n", nil
	case <-time.After(timeout):
		return "", fmt.Errorf("nothing within %s", timeout)
	}
}

// heldStreamState asks provider-mocker for a held stream's state, or releases
// it when action is "/release".
func heldStreamState(ctx context.Context, backend *fixtures.ServiceSession, sessionID, action string) (string, error) {
	method := http.MethodGet
	if action != "" {
		method = http.MethodPost
	}
	request, err := http.NewRequestWithContext(ctx, method, backend.URL("/debug/held-stream/"+sessionID+action), nil)
	if err != nil {
		return "", err
	}
	response, err := backend.HTTPClient(10 * time.Second).Do(request)
	if err != nil {
		return "", err
	}
	defer response.Body.Close()
	var body struct {
		State string `json:"state"`
	}
	if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
		return "", err
	}
	return body.State, nil
}

func postStandaloneStream(ctx context.Context, session *fixtures.ServiceSession, sessionID string) (*http.Response, error) {
	body, err := json.Marshal(map[string]any{
		"model":    "vllm-sr/auto",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "__mock_held_stream__ Stream hello from the standalone Router."}},
	})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, session.URL("/v1/chat/completions"), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")
	request.Header.Set(shortCircuitSessionHeader, sessionID)
	return session.HTTPClient(90 * time.Second).Do(request)
}

// readStandaloneStream reassembles the streamed answer and counts the chunks
// that carried part of it; the stream must end with [DONE].
func readStandaloneStream(body io.Reader) (string, int, error) {
	var content strings.Builder
	chunks := 0
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for scanner.Scan() {
		data, ok := strings.CutPrefix(scanner.Text(), "data: ")
		if !ok {
			continue
		}
		if data == "[DONE]" {
			return content.String(), chunks, nil
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return "", chunks, fmt.Errorf("an SSE chunk is not a chat completion chunk: %s", truncateString(data, 300))
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content != "" {
				content.WriteString(choice.Delta.Content)
				chunks++
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return "", chunks, err
	}
	return "", chunks, fmt.Errorf("the stream ended without [DONE] after %d chunk(s)", chunks)
}
