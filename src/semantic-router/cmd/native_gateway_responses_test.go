//go:build !windows

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestStandaloneStoredResponseIncompleteDetails(t *testing.T) {
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	for _, stream := range []bool{false, true} {
		for _, tc := range []struct {
			name, finish, status, reason string
		}{
			{"completed", "stop", "completed", ""},
			{"output limit", "length", "incomplete", "max_output_tokens"},
			{"content filter", "content_filter", "incomplete", "content_filter"},
		} {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, stream), func(t *testing.T) {
				var calls atomic.Int32
				backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					calls.Add(1)
					var request struct{ Stream bool }
					if decodeErr := json.NewDecoder(r.Body).Decode(&request); decodeErr != nil {
						http.Error(w, decodeErr.Error(), http.StatusBadRequest)
						return
					}
					if request.Stream {
						w.Header().Set("Content-Type", "text/event-stream")
						_, _ = io.WriteString(w, "data: "+`{"id":"chatcmp_provider","object":"chat.completion.chunk","model":"m","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"answer"},"finish_reason":null}]}`+"\n\n")
						_, _ = fmt.Fprintf(w, "data: "+`{"id":"chatcmp_provider","object":"chat.completion.chunk","model":"m","created":1,"choices":[{"index":0,"delta":{},"finish_reason":%q}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\ndata: [DONE]\n\n", tc.finish)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					_, _ = fmt.Fprintf(w, `{"id":"chatcmp_provider","object":"chat.completion","model":"m","created":1,"choices":[{"index":0,"message":{"role":"assistant","content":"answer"},"finish_reason":%q}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`, tc.finish)
				}))
				defer backend.Close()
				document := backendDocument(backend.URL, "") + "global:\n  services:\n    response_api:\n      enabled: true\n      store_backend: memory\n"
				cfg, parseErr := config.ParseYAMLBytes([]byte(document))
				if parseErr != nil {
					t.Fatal(parseErr)
				}
				path := filepath.Join(t.TempDir(), "config.yaml")
				if writeErr := os.WriteFile(path, []byte(document), 0o600); writeErr != nil {
					t.Fatal(writeErr)
				}
				server, serverErr := extproc.NewServer(path, 0, false, "", routerruntime.NewRegistry(cfg),
					extproc.WithGatewayMode(config.GatewayStandalone), extproc.WithConfigParts(upstreamPart(config.GatewayStandalone)))
				if serverErr != nil {
					t.Fatal(serverErr)
				}
				ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
				defer cancel()
				done := make(chan error, 1)
				go func() { done <- server.StartContextWithoutExtProc(ctx, nil) }()
				defer func() {
					cancel()
					if runErr := <-done; runErr != nil {
						t.Errorf("routing core shutdown: %v", runErr)
					}
				}()
				if servingErr := server.WaitForServing(ctx); servingErr != nil {
					t.Fatal(servingErr)
				}
				handler, handlerErr := gateway.NewHandler(gateway.Options{Serving: nativeServing{pin: server.Pin}, Listener: "http-8899"})
				if handlerErr != nil {
					t.Fatal(handlerErr)
				}
				front := httptest.NewServer(handler)
				defer front.Close()
				client := front.Client()
				client.Timeout = 5 * time.Second
				post, postErr := client.Post(front.URL+"/v1/responses", "application/json",
					strings.NewReader(fmt.Sprintf(`{"model":"m","input":"answer","max_output_tokens":16,"stream":%t}`, stream)))
				if postErr != nil {
					t.Fatal(postErr)
				}
				body, readErr := io.ReadAll(post.Body)
				post.Body.Close()
				if readErr != nil || post.StatusCode != http.StatusOK {
					t.Fatalf("POST: %d %s (%v)", post.StatusCode, body, readErr)
				}
				live := storedResponseHTTPResult(t, body, stream)
				checkStoredHTTPIncompleteDetails(t, live, tc.status, tc.reason)
				if live.ID == "" {
					t.Fatal("POST did not publish a response ID")
				}
				get, getErr := client.Get(front.URL + "/v1/responses/" + live.ID)
				if getErr != nil {
					t.Fatal(getErr)
				}
				retained, retainedErr := io.ReadAll(get.Body)
				get.Body.Close()
				if retainedErr != nil || get.StatusCode != http.StatusOK {
					t.Fatalf("GET: %d %s (%v)", get.StatusCode, retained, retainedErr)
				}
				if calls.Load() != 1 {
					t.Fatalf("GET dispatched inference: %d provider calls", calls.Load())
				}
				stored := storedResponseHTTPResult(t, retained, false)
				if stored.ID != live.ID {
					t.Fatalf("GET ID %q, want %q", stored.ID, live.ID)
				}
				checkStoredHTTPIncompleteDetails(t, stored, tc.status, tc.reason)
				if _, _, _, decodeErr := protocolcodec.NewBuiltinEngine().DecodeResponse(llmprotocol.OpenAIResponsesV1, retained); decodeErr != nil {
					t.Fatalf("GET is not a valid Responses object: %v", decodeErr)
				}
			})
		}
	}
}

func storedResponseHTTPResult(t *testing.T, body []byte, stream bool) responseapi.ResponseAPIResponse {
	t.Helper()
	if stream {
		var terminal json.RawMessage
		for _, line := range bytes.Split(body, []byte("\n")) {
			data, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data:"))
			if !ok {
				continue
			}
			var event struct {
				Type     string          `json:"type"`
				Response json.RawMessage `json:"response"`
			}
			if eventErr := json.Unmarshal(bytes.TrimSpace(data), &event); eventErr != nil {
				t.Fatalf("invalid Responses event: %s (%v)", data, eventErr)
			}
			if event.Type == "response.completed" || event.Type == "response.incomplete" {
				terminal = event.Response
			}
		}
		if len(terminal) == 0 {
			t.Fatalf("no terminal Responses object: %s", body)
		}
		body = terminal
	}
	var response responseapi.ResponseAPIResponse
	if decodeErr := json.Unmarshal(body, &response); decodeErr != nil {
		t.Fatal(decodeErr)
	}
	return response
}

func checkStoredHTTPIncompleteDetails(t *testing.T, response responseapi.ResponseAPIResponse, status, reason string) {
	t.Helper()
	if response.Status != status {
		t.Fatalf("status %q, want %q", response.Status, status)
	}
	if reason == "" {
		if response.IncompleteDetails != nil {
			t.Fatalf("completed response acquired incomplete details: %+v", response.IncompleteDetails)
		}
	} else if response.IncompleteDetails == nil || response.IncompleteDetails.Reason != reason {
		t.Fatalf("reason %q not retained: %+v", reason, response.IncompleteDetails)
	}
}
