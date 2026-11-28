package extproc

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

func TestStructuredOutputUsesTheRemainingUpstreamTimeout(t *testing.T) {
	const fast = 35 * time.Millisecond
	for _, test := range []struct {
		name, listener, model, decision string
		first, second                   time.Duration
		status                          int
	}{
		{"second call exceeds listener timeout", "150ms", "", "", fast, 400 * time.Millisecond, http.StatusGatewayTimeout},
		{"first call spends part of the budget", "500ms", "", "", 350 * time.Millisecond, 350 * time.Millisecond, http.StatusGatewayTimeout},
		{"both calls fit", "150ms", "", "", fast, fast, http.StatusOK},
		{"model overrides the listener", "150ms", "2s", "", fast, 400 * time.Millisecond, http.StatusOK},
		{"model disables the total timeout", "150ms", "0s", "", fast, 400 * time.Millisecond, http.StatusOK},
		{"no decision override keeps the model timeout", "3s", "350ms", "", fast, 700 * time.Millisecond, http.StatusGatewayTimeout},
		{"decision extends the model timeout", "3s", "350ms", "2s", fast, 700 * time.Millisecond, http.StatusOK},
		{"decision disables the model total timeout", "3s", "350ms", "0s", fast, 700 * time.Millisecond, http.StatusOK},
	} {
		t.Run(test.name, func(t *testing.T) {
			backend := &umbrellaBackend{t: t, format: llmprotocol.OpenAIChatV1}
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("read the upstream request: %v", err)
					return
				}
				request := backend.decode(body)
				calls.Add(1)
				delay := test.first
				if request.OutputFormat.Kind == llmprotocol.OutputJSONSchema {
					delay = test.second
				}
				timer := time.NewTimer(delay)
				defer timer.Stop()
				select {
				case <-timer.C:
				case <-r.Context().Done():
					return
				}
				reply, contentType := backend.reply(request)
				w.Header().Set("Content-Type", contentType)
				_, _ = w.Write(reply)
			}))
			defer server.Close()

			yaml := umbrellaModesConfig(strings.TrimPrefix(server.URL, "http://"))
			yaml = strings.Replace(yaml, "    timeout: 300s\n", "    timeout: "+test.listener+"\n", 1)
			if test.model != "" {
				yaml = strings.Replace(yaml, "      api_format: openai\n",
					"      api_format: openai\n      reliability:\n        total_timeout: "+test.model+"\n", 1)
			}
			if test.decision != "" {
				yaml = strings.Replace(yaml, "        priority: 10\n",
					"        priority: 10\n        reliability:\n          total_timeout: "+test.decision+"\n", 1)
			}
			cfg, err := config.ParseYAMLBytes([]byte(yaml))
			require.NoError(t, err)
			topology, err := upstream.Compile(cfg)
			require.NoError(t, err)
			pools, err := upstream.New(topology, upstream.Options{})
			require.NoError(t, err)
			defer pools.Close(context.Background())

			handler, err := gateway.NewHandler(gateway.Options{
				Listener: "http-8899",
				Serving: gateway.Static(gateway.Serving{
					Engine: looperAdapters["sessions"](newParityRouter(t, cfg)), Upstream: pools,
				}),
			})
			require.NoError(t, err)
			front := httptest.NewServer(handler)
			defer front.Close()

			request := umbrellaRequest(llmprotocol.OpenAIChatV1, "vllm-sr/auto", true, false)
			client := &http.Client{Timeout: 5 * time.Second}
			res, err := client.Post(front.URL+"/v1/chat/completions", "application/json",
				strings.NewReader(string(umbrellaBody(t, request))))
			require.NoError(t, err)
			defer res.Body.Close()
			body, err := io.ReadAll(res.Body)
			require.NoError(t, err)
			require.Equal(t, int32(2), calls.Load())
			require.Equal(t, test.status, res.StatusCode, "%s", body)
			if test.status == http.StatusGatewayTimeout {
				require.Contains(t, string(body), "upstream_timeout")
			} else {
				got := clientAnswer(t, llmprotocol.OpenAIChatV1, string(body), false)
				require.JSONEq(t, umbrellaAdvice, answerText(got))
			}
		})
	}
}
