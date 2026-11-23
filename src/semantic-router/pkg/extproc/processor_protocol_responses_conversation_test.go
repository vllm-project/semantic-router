package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

// The Router keeps no Conversations objects, so it could neither load a
// conversation's items for a request nor add the response to them. A
// Responses request that names a conversation fails before any backend,
// buffered and streamed, whatever the backend's format; a null conversation
// names none, and previous_response_id continues a stored response with its
// history.
func TestResponsesConversationIsRefusedBeforeAnyBackend(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(noOpFieldsConfig))
	require.NoError(t, err)
	recorder := parity.NewRecorder(newParityRouter(t, cfg), routing.DefaultOptions)
	run := func(name, body string) *parity.Record {
		t.Helper()
		record := recorder.Run(context.Background(), parity.Case{
			Name:    name,
			Request: parity.CaseRequest{Method: "POST", Path: "/v1/responses", Headers: [][]string{{"content-type", "application/json"}}, Body: body},
			Upstream: &parity.Upstream{
				Status: http.StatusOK, Headers: [][]string{{"content-type", "application/json"}},
				Body: string(extProcResponseFixture(llmprotocol.OpenAIChatV1)),
			},
		})
		require.Empty(t, record.Error)
		require.NotNil(t, record.Response)
		return record
	}

	for _, test := range []struct{ name, body string }{
		{"an ID", `{"model":"chat-model","input":"My code word is kestrel.","conversation":"conv_1"}`},
		{"an object", `{"model":"responses-model","input":"My code word is kestrel.","conversation":{"id":"conv_1"}}`},
		{"a streamed request", `{"model":"chat-model","input":"What is my code word?","conversation":"conv_1","stream":true}`},
		{"a request that keeps no response", `{"model":"chat-model","input":"What is my code word?","conversation":"conv_1","store":false}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			record := run(test.name, test.body)
			require.Nil(t, record.Upstream, "a backend received the request")
			require.Equal(t, http.StatusBadRequest, record.Response.Status, "%s", record.Response.Body)
			var body struct {
				Error struct {
					Type  string `json:"type"`
					Code  string `json:"code"`
					Param string `json:"param"`
				} `json:"error"`
			}
			require.NoError(t, json.Unmarshal(record.Response.Body, &body), "%s", record.Response.Body)
			require.Equal(t, "invalid_request_error", body.Error.Type)
			require.Equal(t, "unsupported_conversation", body.Error.Code)
			require.Equal(t, "conversation", body.Error.Param)
		})
	}

	first := run("a null conversation", `{"model":"chat-model","input":"My code word is kestrel.","conversation":null}`)
	require.Equal(t, http.StatusOK, first.Response.Status, "%s", first.Response.Body)
	require.NotNil(t, first.Upstream)
	require.NotContains(t, string(first.Upstream.Body), "conversation")
	var stored struct {
		ID string `json:"id"`
	}
	require.NoError(t, json.Unmarshal(first.Response.Body, &stored))
	require.True(t, strings.HasPrefix(stored.ID, "resp_"), "response ID = %q", stored.ID)

	next := run("previous_response_id", `{"model":"chat-model","input":"What is my code word?","previous_response_id":"`+stored.ID+`"}`)
	require.Equal(t, http.StatusOK, next.Response.Status, "%s", next.Response.Body)
	require.NotNil(t, next.Upstream)
	for _, turn := range []string{"My code word is kestrel.", "hello", "What is my code word?"} {
		require.Contains(t, string(next.Upstream.Body), turn, "the backend did not receive the stored history")
	}
}
