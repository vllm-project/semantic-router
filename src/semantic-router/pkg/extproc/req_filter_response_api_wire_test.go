package extproc

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

func TestGetResponseRetainsCanonicalContentFields(t *testing.T) {
	for _, test := range []struct {
		name    string
		content string
	}{
		{"text_without_citations", `{"type":"output_text","text":"A useful answer","annotations":[]}`},
		{"empty_text", `{"type":"output_text","text":"","annotations":[]}`},
		{"text_with_citation", `{"type":"output_text","text":"A useful answer","annotations":[{"type":"url_citation","url":"https://example.com","title":"Example","start_index":0,"end_index":1}]}`},
		{"refusal", `{"type":"refusal","refusal":"No answer available"}`},
		{"empty_refusal", `{"type":"refusal","refusal":""}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			original := []byte(`{"id":"resp_stored","object":"response","created_at":1,"model":"model","status":"completed","output":[{"id":"msg_stored","type":"message","role":"assistant","status":"completed","content":[` + test.content + `]}]}`)
			engine := protocolcodec.NewBuiltinEngine()
			_, _, _, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, original)
			require.NoError(t, err, "normal source response must meet the existing strict codec contract")
			var stored responseapi.StoredResponse
			require.NoError(t, json.Unmarshal(original, &stored))
			store, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
			require.NoError(t, err)
			t.Cleanup(func() { require.NoError(t, store.Close()) })
			require.NoError(t, store.StoreResponse(t.Context(), &stored))
			response, err := NewResponseAPIFilter(store).HandleGetResponse(t.Context(), stored.ID)
			require.NoError(t, err)
			require.EqualValues(t, 200, response.GetImmediateResponse().GetStatus().GetCode())
			body := response.GetImmediateResponse().GetBody()
			_, _, _, decodeErr := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, body)
			t.Logf("strict GET decode result: %v; body: %s", decodeErr, body)
			require.NoError(t, decodeErr, "GET must retain a valid Responses object")
			var got, want struct {
				Output json.RawMessage `json:"output"`
			}
			require.NoError(t, json.Unmarshal(body, &got))
			require.NoError(t, json.Unmarshal(original, &want))
			require.JSONEq(t, string(want.Output), string(got.Output), "GET must preserve the public output content fields")
		})
	}
}

func assertStoredResponseMatchesClientOutput(t *testing.T, filter *ResponseAPIFilter, id string, clientBody []byte, stream bool) {
	t.Helper()
	if stream {
		var completed json.RawMessage
		for _, line := range strings.Split(string(clientBody), "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			var event struct {
				Type     string          `json:"type"`
				Response json.RawMessage `json:"response"`
			}
			require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event))
			if event.Type == "response.completed" {
				completed = event.Response
			}
		}
		require.NotEmpty(t, completed)
		clientBody = completed
	}
	response, err := filter.HandleGetResponse(t.Context(), id)
	require.NoError(t, err)
	require.EqualValues(t, 200, response.GetImmediateResponse().GetStatus().GetCode())
	var got, want struct {
		ID                 string          `json:"id"`
		PreviousResponseID string          `json:"previous_response_id"`
		Output             json.RawMessage `json:"output"`
	}
	require.NoError(t, json.Unmarshal(clientBody, &want))
	require.NoError(t, json.Unmarshal(response.GetImmediateResponse().GetBody(), &got))
	require.Equal(t, want.ID, got.ID)
	require.Equal(t, want.PreviousResponseID, got.PreviousResponseID)
	require.JSONEq(t, string(want.Output), string(got.Output))
}
