package protocolcodec

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestResponsesCreatedAtNumbers(t *testing.T) {
	for _, test := range []struct {
		name  string
		wire  string
		nanos int64
	}{
		{name: "integer", wire: "1792418136"},
		{name: "float whole seconds", wire: "1792418136.0"},
		{name: "float fractional seconds", wire: "1792418136.25", nanos: 250000000},
	} {
		t.Run(test.name, func(t *testing.T) {
			codec := OpenAIResponsesCodec{}
			policy := llmprotocol.DefaultPolicy()
			body := []byte(fmt.Sprintf(`{"id":"resp_1","object":"response","created_at":%s,"model":"m","status":"completed","output":[]}`, test.wire))
			response, envelope, _, err := codec.DecodeResponse(body, policy)
			if err != nil {
				t.Fatal(err)
			}
			want := time.Unix(1792418136, test.nanos).UTC()
			if !response.CreatedAt.Equal(want) {
				t.Fatalf("created_at = %s, want %s", response.CreatedAt, want)
			}
			response.Generation++
			encoded, _, err := codec.EncodeResponse(response, envelope, policy)
			if err != nil {
				t.Fatal(err)
			}
			var resource struct {
				CreatedAt json.Number `json:"created_at"`
			}
			if unmarshalErr := json.Unmarshal(encoded, &resource); unmarshalErr != nil {
				t.Fatal(unmarshalErr)
			}
			wantWire := "1792418136"
			if string(resource.CreatedAt) != wantWire {
				t.Fatalf("encoded created_at = %s, want %s", resource.CreatedAt, wantWire)
			}
			decoder := codec.NewDecoder(llmprotocol.StreamContext{
				Source: llmprotocol.OpenAIResponsesV1, Target: llmprotocol.OpenAIResponsesV1,
			}, policy)
			payload := strings.ReplaceAll(string(streamFixture(llmprotocol.OpenAIResponsesV1)), `"created_at":100`, `"created_at":`+test.wire)
			events, _, err := decoder.Push([]byte(payload))
			if err != nil {
				t.Fatal(err)
			}
			if len(events) == 0 || events[len(events)-1].Type != llmprotocol.EventResponseCompleted {
				t.Fatalf("stream did not complete: %+v", events)
			}
			if _, _, err := decoder.Finalize(nil); err != nil {
				t.Fatal(err)
			}
		})
	}
}
