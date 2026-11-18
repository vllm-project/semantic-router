package protocolcodec

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Typed OpenAI clients send "parameters": null for a function that takes no
// arguments; the Responses schema declares the member nullable.
var nullToolParameterRequests = map[llmprotocol.WireFormat]string{
	llmprotocol.OpenAIChatV1:      `{"model":"m","messages":[{"role":"user","content":"What time is it?"}],"tools":[{"type":"function","function":{"name":"get_time","strict":false%s}}]}`,
	llmprotocol.OpenAIResponsesV1: `{"model":"m","input":"What time is it?","tools":[{"type":"function","name":"get_time","strict":false%s}]}`,
}

func TestNullToolParametersDecodeLikeOmittedParameters(t *testing.T) {
	engine := NewBuiltinEngine()
	for format, fixture := range nullToolParameterRequests {
		t.Run(string(format), func(t *testing.T) {
			omitted, _, _, err := engine.DecodeRequestForMutation(format, []byte(fmt.Sprintf(fixture, "")))
			if err != nil {
				t.Fatal(err)
			}
			null, _, _, err := engine.DecodeRequestForMutation(format, []byte(fmt.Sprintf(fixture, `,"parameters":null`)))
			if err != nil {
				t.Fatalf("null parameters were refused: %v", err)
			}
			if len(null.Tools) != 1 || string(null.Tools[0].InputSchema) != `{"type":"object"}` ||
				null.Tools[0].Strict == nil || *null.Tools[0].Strict {
				t.Fatalf("null parameters decoded as %+v", null.Tools)
			}
			if !reflect.DeepEqual(null.Tools, omitted.Tools) {
				t.Fatalf("null parameters = %+v, omitted parameters = %+v", null.Tools, omitted.Tools)
			}
		})
	}
}

// Bytes a same-format backend would receive by replay still carry the null, so
// the request is encoded instead; an omitted schema keeps the replay.
func TestNullToolParametersAreNotReplayed(t *testing.T) {
	engine := NewBuiltinEngine()
	for format, fixture := range nullToolParameterRequests {
		t.Run(string(format), func(t *testing.T) {
			omitted := []byte(fmt.Sprintf(fixture, ""))
			request, envelope, _, err := engine.DecodeRequest(format, omitted)
			if err != nil {
				t.Fatal(err)
			}
			replayed, err := engine.EncodeRequest(format, request, envelope)
			if err != nil || !bytes.Equal(replayed.Body, omitted) {
				t.Fatalf("omitted parameters were not replayed: error=%v body=%s", err, replayed.Body)
			}

			request, envelope, _, err = engine.DecodeRequest(format, []byte(fmt.Sprintf(fixture, `,"parameters":null`)))
			if err != nil {
				t.Fatal(err)
			}
			encoded, err := engine.EncodeRequest(format, request, envelope)
			if err != nil {
				t.Fatal(err)
			}
			if bytes.Contains(encoded.Body, []byte(`"parameters":null`)) || !bytes.Contains(encoded.Body, []byte(`"parameters":{"type":"object"}`)) {
				t.Fatalf("backend body does not carry the object schema: %s", encoded.Body)
			}
		})
	}
}

func TestNullToolParametersInsideAResponsesNamespace(t *testing.T) {
	body := []byte(`{"model":"m","input":"What time is it?","tools":[{"type":"namespace","name":"mcp__clock","tools":[{"type":"function","name":"get_time","parameters":null}]}]}`)
	request, _, _, err := NewBuiltinEngine().DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, body)
	if err != nil {
		t.Fatalf("null parameters inside a namespace were refused: %v", err)
	}
	if len(request.Tools) != 1 || request.Tools[0].Name != "mcp__clock__get_time" || string(request.Tools[0].InputSchema) != `{"type":"object"}` {
		t.Fatalf("namespaced tool decoded as %+v", request.Tools)
	}
}
