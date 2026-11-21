package protocolcodec

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Chat and Responses decoding read a system or developer message anywhere as
// an instruction and record how many conversation messages came before it.
// Encoding still sends every instruction first.
func TestOpenAIDecodingRecordsWhereEachInstructionWasSent(t *testing.T) {
	engine := NewBuiltinEngine()
	for _, test := range []struct {
		format llmprotocol.WireFormat
		body   string
		roles  []string
	}{
		{
			llmprotocol.OpenAIChatV1, `{"model":"m","messages":[
				{"role":"system","content":"A"},{"role":"developer","content":"B"},
				{"role":"user","content":"question"},{"role":"system","content":"C"},
				{"role":"assistant","content":"answer"},{"role":"developer","content":"D"}]}`,
			[]string{"system", "developer", "system", "developer", "user", "assistant"},
		},
		{
			llmprotocol.OpenAIResponsesV1, `{"model":"m","instructions":"A","input":[
				{"role":"developer","content":"B"},{"role":"user","content":"question"},
				{"role":"system","content":"C"},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
				{"role":"developer","content":"D"}]}`,
			[]string{"developer", "developer", "system", "developer", "user", "assistant"},
		},
	} {
		request, _, _, err := engine.DecodeRequest(test.format, []byte(test.body))
		if err != nil {
			t.Fatalf("%s: %v", test.format, err)
		}
		var positions []int
		for _, instruction := range request.Instructions {
			positions = append(positions, instruction.Position)
		}
		if !reflect.DeepEqual(positions, []int{0, 0, 1, 2}) {
			t.Fatalf("%s: instruction positions = %v", test.format, positions)
		}
		encoded, err := engine.EncodeRequest(llmprotocol.OpenAIChatV1, request, llmprotocol.Envelope{})
		if err != nil {
			t.Fatalf("%s: %v", test.format, err)
		}
		if roles := requestWireMessageRoles(t, llmprotocol.OpenAIChatV1, encoded.Body); !reflect.DeepEqual(roles, test.roles) {
			t.Fatalf("%s: encoded roles = %v: %s", test.format, roles, encoded.Body)
		}
	}
}
