package protocolcodec

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// Chat and Responses decoding keep only the leading system and developer
// messages as instructions. Later ones stay inside the conversation, and
// encoding preserves that order.
func TestOpenAIDecodingKeepsLeadingAndLaterInstructions(t *testing.T) {
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
			[]string{"system", "developer", "user", "system", "assistant", "developer"},
		},
		{
			llmprotocol.OpenAIResponsesV1, `{"model":"m","instructions":"A","input":[
				{"role":"developer","content":"B"},{"role":"user","content":"question"},
				{"role":"system","content":"C"},
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]},
				{"role":"developer","content":"D"}]}`,
			[]string{"developer", "developer", "user", "system", "assistant", "developer"},
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
		if !reflect.DeepEqual(positions, []int{0, 0}) {
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
