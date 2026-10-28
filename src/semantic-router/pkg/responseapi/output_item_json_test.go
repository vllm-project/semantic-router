package responseapi

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestOutputItemReasoningJSONRoundTrip(t *testing.T) {
	for _, wire := range []string{
		`{"type":"reasoning","id":"reason","summary":[],"content":[{"type":"reasoning_text","text":"Inspect the shape."}]}`,
		`{"type":"reasoning","id":"reason","summary":[{"type":"summary_text","text":"The shape is round."}]}`,
		`{"type":"reasoning","id":"reason","summary":[]}`,
	} {
		t.Run(wire, func(t *testing.T) {
			var item OutputItem
			require.NoError(t, json.Unmarshal([]byte(wire), &item))
			for range 2 {
				encoded, err := json.Marshal(item)
				require.NoError(t, err)
				require.JSONEq(t, wire, string(encoded))
				require.NoError(t, json.Unmarshal(encoded, &item))
			}
		})
	}
	encoded, err := json.Marshal(OutputItem{Type: ItemTypeReasoning})
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"reasoning","summary":[]}`, string(encoded))
}

func TestOutputItemOtherTypesKeepTheirJSONShape(t *testing.T) {
	for _, wire := range []string{
		`{"type":"message","id":"answer","role":"assistant","content":[{"type":"output_text","text":"Done.","annotations":[]}],"status":"completed"}`,
		`{"type":"function_call","id":"tool","name":"inspect","call_id":"call","arguments":"{}"}`,
		`{"type":"function_call_output","call_id":"call","output":"done"}`,
	} {
		t.Run(wire, func(t *testing.T) {
			var item OutputItem
			require.NoError(t, json.Unmarshal([]byte(wire), &item))
			encoded, err := json.Marshal(item)
			require.NoError(t, err)
			require.JSONEq(t, wire, string(encoded))
		})
	}
}
