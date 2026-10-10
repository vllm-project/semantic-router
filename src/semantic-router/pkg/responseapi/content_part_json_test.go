package responseapi

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestContentPartRetainsTypeRequiredFields(t *testing.T) {
	for _, body := range []string{
		`{"type":"input_text","text":""}`,
		`{"type":"input_text","text":"ordinary question"}`,
		`{"type":"output_text","text":"","annotations":[]}`,
		`{"type":"output_text","text":"ordinary answer","annotations":[]}`,
		`{"type":"refusal","refusal":""}`,
		`{"type":"refusal","refusal":"No answer available"}`,
		`{"type":"summary_text","text":""}`,
		`{"type":"reasoning_text","text":""}`,
		`{"type":"input_image","image_url":"https://example.com/image.png","detail":"auto"}`,
		`{"type":"input_file","file_id":"file_sample"}`,
	} {
		t.Run(body, func(t *testing.T) {
			var part ContentPart
			require.NoError(t, json.Unmarshal([]byte(body), &part))
			encoded, err := json.Marshal(part)
			require.NoError(t, err)
			require.JSONEq(t, body, string(encoded))
		})
	}
}

func TestContentPartCanonicalizesMissingOutputAnnotations(t *testing.T) {
	part := ContentPart{Type: "output_text", Text: "ordinary answer"}
	encoded, err := json.Marshal(part)
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"output_text","text":"ordinary answer","annotations":[]}`, string(encoded))
	require.Nil(t, part.Annotations, "serialization must not mutate the stored value")
}
