package responseapi

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGeneratedImageItemJSONRoundTrip(t *testing.T) {
	for _, result := range []string{`"aW1hZ2U="`, `""`, `null`} {
		for _, direction := range []string{"input", "output"} {
			t.Run(direction+"/"+result, func(t *testing.T) {
				wire := `{"type":"image_generation_call","id":"ig_1","status":"completed","result":` + result + `}`
				var item any = &InputItem{}
				if direction == "output" {
					item = &OutputItem{}
				}
				for range 2 {
					require.NoError(t, json.Unmarshal([]byte(wire), item))
					encoded, err := json.Marshal(item)
					require.NoError(t, err)
					require.JSONEq(t, wire, string(encoded))
					wire = string(encoded)
				}
			})
		}
	}
}
