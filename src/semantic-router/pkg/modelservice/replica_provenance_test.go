package modelservice

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestReplicaAliasesPreserveObservedNativeProvenance(t *testing.T) {
	meta := `{"revision":"revision-a","model_sha256":"` + strings.Repeat("a", 64) + `","engine":"native","profile":"exact","numerics":"exact","accelerator":"cpu"}`
	answers := `{"meta":{"type":"noul","noul":0.9,"extension":{"meta":"answer data"}}}`
	native := `{"model":"native-artifact","meta":` + meta + `,"answers":` + answers + `,"states":{"meta":{"model":"native-artifact","meta":` + meta + `,"answers":` + answers + `}}}`
	for _, bundle := range []bool{false, true} {
		t.Run(map[bool]string{false: "native", true: "bundle"}[bundle], func(t *testing.T) {
			body := []byte(native)
			if bundle {
				body = []byte(`{"results":[{"decisions":` + native + `}]}`)
			}
			for _, alias := range []string{"pool-alias", "public-alias"} {
				var err error
				body, err = rewriteResponseModel(body, alias)
				if err != nil {
					t.Fatal(err)
				}
			}
			var object map[string]json.RawMessage
			if err := json.Unmarshal(body, &object); err != nil {
				t.Fatal(err)
			}
			if bundle {
				var results []map[string]json.RawMessage
				_ = json.Unmarshal(object["results"], &results)
				_ = json.Unmarshal(results[0]["decisions"], &object)
			}
			envelopes := []map[string]json.RawMessage{object}
			var states map[string]map[string]json.RawMessage
			_ = json.Unmarshal(object["states"], &states)
			envelopes = append(envelopes, states["meta"])
			for _, envelope := range envelopes {
				var provenance map[string]json.RawMessage
				_ = json.Unmarshal(envelope["meta"], &provenance)
				if string(envelope["model"]) != `"public-alias"` || string(provenance["model_id"]) != `"native-artifact"` {
					t.Fatalf("identity lost through alias hops: %s", body)
				}
				if string(envelope["answers"]) != answers {
					t.Fatalf("answer payload changed: %s", body)
				}
			}
		})
	}
}
