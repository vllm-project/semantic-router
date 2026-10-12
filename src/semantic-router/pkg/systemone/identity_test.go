package systemone

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestResponseIdentityUsesObservedModelAcrossAliases(t *testing.T) {
	identity := nativeIdentity("native-artifact")
	for _, tc := range []struct {
		name, model    string
		explicit       any
		include, valid bool
	}{
		{name: "raw native", model: "native-artifact", valid: true},
		{name: "public alias", model: "public-alias", explicit: "native-artifact", include: true, valid: true},
		{name: "empty provenance", model: "native-artifact", explicit: "", include: true},
		{name: "null provenance", model: "native-artifact", explicit: nil, include: true},
		{name: "malformed provenance", model: "native-artifact", explicit: 3, include: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var meta map[string]any
			encoded, _ := json.Marshal(identity)
			_ = json.Unmarshal(encoded, &meta)
			delete(meta, "model_id")
			if tc.include {
				meta["model_id"] = tc.explicit
			}
			body, _ := json.Marshal(map[string]any{"model": tc.model, "meta": meta})
			got, ok := responseInferenceIdentity(body)
			if ok != tc.valid || ok && got != identity {
				t.Fatalf("identity=%+v known=%v", got, ok)
			}
		})
	}
}

func TestResponseIdentityRejectsConflictingStateProvenance(t *testing.T) {
	identity := nativeIdentity("native-artifact")
	changed := identity
	changed.Revision = "different"
	body, _ := json.Marshal(map[string]any{"model": "public-alias", "meta": identity, "states": map[string]any{"second": map[string]any{"model": "public-alias", "meta": changed}}})
	if _, ok := responseInferenceIdentity(body); ok {
		t.Fatal("conflicting state provenance accepted")
	}
}

func nativeIdentity(model string) InferenceIdentity {
	return InferenceIdentity{ModelID: model, Revision: strings.Repeat("a", 40), ModelSHA256: strings.Repeat("b", 64), Engine: "native", Profile: "exact", Numerics: "exact", Accelerator: "cpu"}
}
