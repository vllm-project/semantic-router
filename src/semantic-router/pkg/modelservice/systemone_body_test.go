package modelservice

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestServedModelIsSplicedIntoTheBodyAsIs(t *testing.T) {
	image := "data:image/png;base64," + strings.Repeat("iVBORw0KGgo", 64)
	for _, tc := range []struct{ body, want string }{
		{`{"model":"a","state":"x"}`, `{"model":"served","state":"x"}`},
		{`{ "state" : {"model": "inner"} , "model" : "a" }`, `{ "state" : {"model": "inner"} , "model" : "served" }`},
		{`{"state":"x"}`, `{"model":"served","state":"x"}`},
		{`{}`, `{"model":"served"}`},
		{" { } ", ` {"model":"served"} `},
		{`{"model":null,"n":[1,2.5e3,true,{"a":[]}]}`, `{"model":"served","n":[1,2.5e3,true,{"a":[]}]}`},
		{`{"state":"a \"}{\" b \\","model":"m","images":["` + image + `"]}`, `{"state":"a \"}{\" b \\","model":"served","images":["` + image + `"]}`},
	} {
		got, ok := withServedModel([]byte(tc.body), "served")
		if !ok || string(got) != tc.want {
			t.Fatalf("%s: got %s (%v), want %s", tc.body, got, ok, tc.want)
		}
		var spliced, rewritten map[string]json.RawMessage
		if err := json.Unmarshal(got, &spliced); err != nil {
			t.Fatalf("%s: %v", got, err)
		}
		if err := json.Unmarshal([]byte(tc.body), &rewritten); err != nil {
			t.Fatal(err)
		}
		rewritten["model"] = json.RawMessage(`"served"`)
		if !reflect.DeepEqual(compactMembers(t, spliced), compactMembers(t, rewritten)) {
			t.Fatalf("%s: splice and rewrite differ", tc.body)
		}
	}
}

func TestBodiesTheSpliceDoesNotReadPlainlyAreRewritten(t *testing.T) {
	for _, body := range []string{
		`{"mod\u0065l":"a"}`,
		`{"model":"a","model":"b"}`,
		`["model"]`,
		`"model"`,
		``,
		`{"model":"a"} x`,
		`{"model":"a"`,
		`{"model" "a"}`,
		`{"state":"unterminated}`,
	} {
		if got, ok := withServedModel([]byte(body), "served"); ok {
			t.Fatalf("%q: spliced to %s", body, got)
		}
	}
}

func compactMembers(t *testing.T, members map[string]json.RawMessage) map[string]string {
	out := make(map[string]string, len(members))
	for key, value := range members {
		var decoded any
		if err := json.Unmarshal(value, &decoded); err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(decoded)
		if err != nil {
			t.Fatal(err)
		}
		out[key] = string(encoded)
	}
	return out
}
