package systemone

import (
	"encoding/base64"
	"encoding/json"
	"math/rand"
	"strings"
	"testing"
)

// referenceModel is how the public handler read a body before requestModel.
func referenceModel(body []byte) (string, bool) {
	var request map[string]json.RawMessage
	if json.Unmarshal(body, &request) != nil || request == nil {
		return "", false
	}
	var model string
	if json.Unmarshal(request["model"], &model) != nil {
		return "", true
	}
	return model, true
}

var bodyCases = []string{
	``, ` `, `{}`, ` { } `, `null`, `[]`, `"x"`, `1`, `true`, `{`, `}`, `{"model"}`, `{"model":}`,
	`{"model":"a"}`, `{"model":"a",}`, `{"model":"a"}x`, `{"model":"a"} `, "\t{\"model\":\"a\"}\n",
	`{"model":"a","model":"b"}`, `{"model":"a","model":null}`, `{"model":null}`, `{"model":1}`, `{"model":""}`,
	`{"model":["a"]}`, `{"model":{"x":1}}`, `{"mod\u0065l":"esc"}`, `{"model":"\u0061\u00e9\ud83d\ude00"}`,
	`{"model":"a\"b"}`, `{"model":"a\\"}`, `{"model":"a\\\"b"}`, `{"model":"a\/b\b\f\n\r\t"}`, `{"model":"a\x"}`,
	`{"model":"a\u12"}`, `{"model":"a\u12g4"}`, "{\"model\":\"a\x01b\"}", "{\"model\":\"a\x7fb\"}",
	"{\"model\":\"caf\xc3\xa9\"}", "{\"model\":\"bad\xff\"}", `{"a":{"model":"inner"},"model":"outer"}`,
	`{"a":{"model":"inner"}}`, `{"a":[1,2,{"b":null}],"model":"m"}`, `{"a":[1,2,]}`, `{"a":[,1]}`, `{"a":01}`,
	`{"a":-0}`, `{"a":-}`, `{"a":1.}`, `{"a":1.5e+10}`, `{"a":1e}`, `{"a":.5}`, `{"a":+1}`, `{"a":1E-2}`,
	`{"a":tru}`, `{"a":truex}`, `{"a":nul}`, `{"a":false,"model":"m"}`, `{"a" : "b" , "model" : "m" }`,
	`{"a":"b" "model":"m"}`, `{"a":"b",,"model":"m"}`, `{1:"b"}`, `{"a":"unterminated}`, `{"a":"\`,
	`{"model":"m"}{}`, `{"model":"m"}]`, "\ufeff{\"model\":\"m\"}", `{"images":["data:image/png;base64,AAAA"],"model":"m"}`,
}

func TestRequestModelMatchesEncodingJSON(t *testing.T) {
	check := func(body []byte) {
		t.Helper()
		gotModel, gotValid := requestModel(body)
		wantModel, wantValid := referenceModel(body)
		if gotModel != wantModel || gotValid != wantValid {
			t.Fatalf("requestModel(%q) = %q, %v; encoding/json gives %q, %v", body, gotModel, gotValid, wantModel, wantValid)
		}
	}
	for _, body := range bodyCases {
		check([]byte(body))
	}
	deep := strings.Repeat("[", maxNesting-1) + strings.Repeat("]", maxNesting-1)
	check([]byte(`{"a":` + deep + `,"model":"m"}`))
	check([]byte(`{"a":[` + deep + `],"model":"m"}`))
	rng := rand.New(rand.NewSource(1))
	alphabet := []byte("{}[]\":,\\ \t\nabmodelu0123456789.eE+-tfn\x01\xff")
	for _, seed := range bodyCases {
		for range 300 {
			body := []byte(seed)
			for range 1 + rng.Intn(3) {
				switch op := rng.Intn(3); {
				case op == 0 && len(body) > 0:
					body[rng.Intn(len(body))] = alphabet[rng.Intn(len(alphabet))]
				case op == 1:
					at := rng.Intn(len(body) + 1)
					body = append(body[:at], append([]byte{alphabet[rng.Intn(len(alphabet))]}, body[at:]...)...)
				case len(body) > 0:
					at := rng.Intn(len(body))
					body = append(body[:at], body[at+1:]...)
				}
			}
			check(body)
		}
	}
}

func FuzzRequestModelMatchesEncodingJSON(f *testing.F) {
	for _, body := range bodyCases {
		f.Add([]byte(body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		gotModel, gotValid := requestModel(body)
		wantModel, wantValid := referenceModel(body)
		if gotModel != wantModel || gotValid != wantValid {
			t.Fatalf("requestModel(%q) = %q, %v; encoding/json gives %q, %v", body, gotModel, gotValid, wantModel, wantValid)
		}
	})
}

func mediaBody(size int) []byte {
	raw := make([]byte, size)
	rand.New(rand.NewSource(2)).Read(raw)
	media := "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(raw)
	body, _ := json.Marshal(map[string]any{
		"model":     "vllm-sr/d3",
		"state":     "What happens in the clip?",
		"videos":    []string{media},
		"questions": map[string]any{"moving": map[string]any{"type": "noul", "instructions": "Does something move?"}},
	})
	return body
}

func BenchmarkRequestModelVideoBody(b *testing.B) {
	body := mediaBody(5 << 20)
	b.SetBytes(int64(len(body)))
	for b.Loop() {
		if _, ok := requestModel(body); !ok {
			b.Fatal("invalid")
		}
	}
}

func BenchmarkEncodingJSONVideoBody(b *testing.B) {
	body := mediaBody(5 << 20)
	b.SetBytes(int64(len(body)))
	for b.Loop() {
		if _, ok := referenceModel(body); !ok {
			b.Fatal("invalid")
		}
	}
}
