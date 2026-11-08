package systemone

import (
	"encoding/base64"
	"encoding/json"
	"math/rand"
	"strings"
	"testing"
)

// requestModel is the model scanRequest reads.
func requestModel(body []byte) (string, bool) {
	request, ok := scanRequest(body)
	return request.model, ok
}

// referenceOptions is the options member encoding/json reads: the last one.
func referenceOptions(body []byte) ([]byte, bool) {
	var request map[string]json.RawMessage
	if json.Unmarshal(body, &request) != nil || request == nil {
		return nil, false
	}
	options, present := request["options"]
	return options, present
}

func TestScanRequestFindsTheOptionsEncodingJSONReads(t *testing.T) {
	repeated := map[string]bool{`{"options":{"a":1},"options":{"b":2}}`: true, `{"options":1,"opti\u006fns":2}`: true}
	cases := []string{
		`{"model":"m","options":{"return_meta":true}}`, `{"options":null,"model":"m"}`, `{"opti\u006fns":{"a":1}}`,
		`{"a":{"options":{"inner":1}},"model":"m"}`, ` { "options" : { "deadline_ms" : 5 } , "state" : "s" } `,
	}
	for body := range repeated {
		cases = append(cases, body)
	}
	for _, body := range append(cases, bodyCases...) {
		request, ok := scanRequest([]byte(body))
		if !ok {
			continue
		}
		want, present := referenceOptions([]byte(body))
		if present != (request.options != nil) || string(request.options) != string(want) ||
			string([]byte(body)[request.optionsStart:request.optionsEnd]) != string(request.options) {
			t.Fatalf("scanRequest(%q) options %q; encoding/json reads %q", body, request.options, want)
		}
		if request.repeatedOptions != repeated[body] {
			t.Fatalf("scanRequest(%q) repeated options %v", body, request.repeatedOptions)
		}
		if request.options == nil {
			continue
		}
		replaced := request.withOptions([]byte(body), []byte(`{"x":1}`))
		if got, _ := referenceOptions(replaced); string(got) != `{"x":1}` {
			t.Fatalf("withOptions(%q) = %q", body, replaced)
		}
	}
}

// referenceModel is how the public handler read a body before scanRequest.
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
	"{\"model\":\"\xc3\xa9t\xc3\xa9\"}", "{\"model\":\"bad\xff\"}", `{"a":{"model":"inner"},"model":"outer"}`,
	`{"a":{"model":"inner"}}`, `{"a":[1,2,{"b":null}],"model":"m"}`, `{"a":[1,2,]}`, `{"a":[,1]}`, `{"a":01}`,
	`{"a":-0}`, `{"a":-}`, `{"a":1.}`, `{"a":1.5e+10}`, `{"a":1e}`, `{"a":.5}`, `{"a":+1}`, `{"a":1E-2}`,
	`{"a":tr}`, `{"a":truex}`, `{"a":nul}`, `{"a":false,"model":"m"}`, `{"a" : "b" , "model" : "m" }`,
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
