package protocolcodec

import (
	"bytes"
	"encoding/json"
	"os"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

var responsesDefaultCases = []struct {
	path    string
	value   string
	invalid []string
}{
	{"kv_transfer_params", "null", []string{`{}`}},
	{"ec_transfer_params", "null", []string{`{}`}},
	{"input_messages", "null", []string{`[]`}},
	{"output_messages", "null", []string{`[]`}},
	{"usage.input_tokens_details.input_tokens_per_turn", "[]", []string{`null`, `[2]`, `{}`}},
	{"usage.input_tokens_details.cached_tokens_per_turn", "[]", []string{`null`}},
	{"usage.output_tokens_details.output_tokens_per_turn", "[]", []string{`null`}},
	{"usage.output_tokens_details.tool_output_tokens_per_turn", "[]", []string{`null`}},
	{"usage.output_tokens_details.tool_output_tokens", "0", []string{`null`, `1`, `0.0`, `0e0`, `1e-400`}},
}

func TestResponsesProviderDefaultBoundaries(t *testing.T) {
	base := responsesDefaultsFixture(t)
	for _, test := range responsesDefaultCases {
		t.Run(test.path, func(t *testing.T) {
			for _, value := range append([]string{test.value}, test.invalid...) {
				t.Run(value, func(t *testing.T) {
					body := setResponsesDefault(t, base, test.path, value)
					for _, vendor := range []llmprotocol.ResponseVendor{"", llmprotocol.ResponseVendorOpenAICompatGateway} {
						policy := llmprotocol.DefaultPolicy()
						policy.ResponseVendor = vendor
						response, envelope, diagnostics, err := (OpenAIResponsesCodec{}).DecodeResponse(body, policy)
						if value != test.value && vendor == "" {
							if err == nil {
								t.Fatal("strict decoder accepted a non-default extension")
							}
							continue
						}
						if err != nil {
							t.Fatalf("vendor=%q: %v", vendor, err)
						}
						if response.Usage.InputTotal.Value == nil || *response.Usage.InputTotal.Value != 2 || response.Usage.OutputTotal.Value == nil || *response.Usage.OutputTotal.Value != 1 || response.Usage.Total.Value == nil || *response.Usage.Total.Value != 3 {
							t.Fatalf("usage changed: %+v", response.Usage)
						}
						encoded, _, err := (OpenAIResponsesCodec{}).EncodeResponse(response, envelope, policy)
						if err != nil {
							t.Fatal(err)
						}
						if vendor == "" {
							if !bytes.Equal(body, encoded) {
								t.Fatal("strict same-format source replay changed")
							}
						} else {
							if bytes.Contains(encoded, []byte(`"`+lastResponsesPathPart(test.path)+`"`)) {
								t.Fatal("compat retained extension")
							}
							if !DiagnosticsDroppedVendorExtensions(diagnostics) || !responsesDefaultDiagnostic(diagnostics, test.path) {
								t.Fatalf("missing compat diagnostic: %+v", diagnostics)
							}
						}
					}
				})
			}
		})
	}
	body := setResponsesDefault(t, base, "usage.output_tokens_details.tool_output_tokens", "-0")
	if _, _, _, err := (OpenAIResponsesCodec{}).DecodeResponse(body, llmprotocol.DefaultPolicy()); err != nil {
		t.Fatalf("integer negative zero: %v", err)
	}
}

func TestResponsesProviderDefaultsStreamResources(t *testing.T) {
	for _, phase := range []string{"response.created", "response.completed"} {
		for _, field := range []struct{ path, value string }{
			{"kv_transfer_params", "{}"},
			{"usage.input_tokens_details.input_tokens_per_turn", "null"},
			{"usage.output_tokens_details.tool_output_tokens", "1"},
		} {
			t.Run(phase+"/"+field.path, func(t *testing.T) {
				payload := patchResponsesDefaultStream(t, phase, field.path, field.value)
				for _, vendor := range []llmprotocol.ResponseVendor{"", llmprotocol.ResponseVendorOpenAICompatGateway} {
					policy := llmprotocol.DefaultPolicy()
					policy.ResponseVendor = vendor
					engine, err := NewEngine(NewBuiltinRegistry(), policy)
					if err != nil {
						t.Fatal(err)
					}
					stream, err := engine.NewStream(llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{Context: t.Context(), PublicModel: "public-model"})
					if err != nil {
						t.Fatal(err)
					}
					_, _, diagnostics, err := stream.Push(payload)
					if vendor == "" {
						if err == nil {
							t.Fatal("stream accepted non-default extension")
						}
						continue
					}
					if err != nil {
						t.Fatal(err)
					}
					if _, _, _, err := stream.Finalize(nil); err != nil {
						t.Fatal(err)
					}
					if !DiagnosticsDroppedVendorExtensions(diagnostics) {
						t.Fatal("missing stream vendor diagnostic")
					}
				}
			})
		}
	}
}

func responsesDefaultsFixture(t *testing.T) []byte {
	t.Helper()
	body, err := os.ReadFile("testdata/golden/response/002-responses-basic-in.json")
	if err != nil {
		t.Fatal(err)
	}
	return body
}

func setResponsesDefault(t *testing.T, body []byte, path, value string) []byte {
	t.Helper()
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		t.Fatal(err)
	}
	name, rest, nested := strings.Cut(path, ".")
	if nested {
		child := object[name]
		if len(child) == 0 {
			child = []byte(`{}`)
		}
		object[name] = setResponsesDefault(t, child, rest, value)
	} else {
		object[name] = json.RawMessage(value)
	}
	result, err := json.Marshal(object)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func patchResponsesDefaultStream(t *testing.T, phase, path, value string) []byte {
	t.Helper()
	lines := strings.Split(string(streamFixture(llmprotocol.OpenAIResponsesV1)), "\n")
	for index, line := range lines {
		if !strings.HasPrefix(line, "data: ") || !strings.Contains(line, `"type":"`+phase+`"`) {
			continue
		}
		lines[index] = "data: " + string(setResponsesDefault(t, []byte(strings.TrimPrefix(line, "data: ")), "response."+path, value))
	}
	return []byte(strings.Join(lines, "\n"))
}

func lastResponsesPathPart(path string) string {
	parts := strings.Split(path, ".")
	return parts[len(parts)-1]
}

func responsesDefaultDiagnostic(diagnostics llmprotocol.Diagnostics, field string) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == field && diagnostic.Action == llmprotocol.DiagnosticDropped {
			return true
		}
	}
	return false
}
