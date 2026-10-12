package handlers

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl/editor"
)

func TestDSLEditorNativeContract(t *testing.T) {
	source := `MODEL qwen { modality: "text" capabilities: ["chat"] }
SIGNAL keyword intent { operator: "any" keywords: ["hello"] }
ROUTE greeting { PRIORITY 1 WHEN keyword("intent") MODEL "qwen" }`
	compiled := editor.Compile(source)
	if compiled.Error != "" {
		t.Fatal(compiled.Error)
	}
	for _, operation := range []string{"compile", "validate", "parse", "decompile", "format"} {
		t.Run(operation, func(t *testing.T) {
			input := source
			if operation == "decompile" {
				input = compiled.YAML
			}
			body, _ := json.Marshal(map[string]string{"source": input})
			response := httptest.NewRecorder()
			DSLEditorHandler(operation)(response, httptest.NewRequest("POST", "/api/dsl/"+operation, strings.NewReader(string(body))))
			if response.Code != 200 {
				t.Fatal(response.Code, response.Body.String())
			}
			var result map[string]any
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result["error"] != nil {
				t.Fatal(result)
			}
			if operation == "compile" && result["yaml"] != compiled.YAML {
				t.Fatal("native and WASM adapter differ")
			}
			if operation == "parse" && result["ast"] == nil {
				t.Fatal("missing AST")
			}
			if operation == "validate" && result["symbols"] == nil {
				t.Fatal("missing symbols")
			}
		})
	}
}

func TestDSLEditorBoundsAndDiagnostics(t *testing.T) {
	for _, body := range []string{`{}`, `{"source":null}`, `{"source":"", "url":"http://untrusted"}`, `{"source":""}{}`, `{"source":"` + strings.Repeat("x", maxDSLSourceBytes+1) + `"}`} {
		response := httptest.NewRecorder()
		DSLEditorHandler("parse")(response, httptest.NewRequest("POST", "/api/dsl/parse", strings.NewReader(body)))
		if response.Code != 400 {
			t.Fatal("unbounded or invalid request accepted", response.Code)
		}
	}
	response := httptest.NewRecorder()
	DSLEditorHandler("parse")(response, httptest.NewRequest("POST", "/api/dsl/parse", strings.NewReader(`{"source":"INVALID !!!"}`)))
	if response.Code != 200 || !strings.Contains(response.Body.String(), `"errorCount"`) {
		t.Fatal("editor diagnostics lost")
	}
	for range cap(dslEditorSlots) {
		dslEditorSlots <- struct{}{}
	}
	defer func() {
		for range cap(dslEditorSlots) {
			<-dslEditorSlots
		}
	}()
	response = httptest.NewRecorder()
	DSLEditorHandler("parse")(response, httptest.NewRequest("POST", "/api/dsl/parse", strings.NewReader(`{"source":""}`)))
	if response.Code != 503 {
		t.Fatal("unbounded compilation concurrency")
	}
}
