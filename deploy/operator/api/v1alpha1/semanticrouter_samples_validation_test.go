package v1alpha1

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	apiextensions "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema/defaulting"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	kubejson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"sigs.k8s.io/yaml"
)

// The shipped samples are executable CRs, not snippets. Decode them strictly
// (so a field that is absent from the API type is refused) and also run them
// through the generated CRD's OpenAPI validator (so enum and pattern
// constraints are enforced). This is the admission-time check without a
// cluster: #7486 shipped three samples whose toolsDb entries were rejected by
// strict Kubernetes validation before the Operator could reconcile them.
func TestOperatorSamplesValidateAgainstCRD(t *testing.T) {
	schemas := sampleCRDSchemas(t)
	paths, err := filepath.Glob(filepath.Join("..", "..", "config", "samples", "*.yaml"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("list samples: %v (paths: %v)", err, paths)
	}

	const wantDocuments = 14
	checked := 0
	seen := map[string]bool{}
	for _, path := range paths {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("read %s: %v", path, err)
		}
		reader := yamlutil.NewYAMLReader(bufio.NewReader(bytes.NewReader(data)))
		for {
			doc, err := reader.Read()
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				t.Fatalf("split %s: %v", path, err)
			}
			var probe struct {
				Kind string `json:"kind"`
			}
			if err := yaml.Unmarshal(doc, &probe); err != nil {
				t.Fatalf("parse doc in %s: %v", path, err)
			}
			if probe.Kind != "SemanticRouter" {
				continue
			}
			checked++
			seen[filepath.Base(path)] = true
			for _, failure := range validateSampleDoc(schemas, doc) {
				t.Errorf("%s: %s", path, failure)
			}
			var sample SemanticRouter
			if err := yaml.UnmarshalStrict(doc, &sample); err != nil {
				// validateSampleDoc already reported this error. Do not assert
				// tool shape against a partially decoded document, and make the
				// skipped assertions visible instead of silently omitting them.
				t.Logf("%s: skipping tool-shape assertions: %v", path, err)
				continue
			}
			assertSampleToolEntries(t, path, sample.Spec.ToolsDb)
		}
	}
	if checked != wantDocuments {
		t.Fatalf("validated %d SemanticRouter documents, want %d", checked, wantDocuments)
	}
	for _, name := range []string{
		"vllm.ai_v1alpha1_semanticrouter_llamastack.yaml",
		"vllm.ai_v1alpha1_semanticrouter_route.yaml",
		"vllm.ai_v1alpha1_semanticrouter_openshift.yaml",
	} {
		if !seen[name] {
			t.Errorf("sample %s was not validated", name)
		}
	}
}

// A tool entry's name and JSON schema belong to tool.function, and the only
// accepted discriminator is "function". The CRD makes both fields optional,
// so schema validation alone would let an empty entry through.
func assertSampleToolEntries(t *testing.T, path string, tools []ToolEntry) {
	t.Helper()
	for i, tool := range tools {
		if tool.Tool.Type != "function" {
			t.Errorf("%s: toolsDb[%d].tool.type = %q, want function", path, i, tool.Tool.Type)
		}
		if tool.Tool.Function.Name == "" {
			t.Errorf("%s: toolsDb[%d].tool.function.name is empty", path, i)
		}
		if tool.Tool.Function.Parameters.Type == "" {
			t.Errorf("%s: toolsDb[%d].tool.function.parameters.type is empty", path, i)
		}
	}
}

type sampleCRDSchema struct {
	name       string
	validator  validation.SchemaValidator
	structural *structuralschema.Structural
}

// Build one validator per served CRD version. The current CRD has a single
// v1alpha1 version, but iterating avoids silently ignoring a new version.
func sampleCRDSchemas(t *testing.T) []sampleCRDSchema {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "config", "crd", "bases", "vllm.ai_semanticrouters.yaml"))
	if err != nil {
		t.Fatalf("read CRD: %v", err)
	}
	var crd apiextensionsv1.CustomResourceDefinition
	if err := yaml.Unmarshal(data, &crd); err != nil {
		t.Fatalf("parse CRD: %v", err)
	}
	var schemas []sampleCRDSchema
	for _, version := range crd.Spec.Versions {
		if version.Schema == nil {
			continue
		}
		var internal apiextensions.JSONSchemaProps
		if err := apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(version.Schema.OpenAPIV3Schema, &internal, nil); err != nil {
			t.Fatalf("convert schema %s: %v", version.Name, err)
		}
		validator, _, err := validation.NewSchemaValidator(&internal)
		if err != nil {
			t.Fatalf("schema validator %s: %v", version.Name, err)
		}
		structural, err := structuralschema.NewStructural(&internal)
		if err != nil {
			t.Fatalf("structural schema %s: %v", version.Name, err)
		}
		schemas = append(schemas, sampleCRDSchema{name: version.Name, validator: validator, structural: structural})
	}
	if len(schemas) == 0 {
		t.Fatal("CRD has no versioned schema")
	}
	return schemas
}

func validateSampleDoc(schemas []sampleCRDSchema, doc []byte) []string {
	var failures []string
	var strict SemanticRouter
	if err := yaml.UnmarshalStrict(doc, &strict); err != nil {
		failures = append(failures, err.Error())
	}
	raw, err := yaml.YAMLToJSON(doc)
	if err != nil {
		return append(failures, "YAMLToJSON: "+err.Error())
	}
	var obj map[string]interface{}
	if err := kubejson.Unmarshal(raw, &obj); err != nil {
		return append(failures, "decode CR: "+err.Error())
	}
	for _, schema := range schemas {
		// The API server removes explicit nulls for non-nullable fields that
		// have no default before it runs CRD validation. A sample with
		// "annotations:" and only a comment beneath it is therefore valid;
		// mirror that pruning so the test judges the same object the API
		// server would judge.
		defaulting.PruneNonNullableNullsWithoutDefaults(obj, schema.structural)
		for _, err := range validation.ValidateCustomResource(field.NewPath(""), obj, schema.validator) {
			failures = append(failures, fmt.Sprintf("%s: %s", schema.name, err.Error()))
		}
	}
	return failures
}

const sampleCRHeader = `apiVersion: vllm.ai/v1alpha1
kind: SemanticRouter
metadata:
  name: r
spec:
`

// The exact shapes the three samples used before the fix, plus the two
// tracing fields the main OpenShift sample had out of date.
func TestCRDRejectsLegacySampleShapes(t *testing.T) {
	schemas := sampleCRDSchemas(t)
	cases := []struct {
		name string
		spec string
		want string
	}{
		{
			name: "old toolsDb name field",
			spec: `  toolsDb:
    - name: web_search
      description: Search the web
`,
			want: `unknown field "name"`,
		},
		{
			name: "old toolsDb parameters field",
			spec: `  toolsDb:
    - description: Search the web
      parameters:
        type: object
`,
			want: `unknown field "parameters"`,
		},
		{
			name: "toolsDb type other than function",
			spec: `  toolsDb:
    - tool:
        type: remote
        function:
          name: web_search
`,
			want: `spec.toolsDb[0].tool.type: Unsupported value: "remote": supported values: "function"`,
		},
		{
			name: "tracing sampling probability",
			spec: `  config:
    observability:
      tracing:
        sampling:
          probability: "0.1"
`,
			want: `unknown field "probability"`,
		},
		{
			name: "tracing resource environment",
			spec: `  config:
    observability:
      tracing:
        resource:
          environment: production
`,
			want: `unknown field "environment"`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			failures := validateSampleDoc(schemas, []byte(sampleCRHeader+tc.spec))
			if len(failures) == 0 {
				t.Fatal("malformed sample was admitted")
			}
			if !strings.Contains(strings.Join(failures, "\n"), tc.want) {
				t.Fatalf("want a failure mentioning %q, got %v", tc.want, failures)
			}
		})
	}
}
