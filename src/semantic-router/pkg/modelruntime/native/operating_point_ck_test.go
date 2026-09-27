package native

import (
	"path/filepath"
	"slices"
	"strings"
	"testing"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/operatingpoint"
)

func TestOperatingPointCKBindsActualRuntimeLibraryAndDynamicSchema(t *testing.T) {
	policy, spec := ortPolicyFixture(t)
	graph := policy.ONNX()
	graph.ExecutionProvider = "ROCMExecutionProvider"
	graph.CustomOpsProfile = "ck_flash_attention"
	graph.ExecutionMode = "dynamic_sequence_b1"
	graph.RuntimeBuild = "qualified ORT build"
	graph.MaxExecutionTokens = 32768
	librarySHA := strings.Repeat("f", 64)
	graph.Artifacts = append(graph.Artifacts, operatingpoint.ArtifactDigest{Role: "custom-ops", SHA256: librarySHA})
	evidence := ort.SessionEvidence{
		Graph: filepath.Join(spec.Deployment.Artifact, graph.File), Provider: graph.ExecutionProvider,
		RuntimeBuild: graph.RuntimeBuild, Precision: "native", CPUFallbackDisabled: true,
		ExecutionMaxInputTokens: 32768, CustomOpsProfile: graph.CustomOpsProfile,
		CustomOpsLibrary: operatingpoint.CKFlashAttentionLibrary, CustomOpsSHA256: librarySHA,
		Artifacts:   []ort.ArtifactDigest{{Role: "graph", SHA256: graph.Artifacts[0].SHA256}, {Role: "custom-ops", SHA256: librarySHA}},
		InputSchema: []ort.ExecutionInput{{Name: "input_ids", Dtype: "int64", Shape: []int64{-1, -1}}, {Name: "attention_mask", Dtype: "int64", Shape: []int64{-1, -1}}},
	}
	validate := func(session ort.SessionEvidence) error {
		return validateOperatingPointGraphSession(graph, spec, ort.Info{Task: "label_scores", Sessions: []ort.SessionEvidence{session}})
	}
	if err := validate(evidence); err != nil {
		t.Fatal(err)
	}
	fixedBatch := cloneCKEvidence(evidence)
	for i := range fixedBatch.InputSchema {
		fixedBatch.InputSchema[i].Shape[0] = 1
	}
	fixedBatch.InputSchema = append(fixedBatch.InputSchema, ort.ExecutionInput{Name: "position_ids", Dtype: "int64", Shape: []int64{1, -1}})
	if err := validate(fixedBatch); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*ort.SessionEvidence){
		"different runtime":          func(s *ort.SessionEvidence) { s.RuntimeBuild += " changed" },
		"different provider":         func(s *ort.SessionEvidence) { s.Provider = "CPUExecutionProvider" },
		"CPU fallback":               func(s *ort.SessionEvidence) { s.CPUFallbackDisabled = false },
		"different profile":          func(s *ort.SessionEvidence) { s.CustomOpsProfile = "none" },
		"different library path":     func(s *ort.SessionEvidence) { s.CustomOpsLibrary = "/tmp/untrusted.so" },
		"different library SHA":      func(s *ort.SessionEvidence) { s.CustomOpsSHA256 = strings.Repeat("a", 64) },
		"different captured library": func(s *ort.SessionEvidence) { s.Artifacts[1].SHA256 = strings.Repeat("a", 64) },
		"missing captured library":   func(s *ort.SessionEvidence) { s.Artifacts = s.Artifacts[:1] },
		"duplicate captured library": func(s *ort.SessionEvidence) { s.Artifacts[0] = s.Artifacts[1] },
		"different graph":            func(s *ort.SessionEvidence) { s.Artifacts[0].SHA256 = strings.Repeat("a", 64) },
		"different forward limit":    func(s *ort.SessionEvidence) { s.ExecutionMaxInputTokens = 65536 },
		"no actual schema":           func(s *ort.SessionEvidence) { s.InputSchema = nil },
		"missing mask":               func(s *ort.SessionEvidence) { s.InputSchema = s.InputSchema[:1] },
		"duplicate input":            func(s *ort.SessionEvidence) { s.InputSchema[1] = s.InputSchema[0] },
		"unknown input":              func(s *ort.SessionEvidence) { s.InputSchema[1].Name = "other" },
		"non-int64 input":            func(s *ort.SessionEvidence) { s.InputSchema[0].Dtype = "int32" },
		"wrong rank":                 func(s *ort.SessionEvidence) { s.InputSchema[0].Shape = []int64{-1} },
		"fixed window":               func(s *ort.SessionEvidence) { s.InputSchema[0].Shape[1] = 32768 },
		"oversize fixed padding":     func(s *ort.SessionEvidence) { s.InputSchema[0].Shape[1] = 65536 },
		"fixed B4":                   func(s *ort.SessionEvidence) { s.InputSchema[0].Shape[0] = 4 },
		"zero dimension":             func(s *ort.SessionEvidence) { s.InputSchema[0].Shape[1] = 0 },
		"dynamic position batch": func(s *ort.SessionEvidence) {
			s.InputSchema = append(s.InputSchema, ort.ExecutionInput{Name: "position_ids", Dtype: "int64", Shape: []int64{-1, -1}})
		},
		"fixed execution masquerading as dynamic": func(s *ort.SessionEvidence) {
			s.ExecutionInputs = []ort.ExecutionInput{{Name: "input_ids", Dtype: "int64", Shape: []int64{1, 32768}}}
		},
	} {
		t.Run(name, func(t *testing.T) {
			bad := cloneCKEvidence(evidence)
			mutate(&bad)
			if err := validate(bad); err == nil {
				t.Fatal("unqualified actual CK execution accepted")
			}
		})
	}
}

func cloneCKEvidence(value ort.SessionEvidence) ort.SessionEvidence {
	value.Artifacts = slices.Clone(value.Artifacts)
	value.InputSchema = slices.Clone(value.InputSchema)
	for i := range value.InputSchema {
		value.InputSchema[i].Shape = slices.Clone(value.InputSchema[i].Shape)
	}
	return value
}
