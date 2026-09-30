package native

import (
	"fmt"
	"path/filepath"
	"slices"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/operatingpoint"
)

// Validate evidence from the actual owned session before exposing score windows.
// A matching filename or successfully registered EP alone is insufficient.
func validateOperatingPointSession(policy *operatingpoint.Policy, spec config.ResolvedModelBinding, info ort.Info) error {
	return validateOperatingPointGraphSession(policy.ONNX(), spec, info)
}

func validateOperatingPointGraphSession(graph *operatingpoint.ONNXExecution, spec config.ResolvedModelBinding, info ort.Info) error {
	if graph == nil || len(info.Sessions) != 1 || info.Task != "label_scores" {
		return fmt.Errorf("%w: operating point requires exactly one owned label-score graph", binding.ErrCapability)
	}
	session := info.Sessions[0]
	want, err := filepath.EvalSymlinks(filepath.Join(spec.Deployment.Artifact, graph.File))
	if err != nil {
		return err
	}
	actual, err := filepath.EvalSymlinks(session.Graph)
	if err != nil {
		return err
	}
	want, err = filepath.Abs(want)
	if err != nil {
		return err
	}
	actual, err = filepath.Abs(actual)
	if err != nil {
		return err
	}
	if actual != want || session.RuntimeBuild == "" || session.Provider != graph.ExecutionProvider || session.Precision != "native" || session.ExecutionMaxInputTokens != graph.MaxExecutionTokens {
		return fmt.Errorf("%w: actual ONNX execution differs from operating point", binding.ErrCapability)
	}
	if graph.CustomOpsProfile != "" {
		if err := validateOperatingPointCKSession(graph, session); err != nil {
			return err
		}
	} else if (session.CustomOpsProfile != "" && session.CustomOpsProfile != "none") || session.CustomOpsLibrary != "" || session.CustomOpsSHA256 != "" {
		return fmt.Errorf("%w: operating point does not qualify custom operators", binding.ErrCapability)
	}
	if len(session.Artifacts) != len(graph.Artifacts) {
		return fmt.Errorf("%w: incomplete actual graph/external tensor identities", binding.ErrCapability)
	}
	wanted := make(map[string]string, len(graph.Artifacts))
	for _, artifact := range graph.Artifacts {
		wanted[artifact.Role] = artifact.SHA256
	}
	for _, artifact := range session.Artifacts {
		if wanted[artifact.Role] != artifact.SHA256 || artifact.SHA256 == "" {
			return fmt.Errorf("%w: actual graph/external tensor identity differs", binding.ErrCapability)
		}
		delete(wanted, artifact.Role)
	}
	if len(wanted) != 0 {
		return fmt.Errorf("%w: duplicate actual graph artifact", binding.ErrCapability)
	}
	if session.Provider == "CPUExecutionProvider" {
		if len(session.ExecutionInputs) != 0 {
			return fmt.Errorf("%w: CPU operating point expects dynamic execution within its window capacity", binding.ErrCapability)
		}
		return nil
	}
	if !session.CPUFallbackDisabled {
		return fmt.Errorf("%w: operating point forbids CPU fallback", binding.ErrCapability)
	}
	if session.Provider == "ROCMExecutionProvider" {
		// CK validation above attests dynamic declared inputs and a bounded B1
		// executor. Do not reinterpret MIGraphX's fixed ExecutionInputs evidence.
		if graph.CustomOpsProfile == "" {
			return fmt.Errorf("%w: ROCm operating point requires a qualified custom-op execution", binding.ErrCapability)
		}
		return nil
	}
	names := map[string]bool{}
	for _, input := range session.ExecutionInputs {
		if names[input.Name] || (input.Name != "input_ids" && input.Name != "attention_mask" && input.Name != "position_ids") || input.Dtype != "int64" || !slices.Equal(input.Shape, []int64{1, int64(graph.MaxExecutionTokens)}) {
			return fmt.Errorf("%w: ONNX execution inputs differ from B1 window geometry", binding.ErrCapability)
		}
		names[input.Name] = true
	}
	if !names["input_ids"] || !names["attention_mask"] {
		return fmt.Errorf("%w: ONNX execution inputs are incomplete", binding.ErrCapability)
	}
	return nil
}

// A CK policy binds the actual trusted library and declared dynamic input schema.
// Empty ExecutionInputs alone proves nothing: all non-MIGraphX sessions leave it
// empty, including graphs that would pad a short input to a larger fixed shape.
func validateOperatingPointCKSession(graph *operatingpoint.ONNXExecution, session ort.SessionEvidence) error {
	if graph.CustomOpsProfile != "ck_flash_attention" || graph.ExecutionMode != "dynamic_sequence_b1" || graph.RuntimeBuild == "" || session.Provider != "ROCMExecutionProvider" || session.CustomOpsProfile != graph.CustomOpsProfile || session.RuntimeBuild != graph.RuntimeBuild || session.CustomOpsLibrary != operatingpoint.CKFlashAttentionLibrary || !session.CPUFallbackDisabled || len(session.ExecutionInputs) != 0 {
		return fmt.Errorf("%w: actual CK execution differs from frozen operating point", binding.ErrCapability)
	}
	librarySHA256 := ""
	for _, artifact := range graph.Artifacts {
		if artifact.Role == "custom-ops" {
			librarySHA256 = artifact.SHA256
		}
	}
	if librarySHA256 == "" || session.CustomOpsSHA256 != librarySHA256 {
		return fmt.Errorf("%w: actual CK library differs from frozen operating point", binding.ErrCapability)
	}
	names := map[string]bool{}
	for _, input := range session.InputSchema {
		if names[input.Name] || (input.Name != "input_ids" && input.Name != "attention_mask" && input.Name != "position_ids") || input.Dtype != "int64" || len(input.Shape) != 2 || input.Shape[1] != -1 || (input.Shape[0] != -1 && input.Shape[0] != 1) || (input.Name == "position_ids" && input.Shape[0] != 1) {
			return fmt.Errorf("%w: CK input schema differs from dynamic sequence/B1 execution", binding.ErrCapability)
		}
		names[input.Name] = true
	}
	if !names["input_ids"] || !names["attention_mask"] {
		return fmt.Errorf("%w: CK actual input schema is incomplete", binding.ErrCapability)
	}
	return nil
}
