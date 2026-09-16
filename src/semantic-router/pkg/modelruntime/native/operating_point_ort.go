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
	graph := policy.ONNX()
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
	if actual != want || session.RuntimeBuild == "" || session.Provider != graph.ExecutionProvider || session.Precision != "native" || session.ExecutionMaxInputTokens != graph.MaxExecutionTokens || (session.CustomOpsProfile != "" && session.CustomOpsProfile != "none") {
		return fmt.Errorf("%w: actual ONNX execution differs from operating point", binding.ErrCapability)
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
