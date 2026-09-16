package native

import (
	"context"
	"fmt"
	"path/filepath"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/operatingpoint"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// OperatingPointScorer interprets an existing owned score-window task. Policy
// identity is recipe-local; resources, admission and close use the normal pool.
type OperatingPointScorer struct {
	handle *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores]
	policy *operatingpoint.Policy
}

type OperatingPointResult struct {
	Scores  []float32
	Windows [][2]int
	Input   *tasks.InputUsage
}

func (r *Runtime) OperatingPoint(ctx context.Context, spec config.ResolvedModelBinding, labels []string) (*OperatingPointScorer, error) {
	policy, err := operatingpoint.Load(ctx, spec, labels)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", binding.ErrCapability, err)
	}
	if err = r.verifyPreparedArtifact(ctx, spec.Deployment.Artifact); err != nil {
		return nil, err
	}
	var handle *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores]
	if graph := policy.ONNX(); graph != nil {
		spec.Binding.Head = graph.File
		handle, err = r.ortScoreWindowsWithPolicy(ctx, spec, policy.Window(), policy)
	} else {
		handle, err = r.ScoreWindows(ctx, spec, policy.Window())
	}
	if err != nil {
		return nil, err
	}
	if err = policy.ValidateCapability(handle.Capability()); err == nil {
		err = policy.VerifyArtifacts(ctx, spec.Deployment.Artifact)
	}
	if err == nil {
		// Validate the policy against the actual tokenizer's result before
		// publishing the wrapper, including its special-token envelope.
		warmup := policy.Window()
		warmup.Text = "warmup"
		var result tasks.WindowedLabelScores
		result, err = handle.Call(ctx, string(spec.Recipe), warmup)
		if err == nil {
			_, err = policy.Reduce(result)
		}
	}
	if err == nil {
		err = r.verifyPreparedArtifact(ctx, spec.Deployment.Artifact)
	}
	if err != nil {
		_ = handle.Close()
		return nil, err
	}
	return &OperatingPointScorer{handle: handle, policy: policy}, nil
}

// A generation caches its preparation identity. Refuse a new policy after an
// in-place replacement rather than accidentally reuse its preceding owner.
func (r *Runtime) verifyPreparedArtifact(ctx context.Context, path string) error {
	prepared, err := r.artifactRevision(ctx, path)
	if err != nil {
		return err
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	abs, err = filepath.EvalSymlinks(abs)
	if err != nil {
		return err
	}
	current, err := fingerprintArtifact(ctx, abs)
	if err != nil {
		return err
	}
	if prepared != current {
		return fmt.Errorf("%w: artifact changed within the prepared generation; prepare a new generation", binding.ErrCapability)
	}
	return nil
}

func (s *OperatingPointScorer) Close() error                   { return s.handle.Close() }
func (s *OperatingPointScorer) PolicySHA256() string           { return s.policy.Digest() }
func (s *OperatingPointScorer) Thresholds() []float32          { return s.policy.Thresholds() }
func (s *OperatingPointScorer) Capability() binding.Capability { return s.handle.Capability() }

func (s *OperatingPointScorer) Score(ctx context.Context, recipe, text string) (OperatingPointResult, error) {
	input := s.policy.Window()
	input.Text = text
	output, err := s.handle.Call(ctx, recipe, input)
	if err != nil {
		return OperatingPointResult{}, err
	}
	scores, err := s.policy.Reduce(output)
	if err != nil {
		return OperatingPointResult{}, err
	}
	ranges := make([][2]int, len(output.Windows))
	for i, window := range output.Windows {
		ranges[i] = [2]int{window.Start, window.End}
	}
	return OperatingPointResult{Scores: scores, Windows: ranges, Input: output.Input}, nil
}
