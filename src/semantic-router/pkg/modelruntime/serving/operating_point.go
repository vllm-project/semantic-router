package serving

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// OperatingPointScorer applies a head's packaged operating point: its window
// geometry, a per-label maximum over covering windows, and one threshold per
// label. Resources, admission and close use the generation's normal pool.
type OperatingPointScorer struct {
	handle     *binding.Resolved[tasks.TextWindowsRequest, tasks.WindowedLabelScores]
	window     tasks.TextWindowsRequest
	thresholds []float32
	digest     string
}

// OperatingPointResult holds the reduced scores and the windows they cover.
type OperatingPointResult struct {
	Scores  []float32
	Windows [][2]int
	Input   *tasks.InputUsage
}

// OperatingPoint binds a scores head with a packaged operating point. labels
// must equal the head's labels in order: thresholds are positional. A binding
// that pins operating_point.sha256 accepts only the policy file the runtime
// verified with that digest.
func (r *Runtime) OperatingPoint(ctx context.Context, spec config.ResolvedModelBinding, labels []string) (_ *OperatingPointScorer, callErr error) {
	defer func() { observePreparationFailure(spec, callErr) }()
	ctx, cancel := preparationContext(ctx)
	defer cancel()
	card, err := r.card(ctx, spec)
	if err != nil {
		return nil, err
	}
	head, ok := card.Head(spec.Binding.Head)
	if !ok || head.Kind != kindScores {
		return nil, fmt.Errorf("%w: deployment %q has no scores head %q", binding.ErrCapability, spec.Binding.Deployment, spec.Binding.Head)
	}
	if !slices.Equal(head.Labels, labels) || len(head.Thresholds) != len(head.Labels) || head.Window == nil || (head.Reduction != "" && head.Reduction != "max") {
		return nil, fmt.Errorf("%w: head %q has no operating point for the rule's labels (one threshold per label, a window and a per-label maximum)", binding.ErrCapability, head.Name)
	}
	if pinned := spec.Binding.OperatingPoint; pinned != nil && pinned.SHA256 != head.OperatingPointSHA256 {
		served := "no verified policy file"
		if head.OperatingPointSHA256 != "" {
			served = "sha256 " + head.OperatingPointSHA256
		}
		return nil, fmt.Errorf("%w: binding %q pins operating point sha256 %s, but deployment %q serves %s", binding.ErrCapability, spec.Name, pinned.SHA256, spec.Binding.Deployment, served)
	}
	window := tasks.TextWindowsRequest{Size: head.Window.Tokens, Overlap: head.Window.Overlap}
	handle, err := r.scoreWindowsTask(ctx, spec, window)
	if err != nil {
		return nil, err
	}
	descriptor, _ := json.Marshal(struct {
		Model, Revision, Head string
		Labels                []string
		Thresholds            []float64
		Window                tasks.TextWindowsRequest
	}{card.ModelSHA256, card.Revision, head.Name, head.Labels, head.Thresholds, window})
	digest := sha256.Sum256(descriptor)
	scorer := &OperatingPointScorer{handle: handle, window: window, thresholds: float32s(head.Thresholds), digest: hex.EncodeToString(digest[:])}
	r.mu.Lock()
	r.operatingPoints[handle] = scorer
	r.mu.Unlock()
	return scorer, nil
}

func (s *OperatingPointScorer) Close() error                   { return s.handle.Close() }
func (s *OperatingPointScorer) PolicySHA256() string           { return s.digest }
func (s *OperatingPointScorer) Thresholds() []float32          { return slices.Clone(s.thresholds) }
func (s *OperatingPointScorer) Capability() binding.Capability { return s.handle.Capability() }

// Score reads text in the operating point's windows and keeps each label's maximum.
func (s *OperatingPointScorer) Score(ctx context.Context, recipe, text string) (OperatingPointResult, error) {
	input := s.window
	input.Text = text
	output, err := s.handle.Call(ctx, recipe, input)
	if err != nil {
		return OperatingPointResult{}, err
	}
	scores := slices.Clone(output.Windows[0].Scores)
	ranges := make([][2]int, len(output.Windows))
	for i, window := range output.Windows {
		for label, score := range window.Scores {
			scores[label] = max(scores[label], score)
		}
		ranges[i] = [2]int{window.Start, window.End}
	}
	return OperatingPointResult{Scores: scores, Windows: ranges, Input: output.Input}, nil
}
