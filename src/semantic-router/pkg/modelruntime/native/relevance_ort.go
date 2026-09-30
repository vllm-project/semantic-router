package native

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func (r *Runtime) ortRelevance(ctx context.Context, spec config.ResolvedModelBinding, selection config.PairScorerSelection) (*preparedRelevance, error) {
	prepared := &preparedRelevance{}
	selectionJSON, err := json.Marshal(selection)
	if err != nil {
		return nil, err
	}
	resource, err := r.ortResource(ctx, spec, "relevance:"+string(selectionJSON), func(options ort.Options) (io.Closer, error) {
		model, loadErr := ort.LoadPairScorer(options, ort.PairScorerSelection{Layer: selection.Layer, Dimension: selection.Dimension})
		if loadErr != nil {
			return nil, loadErr
		}
		return model, nil
	})
	if err != nil {
		return nil, err
	}
	if spec.Binding.Head != "" {
		path := spec.Binding.Head
		if !filepath.IsAbs(path) {
			path = filepath.Join(spec.Deployment.Artifact, path)
		}
		prepared.headRevision, err = r.artifactRevision(ctx, filepath.Dir(path))
		if err != nil {
			_ = resource.Close()
			return nil, err
		}
	}
	err = resource.Use(ctx, func(value io.Closer) error {
		info, infoErr := value.(*ort.PairScorer).Info()
		if infoErr != nil {
			return ortError(infoErr)
		}
		if info.PairScorer == nil {
			return fmt.Errorf("%w: missing actual ONNX pair scorer selection", binding.ErrCapability)
		}
		prepared.selection = config.PairScorerSelection{Layer: info.PairScorer.Layer, Dimension: info.PairScorer.Dimension}
		prepared.capability, infoErr = ortCapability(spec, info)
		if infoErr != nil {
			return infoErr
		}
		for _, session := range info.Sessions {
			graphHash, hashErr := r.artifactRevision(ctx, session.Graph)
			if hashErr != nil {
				return hashErr
			}
			externalHash, hashErr := r.artifactRevision(ctx, filepath.Dir(session.Graph))
			if hashErr != nil {
				return hashErr
			}
			prepared.execution = append(prepared.execution, ortRelevanceExecution(session, prepared.capability.Device, graphHash, externalHash))
		}
		return nil
	})
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	prepared.infer = func(_ context.Context, value io.Closer, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
		input := make([]ort.TextPair, len(pairs))
		for i, pair := range pairs {
			input[i] = ort.TextPair{Query: pair.Query, Document: pair.Document}
		}
		output, inferErr := value.(*ort.PairScorer).ScorePairs(input)
		result := tasks.RelevanceScores{Scores: output.Scores, Inputs: make([]tasks.InputUsage, len(output.Inputs))}
		for i, usage := range output.Inputs {
			result.Inputs[i] = *ortInputUsage(usage)
		}
		return result, ortError(inferErr)
	}
	prepared.resource = resource
	return prepared, nil
}

func ortRelevanceExecution(session ort.SessionEvidence, device, graphHash, externalHash string) relevanceExecution {
	return relevanceExecution{
		GraphFingerprint: graphHash, ExternalArtifactFingerprint: externalHash,
		Provider: session.Provider, Device: device, Precision: session.Precision,
		RuntimeBuild: session.RuntimeBuild, CompilerFlags: session.CompilerFlags,
		CustomOpsProfile: session.CustomOpsProfile, CustomOpsSHA256: session.CustomOpsSHA256,
		CPUFallbackDisabled: session.CPUFallbackDisabled,
	}
}
