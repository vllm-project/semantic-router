package classification

import (
	"context"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

type ownedRemoteGuardDistribution struct {
	handle *binding.Resolved[string, tasks.LabelDistribution]
	recipe string
}

func (b *ownedRemoteGuardDistribution) Classify(ctx context.Context, text string) (tasks.LabelDistribution, error) {
	return b.handle.Call(ctx, b.recipe, text)
}
func (b *ownedRemoteGuardDistribution) Close() error { return b.handle.Close() }

type ownedRemoteGuardDecision struct {
	handle *binding.Resolved[string, tasks.LabelDecision]
	recipe string
}

func (b *ownedRemoteGuardDecision) Classify(context.Context, string) (tasks.LabelDistribution, error) {
	return tasks.LabelDistribution{}, tasks.ErrProbabilitiesUnavailable
}

func (b *ownedRemoteGuardDecision) Decide(ctx context.Context, text string) (tasks.LabelDecision, error) {
	return b.handle.Call(ctx, b.recipe, text)
}
func (b *ownedRemoteGuardDecision) Close() error { return b.handle.Close() }

func bindRemoteJailbreak(models []*classifierModelRuntime, cfg *config.RouterConfig, backend *config.RemoteClassifierBackend, external *config.ExternalModelConfig, mapping *JailbreakMapping, inference SequenceClassifierBackend) (SequenceClassifierBackend, error) {
	runtime := consumerModelRuntime(models)
	contract := config.RemoteClassifierContractLabelDistribution
	if backend.Protocol == config.RemoteClassifierProtocolHTTPChat {
		contract = config.RemoteClassifierContractLabelDecision
	}
	spec := config.ResolvedModelBinding{Recipe: runtime.recipe, Name: "prompt_guard", Binding: config.ModelBinding{Deployment: backend.Model, Contract: contract, Adapter: backend.Protocol}, Deployment: config.ModelDeployment{Provider: "http", ExternalModel: backend.Model}, Admission: cfg.ModelAdmission["prompt_guard"]}
	if declared, ok := runtime.plan.Lookup(runtime.recipe, "prompt_guard"); ok {
		spec = declared
	}
	closer, ok := inference.(io.Closer)
	if !ok {
		return nil, fmt.Errorf("remote guard connector has no ownership lifecycle")
	}
	if decision, ok := inference.(LabelDecisionBackend); ok {
		handle, err := remoteTaskBinding(context.Background(), runtime, spec, external, closer, decision.Decide, func(_ string, out tasks.LabelDecision) error {
			if _, ok := mapping.GetIndexForJailbreakType(out.Label); !ok {
				return fmt.Errorf("unknown guard label %q", out.Label)
			}
			if out.Score != nil {
				if out.ScoreSemantics == nil {
					return fmt.Errorf("guard score requires semantics")
				}
				return out.ScoreSemantics.Validate(*out.Score)
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		return &ownedRemoteGuardDecision{handle: handle, recipe: string(spec.Recipe)}, nil
	}
	handle, err := remoteTaskBinding(context.Background(), runtime, spec, external, closer, inference.Classify, func(_ string, out tasks.LabelDistribution) error {
		return validateJailbreakDistribution(mapping, cfg.PromptGuard.PositiveLabels, out)
	})
	if err != nil {
		return nil, err
	}
	return &ownedRemoteGuardDistribution{handle: handle, recipe: string(spec.Recipe)}, nil
}
