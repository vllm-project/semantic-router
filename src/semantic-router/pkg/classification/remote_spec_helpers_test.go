package classification

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// remoteSpecFor is the binding a remote deployment answers consumer through.
func remoteSpecFor(models []*classifierModelRuntime, consumer string, deployment config.ModelDeployment, adapter, contract string) config.ResolvedModelBinding {
	runtime := consumerModelRuntime(models)
	return config.ResolvedModelBinding{
		Recipe: runtime.recipe, Name: consumer,
		Binding:    config.ModelBinding{Deployment: consumer, Adapter: adapter, Contract: contract},
		Deployment: deployment,
	}
}

func llmLabelClassifierFor(rule config.ClassifierSignalRule, external *config.ExternalModelConfig, reasoning *reasoningRequestControl, models ...*classifierModelRuntime) (labelClassifier, error) {
	if external == nil {
		return nil, fmt.Errorf("external model %q is not configured", rule.Model)
	}
	spec := remoteSpecFor(models, "classifier."+rule.Name, externalChatDeployment(external), config.RemoteClassifierProtocolHTTPChat, config.RemoteClassifierContractLabelDistribution)
	return newLLMLabelClassifier(rule, spec, reasoning, consumerModelRuntime(models))
}

func sequenceLabelClassifierFor(rule config.ClassifierSignalRule, external *config.ExternalModelConfig, models ...*classifierModelRuntime) (labelClassifier, error) {
	if external == nil {
		return nil, fmt.Errorf("external model %q is not configured", rule.Model)
	}
	spec := remoteSpecFor(models, "classifier."+rule.Name, externalClassifyDeployment(external, 0), config.RemoteClassifierProtocolHTTPClassify, config.RemoteClassifierContractLabelDistribution)
	return newSequenceLabelClassifier(rule, spec, consumerModelRuntime(models))
}
