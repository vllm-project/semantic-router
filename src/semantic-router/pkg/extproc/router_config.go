package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (r *OpenAIRouter) routerConfig() *config.RouterConfig {
	if r != nil {
		if r.Config != nil {
			return r.Config
		}
		if r.RuntimeRegistry != nil {
			return r.RuntimeRegistry.CurrentConfig()
		}
	}
	return config.Get()
}

func (r *OpenAIRouter) decisionByName(name string) *config.Decision {
	if name == "" {
		return nil
	}
	if r == nil {
		return decisionFromConfig(config.Get(), name)
	}
	if decision := decisionFromConfig(r.routerConfig(), name); decision != nil {
		return decision
	}
	if r.Classifier == nil {
		return nil
	}
	return decisionFromConfig(r.Classifier.Config, name)
}

func decisionFromConfig(cfg *config.RouterConfig, name string) *config.Decision {
	if cfg == nil {
		return nil
	}
	return cfg.GetDecisionByName(name)
}
