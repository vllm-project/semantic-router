package controllers

import (
	"fmt"

	"gopkg.in/yaml.v3"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"

	authoringconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config/authoring"
)

func compileAuthoringConfig(raw *apiextensionsv1.JSON) (map[string]interface{}, error) {
	if raw == nil || len(raw.Raw) == 0 {
		return nil, fmt.Errorf("spec.authoringConfig must not be empty")
	}

	cfg, err := authoringconfig.Parse(raw.Raw)
	if err != nil {
		return nil, fmt.Errorf("parse authoringConfig: %w", err)
	}
	runtimeCfg, err := authoringconfig.CompileRuntime(cfg)
	if err != nil {
		return nil, fmt.Errorf("compile authoringConfig: %w", err)
	}

	data, err := yaml.Marshal(runtimeCfg)
	if err != nil {
		return nil, fmt.Errorf("marshal compiled authoring runtime config: %w", err)
	}

	var compiled map[string]interface{}
	if err := yaml.Unmarshal(data, &compiled); err != nil {
		return nil, fmt.Errorf("unmarshal compiled authoring runtime config: %w", err)
	}
	if cfg.Version != "" {
		compiled["version"] = cfg.Version
	}
	return compiled, nil
}
