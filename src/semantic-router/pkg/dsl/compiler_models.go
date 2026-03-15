package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

func (c *Compiler) compileModels() {
	if len(c.prog.Models) == 0 {
		return
	}
	if c.config.ModelConfig == nil {
		c.config.ModelConfig = make(map[string]config.ModelParams, len(c.prog.Models))
	}

	for _, model := range c.prog.Models {
		params := c.config.ModelConfig[model.Name]
		if v, ok := getStringField(model.Fields, "reasoning_family_ref"); ok {
			params.ReasoningFamily = v
		} else if v, ok := getStringField(model.Fields, "reasoning_family"); ok {
			params.ReasoningFamily = v
		}
		if v, ok := getStringField(model.Fields, "param_size"); ok {
			params.ParamSize = v
		}
		if v, ok := getIntField(model.Fields, "context_window_size"); ok {
			params.ContextWindowSize = v
		}
		if v, ok := getStringField(model.Fields, "description"); ok {
			params.Description = v
		}
		if v, ok := getStringArrayField(model.Fields, "capabilities"); ok {
			params.Capabilities = v
		}
		if v, ok := getLoRAAdapterField(model.Fields, "loras"); ok {
			params.LoRAs = v
		}
		if v, ok := getStringArrayField(model.Fields, "tags"); ok {
			params.Tags = v
		}
		if v, ok := getFloat64Field(model.Fields, "quality_score"); ok {
			params.QualityScore = v
		}
		if v, ok := getStringField(model.Fields, "modality"); ok {
			params.Modality = v
		}
		c.config.ModelConfig[model.Name] = params
	}
}

func getLoRAAdapterField(fields map[string]Value, key string) ([]config.LoRAAdapter, bool) {
	raw, ok := fields[key]
	if !ok {
		return nil, false
	}

	av, ok := raw.(ArrayValue)
	if !ok {
		return nil, false
	}

	adapters := make([]config.LoRAAdapter, 0, len(av.Items))
	for _, item := range av.Items {
		ov, ok := item.(ObjectValue)
		if !ok {
			continue
		}
		name, ok := getStringField(ov.Fields, "name")
		if !ok || name == "" {
			continue
		}
		adapter := config.LoRAAdapter{Name: name}
		if description, ok := getStringField(ov.Fields, "description"); ok {
			adapter.Description = description
		}
		adapters = append(adapters, adapter)
	}
	return adapters, true
}
