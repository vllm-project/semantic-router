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
		applyRoutingModelTextFields(&params, model.Fields)
		applyRoutingModelNumericFields(&params, model.Fields)
		applyRoutingModelArrayFields(&params, model.Fields)
		c.config.ModelConfig[model.Name] = params
	}
}

func applyRoutingModelTextFields(params *config.ModelParams, fields map[string]Value) {
	if v, ok := getStringField(fields, "param_size"); ok {
		params.ParamSize = v
	}
	if v, ok := getStringField(fields, "description"); ok {
		params.Description = v
	}
	if v, ok := getStringField(fields, "modality"); ok {
		params.Modality = v
	}
}

func applyRoutingModelNumericFields(
	params *config.ModelParams,
	fields map[string]Value,
) {
	if v, ok := getIntField(fields, "context_window_size"); ok {
		params.ContextWindowSize = v
	}
	if v, ok := getFloat64Field(fields, "quality_score"); ok {
		params.QualityScore = v
	}
}

func applyRoutingModelArrayFields(params *config.ModelParams, fields map[string]Value) {
	if v, ok := getStringArrayField(fields, "capabilities"); ok {
		params.Capabilities = v
	}
	if v, ok := getLoRAAdapterField(fields, "loras"); ok {
		params.LoRAs = v
	}
	if v, ok := getStringArrayField(fields, "tags"); ok {
		params.Tags = v
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
