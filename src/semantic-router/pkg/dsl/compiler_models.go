package dsl

import (
	modelcatalog "github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
	if v, ok := getEvaluationsField(fields, "evaluations"); ok {
		params.Evaluations = v
	}
}

func getEvaluationsField(fields map[string]Value, key string) ([]modelcatalog.UserEvaluation, bool) {
	raw, ok := fields[key]
	if !ok {
		return nil, false
	}
	array, ok := raw.(ArrayValue)
	if !ok {
		return nil, false
	}
	evaluations := make([]modelcatalog.UserEvaluation, 0, len(array.Items))
	for _, item := range array.Items {
		object, ok := item.(ObjectValue)
		if !ok {
			continue
		}
		benchmark, ok := getStringField(object.Fields, "benchmark")
		if !ok || benchmark == "" {
			continue
		}
		evaluation := modelcatalog.UserEvaluation{Benchmark: benchmark}
		if metrics, ok := object.Fields["metrics"].(ObjectValue); ok {
			evaluation.Metrics = make(map[string]float64, len(metrics.Fields))
			for metric, value := range metrics.Fields {
				switch typed := value.(type) {
				case FloatValue:
					evaluation.Metrics[metric] = typed.V
				case IntValue:
					evaluation.Metrics[metric] = float64(typed.V)
				}
			}
		}
		evaluation.Source, _ = getStringField(object.Fields, "source")
		evaluation.MeasuredAt, _ = getStringField(object.Fields, "measured_at")
		if metadata, ok := object.Fields["metadata"].(ObjectValue); ok {
			evaluation.Metadata = fieldsToMap(metadata.Fields)
		}
		evaluations = append(evaluations, evaluation)
	}
	return evaluations, true
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
