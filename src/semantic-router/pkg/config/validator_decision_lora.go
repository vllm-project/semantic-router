package config

import "fmt"

// validateLoRAName checks if the specified LoRA name is defined in the
// facts of the selected model's providers.models[] entry.
func validateLoRAName(cfg *RouterConfig, modelName string, loraName string) error {
	modelParams, exists := cfg.ModelConfig[modelName]
	if !exists {
		return fmt.Errorf(
			"lora_name %q specified but model %q is not declared in providers.models",
			loraName,
			modelName,
		)
	}

	if len(modelParams.LoRAs) == 0 {
		return fmt.Errorf(
			"lora_name %q specified but model %q declares no providers.models[].loras entries",
			loraName,
			modelName,
		)
	}

	for _, lora := range modelParams.LoRAs {
		if lora.Name == loraName {
			return nil
		}
	}

	availableLoRAs := make([]string, len(modelParams.LoRAs))
	for i, lora := range modelParams.LoRAs {
		availableLoRAs[i] = lora.Name
	}
	return fmt.Errorf(
		"lora_name %q is not declared in providers.models[%q].loras. Available LoRAs: %v",
		loraName,
		modelName,
		availableLoRAs,
	)
}
