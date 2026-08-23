package config

import "fmt"

// validateLoRAName checks if the specified LoRA name is defined in the
// canonical routing model catalog for the selected model.
func validateLoRAName(cfg *RouterConfig, modelName string, loraName string) error {
	modelParams, exists := cfg.ModelConfig[modelName]
	if !exists {
		return fmt.Errorf(
			"lora_name %q specified but model %q is not declared in the compiled Model set",
			loraName,
			modelName,
		)
	}

	if len(modelParams.LoRAs) == 0 {
		return fmt.Errorf(
			"lora_name %q specified but model %q declares no LoRA adapters",
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
		"lora_name %q is not declared for model %q. Available LoRAs: %v",
		loraName,
		modelName,
		availableLoRAs,
	)
}
