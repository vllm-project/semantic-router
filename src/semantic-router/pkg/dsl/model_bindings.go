package dsl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"maps"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func cloneModelBindings(bindings map[string]config.ModelBinding) map[string]config.ModelBinding {
	return maps.Clone(bindings)
}

func parseModelBindings(value Value) (map[string]config.ModelBinding, error) {
	object, ok := value.(ObjectValue)
	if !ok {
		return nil, fmt.Errorf("must be an object keyed by task consumer")
	}
	payload, err := json.Marshal(dslFieldObjectFromValues(object.Fields).asInterfaceMap())
	if err != nil {
		return nil, err
	}
	var authored map[string]config.ModelBinding
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&authored); err != nil {
		return nil, err
	}
	// A signal-type key and the consumer name of earlier releases bind the
	// same consumer.
	bindings := make(map[string]config.ModelBinding, len(authored))
	for key, binding := range authored {
		consumer := config.BindingConsumer(key)
		if _, both := bindings[consumer]; both {
			return nil, fmt.Errorf("%s and %s bind the same signal; keep %s", key, config.BindingKey(consumer), config.BindingKey(consumer))
		}
		bindings[consumer] = binding
	}
	return bindings, nil
}
