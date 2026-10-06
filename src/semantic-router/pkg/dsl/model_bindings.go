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
	var bindings map[string]config.ModelBinding
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&bindings); err != nil {
		return nil, err
	}
	return bindings, nil
}
