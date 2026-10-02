package config

import (
	"encoding/json"
	"fmt"
)

// OutputTokenDefault supplies a fixed positive default or opts into the selected
// model's available output capacity. Absence is represented by a nil pointer.
type OutputTokenDefault struct {
	Auto  bool
	Value int
}

func FixedOutputTokenDefault(value int) *OutputTokenDefault {
	return &OutputTokenDefault{Value: value}
}

func (value OutputTokenDefault) Validate() error {
	if value.Auto && value.Value == 0 || !value.Auto && value.Value > 0 {
		return nil
	}
	return fmt.Errorf("default_max_tokens must be a positive integer or auto")
}

func (value *OutputTokenDefault) Fixed() *int {
	if value == nil || value.Auto {
		return nil
	}
	return &value.Value
}

func (value *OutputTokenDefault) IsAuto() bool { return value != nil && value.Auto }

func (value OutputTokenDefault) MarshalJSON() ([]byte, error) {
	if err := value.Validate(); err != nil {
		return nil, err
	}
	if value.Auto {
		return json.Marshal("auto")
	}
	return json.Marshal(value.Value)
}

func (value *OutputTokenDefault) UnmarshalJSON(data []byte) error {
	if value == nil {
		return fmt.Errorf("output token default is nil")
	}
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		if text != "auto" {
			return fmt.Errorf("default_max_tokens must be a positive integer or auto")
		}
		*value = OutputTokenDefault{Auto: true}
		return nil
	}
	var count int
	if err := json.Unmarshal(data, &count); err != nil {
		return fmt.Errorf("default_max_tokens must be a positive integer or auto")
	}
	*value = OutputTokenDefault{Value: count}
	return value.Validate()
}

func (value OutputTokenDefault) MarshalYAML() (interface{}, error) {
	if err := value.Validate(); err != nil {
		return nil, err
	}
	if value.Auto {
		return "auto", nil
	}
	return value.Value, nil
}

func (value *OutputTokenDefault) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var raw interface{}
	if err := unmarshal(&raw); err != nil {
		return err
	}
	encoded, err := json.Marshal(raw)
	if err != nil {
		return err
	}
	return value.UnmarshalJSON(encoded)
}
