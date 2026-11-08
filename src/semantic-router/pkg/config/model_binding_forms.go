package config

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// A binding names the deployment that answers one application signal type,
// written as the deployment's name (`pii: local-vela`) or as
// {deployment: local-vela}; the signal decides the question and the provider
// how it is asked.

// knownBindingKey reports whether a key names a signal a binding can serve:
// an application signal type.
func knownBindingKey(key string) bool {
	return slices.Contains(SharedAnswerSignalTypes, key)
}

// modelBindingObject is the object form of a binding.
type modelBindingObject struct {
	Deployment string `yaml:"deployment" json:"deployment"`
}

// UnmarshalYAML reads a binding written as a deployment name or an object,
// which may set no field a binding does not have.
func (b *ModelBinding) UnmarshalYAML(unmarshal func(interface{}) error) error {
	var name string
	if err := unmarshal(&name); err == nil {
		*b = ModelBinding{Deployment: strings.TrimSpace(name)}
		return nil
	}
	var fields map[string]interface{}
	if err := unmarshal(&fields); err != nil {
		return err
	}
	for field := range fields {
		if field != "deployment" {
			return fmt.Errorf("a binding names only its deployment; %s is not a binding field", field)
		}
	}
	var object modelBindingObject
	if err := unmarshal(&object); err != nil {
		return err
	}
	*b = ModelBinding(object)
	return nil
}

// MarshalYAML writes a binding as its deployment's name.
func (b ModelBinding) MarshalYAML() (interface{}, error) {
	return b.Deployment, nil
}

// UnmarshalJSON reads a binding written as a deployment name or an object,
// which may set no field a binding does not have.
func (b *ModelBinding) UnmarshalJSON(data []byte) error {
	var name string
	if err := json.Unmarshal(data, &name); err == nil {
		*b = ModelBinding{Deployment: strings.TrimSpace(name)}
		return nil
	}
	var object modelBindingObject
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&object); err != nil {
		return err
	}
	*b = ModelBinding(object)
	return nil
}

// MarshalJSON writes a binding as its deployment's name.
func (b ModelBinding) MarshalJSON() ([]byte, error) {
	return json.Marshal(b.Deployment)
}
