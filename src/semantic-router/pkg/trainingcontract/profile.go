package trainingcontract

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// UnmarshalJSON rejects null label indices, which encoding/json otherwise
// silently converts to zero when decoding directly into map[string]int.
func (p *ClassifierProfile) UnmarshalJSON(data []byte) error {
	var wire struct {
		LabelMapping map[string]*int `json:"label_mapping"`
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&wire); err != nil {
		return err
	}
	labels := make(map[string]int, len(wire.LabelMapping))
	for label, index := range wire.LabelMapping {
		if index == nil {
			return fmt.Errorf("label_mapping[%q] must be an integer, got null", label)
		}
		labels[label] = *index
	}
	p.LabelMapping = labels
	return nil
}
