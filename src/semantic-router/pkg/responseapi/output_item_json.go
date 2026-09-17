package responseapi

import "encoding/json"

// MarshalJSON preserves the required reasoning summary array across storage
// and public Responses reads, including when no summary was produced.
func (item OutputItem) MarshalJSON() ([]byte, error) {
	type outputItem OutputItem
	if item.Type != ItemTypeReasoning {
		return json.Marshal(outputItem(item))
	}
	summary := item.Summary
	if summary == nil {
		summary = []ContentPart{}
	}
	return json.Marshal(struct {
		outputItem
		Summary []ContentPart `json:"summary"`
	}{outputItem: outputItem(item), Summary: summary})
}
