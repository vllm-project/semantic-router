package responseapi

import "encoding/json"

// MarshalJSON retains the fields required by each content variant even when
// their value is empty. Storage and public object reads share this contract.
func (part ContentPart) MarshalJSON() ([]byte, error) {
	type contentPart ContentPart
	switch part.Type {
	case "output_text":
		annotations := part.Annotations
		if annotations == nil {
			annotations = []Annotation{}
		}
		return json.Marshal(struct {
			contentPart
			Text        string       `json:"text"`
			Annotations []Annotation `json:"annotations"`
		}{contentPart: contentPart(part), Text: part.Text, Annotations: annotations})
	case "input_text", "reasoning_text", "summary_text":
		return json.Marshal(struct {
			contentPart
			Text string `json:"text"`
		}{contentPart: contentPart(part), Text: part.Text})
	case "refusal":
		return json.Marshal(struct {
			contentPart
			Refusal string `json:"refusal"`
		}{contentPart: contentPart(part), Refusal: part.Refusal})
	default:
		return json.Marshal(contentPart(part))
	}
}
