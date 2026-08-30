package responseapi

import (
	"encoding/json"
	"testing"
)

func TestResponseObjectTypesRetainSupportedOutputAndUsageFields(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","object":"response","created_at":1,"model":"model","status":"completed",
		"output":[
			{"type":"reasoning","id":"reasoning_1","summary":[{"type":"summary_text","text":"inspect"}],"content":[{"type":"reasoning_text","text":"details"}]},
			{"type":"message","id":"message_1","role":"assistant","content":[
				{"type":"refusal","refusal":"cannot comply"},
				{"type":"output_text","text":"safe alternative","annotations":[{"type":"url_citation","url":"https://example.com","title":"Example","start_index":0,"end_index":4}]}
			]}
		],
		"usage":{"input_tokens":9,"input_tokens_details":{"cached_tokens":4,"cache_write_tokens":2},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":1},"total_tokens":12}
	}`)
	var response ResponseAPIResponse
	if err := json.Unmarshal(body, &response); err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	var document map[string]any
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatal(err)
	}
	output := document["output"].([]any)
	reasoning := output[0].(map[string]any)
	if reasoning["summary"].([]any)[0].(map[string]any)["text"] != "inspect" {
		t.Fatalf("reasoning summary was not retained: %s", encoded)
	}
	content := output[1].(map[string]any)["content"].([]any)
	if content[0].(map[string]any)["refusal"] != "cannot comply" {
		t.Fatalf("refusal was not retained: %s", encoded)
	}
	annotation := content[1].(map[string]any)["annotations"].([]any)[0].(map[string]any)
	if annotation["url"] != "https://example.com" || annotation["title"] != "Example" {
		t.Fatalf("citation was not retained: %s", encoded)
	}
	if annotation["start_index"] != float64(0) || annotation["end_index"] != float64(4) {
		t.Fatalf("citation offsets were not retained: %s", encoded)
	}
	usage := document["usage"].(map[string]any)
	inputDetails := usage["input_tokens_details"].(map[string]any)
	if inputDetails["cached_tokens"] != float64(4) || inputDetails["cache_write_tokens"] != float64(2) {
		t.Fatalf("input usage details were not retained: %s", encoded)
	}
}
