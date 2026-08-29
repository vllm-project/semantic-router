package responseapi

import "encoding/json"

// CountNativeImageInputs counts image content parts across a request's input
// and the stored conversation history in their native Response API form.
// Chat Completions can only express URL images, so translation drops
// input_image parts referenced by file_id or inline file_data; counting
// before translation keeps those images visible to request-shape facts.
func CountNativeImageInputs(input json.RawMessage, history []*StoredResponse) int {
	count := countRawInputImageParts(input)
	for _, resp := range history {
		if resp == nil {
			continue
		}
		for _, item := range resp.Input {
			count += countInputItemImageParts(item)
		}
	}
	return count
}

func countRawInputImageParts(input json.RawMessage) int {
	if len(input) == 0 {
		return 0
	}
	var items []InputItem
	if err := json.Unmarshal(input, &items); err != nil {
		return 0
	}
	count := 0
	for _, item := range items {
		count += countInputItemImageParts(item)
	}
	return count
}

func countInputItemImageParts(item InputItem) int {
	if len(item.Content) == 0 {
		return 0
	}
	var parts []ContentPart
	if err := json.Unmarshal(item.Content, &parts); err != nil {
		return 0
	}
	count := 0
	for _, part := range parts {
		if part.Type == ContentTypeInputImage &&
			(part.ImageURL != "" || part.FileID != "" || part.FileData != "") {
			count++
		}
	}
	return count
}
