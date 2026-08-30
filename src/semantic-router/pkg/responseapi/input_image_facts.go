package responseapi

import "encoding/json"

// CountNativeImageInputs counts image content parts across a request's input
// and the stored conversation history in their native Response API form
// (image_url, file_data, or file_id). Counting before translation keeps the
// image routing fact tied to what the client sent rather than to how the
// translator renders each part for Chat Completions.
// Only user items count, matching the translator (which drops non-user
// images) and every other image counter on the request path.
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
	if !isUserInputItem(item) || len(item.Content) == 0 {
		return 0
	}
	var parts []ContentPart
	if err := json.Unmarshal(item.Content, &parts); err != nil {
		return 0
	}
	count := 0
	for _, part := range parts {
		if isImagePart(part) {
			count++
		}
	}
	return count
}

// isUserInputItem reports whether an input item carries user content. An empty
// role defaults to user, mirroring inputItemToMessage in the translator.
func isUserInputItem(item InputItem) bool {
	return item.Role == "" || item.Role == RoleUser
}
