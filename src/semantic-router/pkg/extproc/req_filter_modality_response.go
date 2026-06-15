package extproc

import (
	"fmt"
	"time"
)

func extractResponseModelName(resp map[string]interface{}) string {
	model, _ := resp["model"].(string)
	return model
}

func extractPrimaryChoiceMessage(resp map[string]interface{}) map[string]interface{} {
	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return nil
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return nil
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return nil
	}
	return message
}

func buildOmniResponseOutputItems(msg map[string]interface{}) []map[string]interface{} {
	if msg == nil {
		return nil
	}

	textParts, imageParts := extractOmniContentParts(msg["content"])
	outputItems := make([]map[string]interface{}, 0, len(imageParts)+1)
	for _, imgURL := range imageParts {
		outputItems = append(outputItems, map[string]interface{}{
			"type":   "image_generation_call",
			"id":     fmt.Sprintf("ig_%d", time.Now().UnixNano()),
			"status": "completed",
			"result": imgURL,
		})
	}

	if len(textParts) == 0 {
		return outputItems
	}

	contentParts := make([]map[string]interface{}, 0, len(textParts))
	for _, text := range textParts {
		contentParts = append(contentParts, map[string]interface{}{
			"type":        "output_text",
			"text":        text,
			"annotations": []interface{}{},
		})
	}

	outputItems = append(outputItems, map[string]interface{}{
		"type":    "message",
		"role":    "assistant",
		"content": contentParts,
	})
	return outputItems
}

func defaultGeneratedImageOutputItem() map[string]interface{} {
	return map[string]interface{}{
		"type": "message",
		"role": "assistant",
		"content": []map[string]interface{}{
			{
				"type":        "output_text",
				"text":        defaultImageResponseText,
				"annotations": []interface{}{},
			},
		},
	}
}

func extractOmniContentPart(partMap map[string]interface{}) (string, string) {
	switch partMap["type"] {
	case "text":
		text, _ := partMap["text"].(string)
		return text, ""
	case "image_url":
		imageURL, _ := partMap["image_url"].(map[string]interface{})
		url, _ := imageURL["url"].(string)
		return "", url
	default:
		return "", ""
	}
}

func extractPrimaryChoiceTextContent(resp map[string]interface{}) string {
	msg := extractPrimaryChoiceMessage(resp)
	if msg == nil {
		return ""
	}
	content, _ := msg["content"].(string)
	return content
}

func buildBothResponseContentParts(textContent string, imgResult *ImageGenResult) []map[string]interface{} {
	contentParts := make([]map[string]interface{}, 0, 2)
	if textContent != "" {
		contentParts = append(contentParts, map[string]interface{}{
			"type": "text",
			"text": textContent,
		})
	}

	if imgResult != nil && imgResult.ImageURL != "" {
		contentParts = append(contentParts, map[string]interface{}{
			"type": "image_url",
			"image_url": map[string]string{
				"url": imgResult.ImageURL,
			},
		})
	}

	if len(contentParts) > 0 {
		return contentParts
	}

	return []map[string]interface{}{
		{
			"type": "text",
			"text": "Failed to generate both text and image responses.",
		},
	}
}
