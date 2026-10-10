package services

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

// ErrInvalidClassificationInput identifies a malformed typed classification item.
var ErrInvalidClassificationInput = errors.New("invalid classification input")

// ClassificationInput is exactly one text or inline image. ImageURL never
// accepts remote URLs or filesystem paths. Invalid JSON item shapes are retained
// until validation so one bad batch item cannot discard its siblings.
type ClassificationInput struct {
	Type     string  `json:"type"`
	Text     *string `json:"text,omitempty"`
	ImageURL *string `json:"image_url,omitempty"`
	invalid  bool
	hasText  bool
	hasImage bool
}

// UnmarshalJSON retains invalid item shapes for per-item batch validation.
func (input *ClassificationInput) UnmarshalJSON(data []byte) error {
	type plain ClassificationInput
	var value plain
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&value)
	*input = ClassificationInput(value)
	var fields map[string]json.RawMessage
	_ = json.Unmarshal(data, &fields)
	_, input.hasText = fields["text"]
	_, input.hasImage = fields["image_url"]
	input.invalid = err != nil || len(bytes.TrimSpace(data)) == 0 || bytes.TrimSpace(data)[0] != '{'
	if !input.invalid {
		// Repeated discriminator/payload keys must not pick a last-wins variant.
		scanner := json.NewDecoder(bytes.NewReader(data))
		_, _ = scanner.Token()
		seen := make(map[string]bool)
		for scanner.More() {
			key, _ := scanner.Token()
			name, _ := key.(string)
			if seen[name] || (name != "type" && name != "text" && name != "image_url") {
				input.invalid = true
			}
			seen[name] = true
			var value json.RawMessage
			_ = scanner.Decode(&value)
		}
	}
	return nil
}

// Validate checks the discriminator and payload without performing inference.
func (input ClassificationInput) Validate() error {
	if input.invalid {
		return fmt.Errorf("%w: expected an object containing only type and its payload", ErrInvalidClassificationInput)
	}
	switch input.Type {
	case "text":
		if input.Text == nil || (input.ImageURL != nil || input.hasImage) || strings.TrimSpace(*input.Text) == "" {
			return fmt.Errorf("%w: text input requires nonempty text and no image_url", ErrInvalidClassificationInput)
		}
	case "image":
		if input.ImageURL == nil || input.Text != nil || input.hasText {
			return fmt.Errorf("%w: image input requires image_url and no text", ErrInvalidClassificationInput)
		}
		if _, ok := imageurl.DecodeBase64(strings.TrimSpace(*input.ImageURL)); !ok {
			return fmt.Errorf("%w: image_url must be an allowlisted inline base64 image data URI", ErrInvalidClassificationInput)
		}
	default:
		return fmt.Errorf("%w: type must be text or image", ErrInvalidClassificationInput)
	}
	return nil
}

func (input ClassificationInput) intentRequest(options *IntentOptions) (IntentRequest, error) {
	if err := input.Validate(); err != nil {
		return IntentRequest{}, err
	}
	req := IntentRequest{Options: options}
	if input.Type == "text" {
		req.Text = *input.Text
		return req, nil
	}
	canonical, _ := imageurl.CanonicalDataURL(strings.TrimSpace(*input.ImageURL))
	content, err := json.Marshal([]intentMessageContentPart{{Type: "image_url", ImageURL: &intentMessageImageURL{URL: canonical}}})
	if err != nil {
		return IntentRequest{}, err
	}
	req.Messages = []IntentMessage{{Role: "user", Content: content}}
	return req, nil
}

func (req IntentRequest) withClassificationInput() (IntentRequest, error) {
	if req.Input == nil {
		return req, nil
	}
	if req.Text != "" || len(req.Messages) != 0 {
		return IntentRequest{}, fmt.Errorf("%w: input cannot be combined with text or messages", ErrInvalidClassificationInput)
	}
	input, err := req.Input.intentRequest(req.Options)
	if err != nil {
		return IntentRequest{}, err
	}
	req.Text, req.Messages = input.Text, input.Messages
	return req, nil
}
