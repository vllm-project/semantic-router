package recipe

import (
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"strings"
)

type probeFixtures struct {
	Images map[string]probeImageFixture `yaml:"images"`
}

type probeImageFixture struct {
	Description string `yaml:"description"`
	MediaType   string `yaml:"media_type"`
	DataBase64  string `yaml:"data_base64"`
	SHA256      string `yaml:"sha256"`
	Bytes       int    `yaml:"-"`
}

type probeGeneratedText struct {
	MessageIndex    *int   `yaml:"message_index"`
	ContentIndex    *int   `yaml:"content_index"`
	TargetTextBytes *int   `yaml:"target_text_bytes"`
	Character       string `yaml:"character"`
}

func validateProbeFixtures(fixtures *probeFixtures, issues *[]string) {
	if len(fixtures.Images) == 0 {
		*issues = append(*issues, "fixtures.images must contain at least one image fixture")
		return
	}
	for name, fixture := range fixtures.Images {
		fixtures.Images[name] = validateProbeImageFixture(name, fixture, issues)
	}
}

func validateProbeImageFixture(
	name string,
	fixture probeImageFixture,
	issues *[]string,
) probeImageFixture {
	label := "fixtures.images." + name
	if !validProbeIdentifier(name) {
		*issues = append(*issues, label+" has an invalid fixture name")
	}
	fixture.Description = strings.TrimSpace(fixture.Description)
	if err := requireBoundedText(label+".description", fixture.Description, 512); err != nil {
		*issues = append(*issues, err.Error())
	}
	if !supportedImageFixtureMediaType(fixture.MediaType) {
		*issues = append(*issues, label+".media_type is unsupported")
	}
	fixture.DataBase64 = strings.Join(strings.Fields(fixture.DataBase64), "")
	decoded, err := base64.StdEncoding.Strict().DecodeString(fixture.DataBase64)
	if err != nil {
		*issues = append(*issues, label+".data_base64 is invalid")
		return fixture
	}
	if len(decoded) == 0 || len(decoded) > maxImageFixtureBytes {
		*issues = append(*issues, fmt.Sprintf(
			"%s.data_base64 must decode to between 1 and %d bytes",
			label,
			maxImageFixtureBytes,
		))
		return fixture
	}
	fixture.Bytes = len(decoded)
	actualSHA256 := fmt.Sprintf("sha256:%x", sha256.Sum256(decoded))
	if fixture.SHA256 != actualSHA256 {
		*issues = append(*issues, fmt.Sprintf(
			"%s.sha256 mismatch: expected %q, actual %q",
			label,
			fixture.SHA256,
			actualSHA256,
		))
		return fixture
	}
	if err := validateImageFixturePayload(decoded, fixture.MediaType, label); err != nil {
		*issues = append(*issues, err.Error())
	}
	return fixture
}

func supportedImageFixtureMediaType(mediaType string) bool {
	switch mediaType {
	case "image/gif", "image/jpeg", "image/png", "image/webp":
		return true
	default:
		return false
	}
}

func validateProbeImageFixtureReferences(
	variant *probeVariant,
	imageFixtures map[string]probeImageFixture,
	variantLabel string,
	issues *[]string,
) {
	for messageIndex, message := range variant.Messages {
		content, ok := message["content"].([]any)
		if !ok {
			continue
		}
		for contentIndex, raw := range content {
			item, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			label := fmt.Sprintf(
				"%s.messages[%d].content[%d]",
				variantLabel,
				messageIndex,
				contentIndex,
			)
			itemType, _ := item["type"].(string)
			switch strings.ToLower(strings.TrimSpace(itemType)) {
			case "image_url", "input_image":
				*issues = append(
					*issues,
					label+" must use a declared image_fixture instead of an inline or remote image",
				)
				continue
			case "image_fixture":
				if itemType != "image_fixture" {
					*issues = append(*issues, label+`.type must be exactly "image_fixture"`)
					continue
				}
			default:
				continue
			}
			if len(item) != 2 {
				*issues = append(*issues, label+" image fixture reference must contain only type and fixture")
			}
			fixtureName, ok := item["fixture"].(string)
			fixtureName = strings.TrimSpace(fixtureName)
			if !ok || fixtureName == "" {
				*issues = append(*issues, label+".fixture must be a non-empty string")
				continue
			}
			if _, exists := imageFixtures[fixtureName]; !exists {
				*issues = append(*issues, fmt.Sprintf(
					"%s.fixture references unknown image fixture %q",
					label,
					fixtureName,
				))
			}
		}
	}
}

func validateProbeGeneratedText(variant *probeVariant, variantLabel string, issues *[]string) {
	generated := variant.GeneratedText
	label := variantLabel + ".generated_text"
	if len(variant.Messages) == 0 {
		*issues = append(*issues, label+" requires a messages probe")
		return
	}
	if generated.MessageIndex == nil {
		*issues = append(*issues, label+".message_index is required")
		return
	}
	if generated.ContentIndex == nil {
		*issues = append(*issues, label+".content_index is required")
		return
	}
	if generated.TargetTextBytes == nil {
		*issues = append(*issues, label+".target_text_bytes is required")
		return
	}
	if *generated.MessageIndex < 0 || *generated.MessageIndex >= len(variant.Messages) {
		*issues = append(*issues, label+".message_index is out of range")
		return
	}
	content, ok := variant.Messages[*generated.MessageIndex]["content"].([]any)
	if !ok {
		*issues = append(*issues, label+" requires the selected message content to be a list")
		return
	}
	if *generated.ContentIndex < 0 || *generated.ContentIndex > len(content) {
		*issues = append(*issues, label+".content_index is out of range")
	}
	if *generated.TargetTextBytes < 1 || *generated.TargetTextBytes > maxGeneratedTextBytes {
		*issues = append(*issues, fmt.Sprintf("%s.target_text_bytes must be between 1 and %d", label, maxGeneratedTextBytes))
	}
	if generated.Character == "" {
		generated.Character = "x"
	}
	if len(generated.Character) != 1 || generated.Character[0] < 0x20 || generated.Character[0] > 0x7e {
		*issues = append(*issues, label+".character must be one printable ASCII character")
	}
	existingTextBytes := messageContentTextBytes(content)
	if *generated.TargetTextBytes <= existingTextBytes {
		*issues = append(*issues, fmt.Sprintf(
			"%s.target_text_bytes must exceed the selected message's %d explicit text bytes",
			label,
			existingTextBytes,
		))
	}
}

func normalizedGeneratedText(source *probeGeneratedText) *GeneratedText {
	if source == nil || source.MessageIndex == nil || source.ContentIndex == nil || source.TargetTextBytes == nil {
		return nil
	}
	return &GeneratedText{
		MessageIndex:    *source.MessageIndex,
		ContentIndex:    *source.ContentIndex,
		TargetTextBytes: *source.TargetTextBytes,
		Character:       source.Character,
	}
}

func referencedImageFixtureMetadata(
	messages []map[string]any,
	fixtures map[string]probeImageFixture,
) map[string]ImageFixtureMetadata {
	metadata := map[string]ImageFixtureMetadata{}
	for _, message := range messages {
		content, ok := message["content"].([]any)
		if !ok {
			continue
		}
		for _, raw := range content {
			item, ok := raw.(map[string]any)
			if !ok || item["type"] != "image_fixture" {
				continue
			}
			name, _ := item["fixture"].(string)
			name = strings.TrimSpace(name)
			fixture, exists := fixtures[name]
			if !exists {
				continue
			}
			metadata[name] = ImageFixtureMetadata{
				Description: fixture.Description,
				MediaType:   fixture.MediaType,
				Bytes:       fixture.Bytes,
				SHA256:      fixture.SHA256,
			}
		}
	}
	if len(metadata) == 0 {
		return nil
	}
	return metadata
}
