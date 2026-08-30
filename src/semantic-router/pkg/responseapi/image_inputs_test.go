package responseapi

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	pngBytes  = []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
	jpegBytes = []byte("\xff\xd8\xff\xe0\x00\x10JFIF")
)

type fakeImageFileResolver struct {
	files map[string][]byte
}

func (f fakeImageFileResolver) ResolveImageFile(fileID string) ([]byte, error) {
	data, ok := f.files[fileID]
	if !ok {
		return nil, fmt.Errorf("file not found: %s", fileID)
	}
	return data, nil
}

func TestImagePartURL_URLPassesThrough(t *testing.T) {
	url, err := imagePartURL(ContentPart{Type: ContentTypeInputImage, ImageURL: "https://example.com/a.png"}, nil)
	require.NoError(t, err)
	assert.Equal(t, "https://example.com/a.png", url)
}

func TestImagePartURL_NoImageContent(t *testing.T) {
	url, err := imagePartURL(ContentPart{Type: ContentTypeInputImage}, nil)
	require.NoError(t, err)
	assert.Empty(t, url)
}

func TestImagePartURL_FileDataDataURLCanonicalized(t *testing.T) {
	payload := base64.StdEncoding.EncodeToString(pngBytes)
	url, err := imagePartURL(ContentPart{Type: ContentTypeInputImage, FileData: "DATA:IMAGE/PNG;BASE64," + payload}, nil)
	require.NoError(t, err)
	assert.Equal(t, "data:image/png;base64,"+payload, url)
}

func TestImagePartURL_FileDataRawBase64Sniffed(t *testing.T) {
	payload := base64.StdEncoding.EncodeToString(jpegBytes)
	url, err := imagePartURL(ContentPart{Type: ContentTypeInputImage, FileData: payload}, nil)
	require.NoError(t, err)
	assert.Equal(t, "data:image/jpeg;base64,"+payload, url)
}

func TestImagePartURL_FileDataRejections(t *testing.T) {
	cases := map[string]string{
		"non-image bytes":    base64.StdEncoding.EncodeToString([]byte("hello, world")),
		"invalid base64":     "not base64!!",
		"non-image data URL": "data:application/pdf;base64,AAAA",
	}
	for name, fileData := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := imagePartURL(ContentPart{Type: ContentTypeInputImage, FileData: fileData}, nil)
			var imageErr *ImageInputError
			require.ErrorAs(t, err, &imageErr)
			assert.Empty(t, imageErr.FileID)
		})
	}
}

func TestImagePartURL_FileIDResolved(t *testing.T) {
	resolver := fakeImageFileResolver{files: map[string][]byte{"file-1": pngBytes}}
	url, err := imagePartURL(ContentPart{Type: ContentTypeInputImage, FileID: "file-1"}, resolver)
	require.NoError(t, err)
	assert.Equal(t, "data:image/png;base64,"+base64.StdEncoding.EncodeToString(pngBytes), url)
}

func TestImagePartURL_FileIDWithoutResolver(t *testing.T) {
	_, err := imagePartURL(ContentPart{Type: ContentTypeInputImage, FileID: "file-1"}, nil)
	var imageErr *ImageInputError
	require.ErrorAs(t, err, &imageErr)
	assert.Equal(t, "file-1", imageErr.FileID)
	assert.Contains(t, err.Error(), "file-1")
}

func TestImagePartURL_FileIDNotFound(t *testing.T) {
	_, err := imagePartURL(ContentPart{Type: ContentTypeInputImage, FileID: "file-missing"}, fakeImageFileResolver{})
	var imageErr *ImageInputError
	require.ErrorAs(t, err, &imageErr)
	assert.Contains(t, err.Error(), "file not found: file-missing")
}

func TestImagePartURL_FileIDNotAnImage(t *testing.T) {
	resolver := fakeImageFileResolver{files: map[string][]byte{"file-txt": []byte("plain text notes")}}
	_, err := imagePartURL(ContentPart{Type: ContentTypeInputImage, FileID: "file-txt"}, resolver)
	var imageErr *ImageInputError
	require.ErrorAs(t, err, &imageErr)
	assert.Contains(t, err.Error(), "unsupported image type")
}

func TestImagePartURL_FileIDTooLarge(t *testing.T) {
	huge := append([]byte{}, pngBytes...)
	huge = append(huge, make([]byte, maxInlineImageBytes)...)
	resolver := fakeImageFileResolver{files: map[string][]byte{"file-big": huge}}
	_, err := imagePartURL(ContentPart{Type: ContentTypeInputImage, FileID: "file-big"}, resolver)
	var imageErr *ImageInputError
	require.ErrorAs(t, err, &imageErr)
	assert.Contains(t, err.Error(), "the limit is")
}

func imageInput(part string) json.RawMessage {
	return json.RawMessage(`[{
		"type": "message",
		"role": "user",
		"content": [
			{"type": "input_text", "text": "What is in this image?"},
			` + part + `
		]
	}]`)
}

func TestTranslateToCompletionRequest_FileIDImageInlined(t *testing.T) {
	tr := NewTranslator()
	tr.SetImageFileResolver(fakeImageFileResolver{files: map[string][]byte{"file-abc": pngBytes}})
	req := &ResponseAPIRequest{
		Model: "vision-model",
		Input: imageInput(`{"type": "input_image", "file_id": "file-abc", "detail": "high"}`),
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	parts := result.Messages[0].OfUser.Content.OfArrayOfContentParts
	require.Len(t, parts, 2)
	assert.Equal(t, "What is in this image?", parts[0].OfText.Text)
	require.NotNil(t, parts[1].OfImageURL)
	assert.True(t, strings.HasPrefix(parts[1].OfImageURL.ImageURL.URL, "data:image/png;base64,"))
	assert.Equal(t, "high", parts[1].OfImageURL.ImageURL.Detail)
}

func TestTranslateToCompletionRequest_FileDataImageInlined(t *testing.T) {
	tr := NewTranslator()
	payload := base64.StdEncoding.EncodeToString(jpegBytes)
	req := &ResponseAPIRequest{
		Model: "vision-model",
		Input: imageInput(`{"type": "input_image", "file_data": "` + payload + `"}`),
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)
	parts := result.Messages[0].OfUser.Content.OfArrayOfContentParts
	require.Len(t, parts, 2)
	assert.Equal(t, "data:image/jpeg;base64,"+payload, parts[1].OfImageURL.ImageURL.URL)
}

func TestTranslateToCompletionRequest_UnresolvableFileIDRejected(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "vision-model",
		Input: imageInput(`{"type": "input_image", "file_id": "file-abc"}`),
	}

	_, err := tr.TranslateToCompletionRequest(req, nil)
	var imageErr *ImageInputError
	require.ErrorAs(t, err, &imageErr)
	assert.Equal(t, "file-abc", imageErr.FileID)
	assert.NotContains(t, err.Error(), "failed to parse input")
}

func TestTranslateToCompletionRequest_NonUserFileIDImageDropped(t *testing.T) {
	tr := NewTranslator()
	req := &ResponseAPIRequest{
		Model: "vision-model",
		Input: json.RawMessage(`[{
			"type": "message",
			"role": "assistant",
			"content": [
				{"type": "output_text", "text": "Earlier answer."},
				{"type": "input_image", "file_id": "file-abc"}
			]
		}]`),
	}

	result, err := tr.TranslateToCompletionRequest(req, nil)
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	assert.Equal(t, "Earlier answer.", result.Messages[0].OfAssistant.Content.OfString.Value)
}

func TestTranslateToCompletionRequest_HistoryWithUnresolvableFileIDSkipped(t *testing.T) {
	tr := NewTranslator()
	history := []*StoredResponse{{
		Input: []InputItem{{
			Type:    ItemTypeMessage,
			Role:    RoleUser,
			Content: json.RawMessage(`[{"type": "input_image", "file_id": "file-gone"}]`),
		}},
	}}
	req := &ResponseAPIRequest{
		Model: "vision-model",
		Input: json.RawMessage(`"follow-up question"`),
	}

	result, err := tr.TranslateToCompletionRequest(req, history)
	require.NoError(t, err)
	require.Len(t, result.Messages, 1)
	assert.Equal(t, "follow-up question", result.Messages[0].OfUser.Content.OfString.Value)
}
