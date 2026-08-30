package extproc

import (
	"encoding/base64"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

// resolveImageFileReferences inlines image content that references a file held
// by the Router's own file store (POST /v1/files with purpose=vision) so the
// selected backend receives the image bytes that drove routing instead of a
// Router-local identifier. It runs after retained Responses history has been
// materialized, so images referenced by earlier turns are covered too.
// References the store does not hold are left untouched: they may name a file
// on the provider's Files API, and the target codec decides whether it can
// carry them.
func (r *OpenAIRouter) resolveImageFileReferences(request *llmprotocol.Request) (bool, error) {
	if request == nil {
		return false, nil
	}
	fileStore := r.currentFileStore()
	if fileStore == nil {
		return false, nil
	}
	changed := false
	for messageIndex := range request.Messages {
		content := request.Messages[messageIndex].Content
		for contentIndex := range content {
			part := &content[contentIndex]
			if part.Kind != llmprotocol.ContentImage || part.FileID == "" || part.URL != "" || part.Data != "" {
				continue
			}
			if _, err := fileStore.Get(part.FileID); err != nil {
				continue
			}
			data, err := fileStore.Read(part.FileID)
			if err != nil {
				return changed, llmprotocol.NewError(llmprotocol.ErrorInternal, "image_file_read", "uploaded image could not be read", err)
			}
			mediaType, encoded, err := inlineImageFile(part.FileID, data)
			if err != nil {
				return changed, err
			}
			part.MediaType, part.Data, part.FileID = mediaType, encoded, ""
			changed = true
		}
	}
	return changed, nil
}

// inlineImageFile sniffs the image type from the stored bytes and returns the
// media type and base64 payload for neutral inline image content. The size
// check mirrors the protocol policy so the request fails here with a clear
// reason rather than at provider encoding.
func inlineImageFile(fileID string, data []byte) (string, string, error) {
	if len(data) == 0 {
		return "", "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_file_empty",
			fmt.Sprintf("uploaded file %q is empty", fileID), nil)
	}
	mediaType := http.DetectContentType(data)
	if !imageurl.IsAllowedMIME(mediaType) {
		return "", "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_file_unsupported",
			fmt.Sprintf("uploaded file %q is %s; png, jpeg, gif, and webp images are accepted", fileID, mediaType), nil)
	}
	if limit := llmprotocol.DefaultPolicy().Limits.MediaDataBytes; limit > 0 && base64.StdEncoding.EncodedLen(len(data)) > limit {
		return "", "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, "image_file_limit",
			fmt.Sprintf("uploaded file %q exceeds the inline media limit", fileID), nil)
	}
	return mediaType, base64.StdEncoding.EncodeToString(data), nil
}
