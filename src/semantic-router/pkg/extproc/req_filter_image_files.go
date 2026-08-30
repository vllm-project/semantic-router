package extproc

import (
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

// Error codes for image file references the client can correct. They are the
// only dispatch-preparation failures rendered as a client 400; every other
// error keeps the gRPC failure path.
const (
	imageFileNotFoundCode    = "image_file_not_found"
	imageFileEmptyCode       = "image_file_empty"
	imageFileUnsupportedCode = "image_file_unsupported"
	imageFileLimitCode       = "image_file_limit"
)

// resolveImageFileReferences inlines image content that references a file held
// by the Router's own file store (POST /v1/files with purpose=vision) so the
// selected backend receives the image bytes that drove routing instead of a
// Router-local identifier. It runs after retained Responses history has been
// materialized, so images referenced by earlier turns are covered too.
// When the store is enabled, an image file it does not hold is rejected: the
// backend could never fetch a Router-local file, so failing here names the
// file instead of surfacing a provider error later. Without a store every
// reference passes through untouched and the target codec decides whether it
// can carry provider file IDs.
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
				return changed, llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, imageFileNotFoundCode,
					fmt.Sprintf("uploaded file %q was not found in the router file store", part.FileID), err)
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
		return "", "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, imageFileEmptyCode,
			fmt.Sprintf("uploaded file %q is empty", fileID), nil)
	}
	mediaType := http.DetectContentType(data)
	if !imageurl.IsAllowedMIME(mediaType) {
		return "", "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, imageFileUnsupportedCode,
			fmt.Sprintf("uploaded file %q is %s; png, jpeg, gif, and webp images are accepted", fileID, mediaType), nil)
	}
	if limit := llmprotocol.DefaultPolicy().Limits.MediaDataBytes; limit > 0 && base64.StdEncoding.EncodedLen(len(data)) > limit {
		return "", "", llmprotocol.NewError(llmprotocol.ErrorInvalidRequest, imageFileLimitCode,
			fmt.Sprintf("uploaded file %q exceeds the inline media limit", fileID), nil)
	}
	return mediaType, base64.StdEncoding.EncodeToString(data), nil
}

// imageFileDispatchFailure converts an image file rejection raised while
// preparing the provider dispatch into an immediate 400 rendered in the
// client's wire format. Any other error is returned unchanged.
func (r *OpenAIRouter) imageFileDispatchFailure(err error, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || !isImageFileRequestError(protocolError) {
		return nil, err
	}
	metrics.RecordRequestError(ctx.RequestModel, "invalid_request")
	ctx.ImmediateProtocolError = protocolError
	return r.createErrorResponse(http.StatusBadRequest, protocolError.Message), nil
}

func isImageFileRequestError(protocolError *llmprotocol.ProtocolError) bool {
	switch protocolError.Code {
	case imageFileNotFoundCode, imageFileEmptyCode, imageFileUnsupportedCode, imageFileLimitCode:
		return protocolError.Category == llmprotocol.ErrorInvalidRequest
	default:
		return false
	}
}
