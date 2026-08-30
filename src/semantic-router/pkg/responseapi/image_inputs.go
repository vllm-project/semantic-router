package responseapi

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/imageurl"
)

// maxInlineImageBytes caps the decoded size of an image the translator inlines
// as a data URL. It matches the per-image limit of the OpenAI vision input
// contract and keeps one file_id from ballooning the upstream body.
const maxInlineImageBytes = 20 << 20

// ImageFileResolver returns the bytes of an uploaded file by its file_id so an
// input_image part that references a file can be delivered to Chat Completions
// backends, which only accept images as URLs.
type ImageFileResolver interface {
	ResolveImageFile(fileID string) ([]byte, error)
}

// ImageInputError reports an input_image part the translator could not turn
// into a Chat Completions image_url part. It reaches the client as a 400
// because silently dropping the part would send a vision request without its
// image.
type ImageInputError struct {
	FileID string
	Reason string
}

func (e *ImageInputError) Error() string {
	if e.FileID != "" {
		return fmt.Sprintf("input_image file_id %q: %s", e.FileID, e.Reason)
	}
	return "input_image: " + e.Reason
}

// isImagePart reports whether a content part carries image content in any of
// the forms the Response API accepts: a URL or data URL, inline file_data, or
// a file_id reference.
func isImagePart(part ContentPart) bool {
	return part.Type == ContentTypeInputImage &&
		(part.ImageURL != "" || part.FileID != "" || part.FileData != "")
}

// imagePartURL returns the value to place in a Chat Completions image_url part
// for an input_image part. URL images pass through untouched; file_data is
// normalized to an allowlisted image data URL; file_id is resolved through the
// router file store and inlined the same way. An empty string means the part
// carries no image.
func imagePartURL(part ContentPart, resolver ImageFileResolver) (string, error) {
	switch {
	case part.ImageURL != "":
		return part.ImageURL, nil
	case part.FileData != "":
		return inlineImageData(part.FileData)
	case part.FileID != "":
		if resolver == nil {
			return "", &ImageInputError{
				FileID: part.FileID,
				Reason: "file_id images cannot be resolved on this router; send the image as image_url or file_data",
			}
		}
		data, err := resolver.ResolveImageFile(part.FileID)
		if err != nil {
			return "", &ImageInputError{FileID: part.FileID, Reason: err.Error()}
		}
		return imageBytesDataURL(data, part.FileID)
	default:
		return "", nil
	}
}

// inlineImageData accepts either a complete image data URL or bare base64
// image bytes and returns a canonical allowlisted data URL.
func inlineImageData(fileData string) (string, error) {
	if data, ok := imageurl.DecodeBase64(fileData); ok {
		if len(data) > maxInlineImageBytes {
			return "", &ImageInputError{Reason: oversizeReason(len(data))}
		}
		canonical, _ := imageurl.CanonicalDataURL(fileData)
		return canonical, nil
	}
	if strings.HasPrefix(strings.ToLower(strings.TrimSpace(fileData)), "data:") {
		return "", &ImageInputError{Reason: "file_data must be a base64 data URL of a png, jpeg, gif, or webp image"}
	}
	data, err := base64.StdEncoding.DecodeString(strings.TrimSpace(fileData))
	if err != nil {
		return "", &ImageInputError{Reason: "file_data is not valid base64"}
	}
	return imageBytesDataURL(data, "")
}

// imageBytesDataURL sniffs the image type from the bytes and builds an
// allowlisted data URL.
func imageBytesDataURL(data []byte, fileID string) (string, error) {
	if len(data) == 0 {
		return "", &ImageInputError{FileID: fileID, Reason: "image is empty"}
	}
	if len(data) > maxInlineImageBytes {
		return "", &ImageInputError{FileID: fileID, Reason: oversizeReason(len(data))}
	}
	mime := http.DetectContentType(data)
	url := "data:" + mime + ";base64," + base64.StdEncoding.EncodeToString(data)
	if !imageurl.IsSafeImageDataURL(url) {
		return "", &ImageInputError{
			FileID: fileID,
			Reason: fmt.Sprintf("unsupported image type %q; png, jpeg, gif, and webp are accepted", mime),
		}
	}
	return url, nil
}

func oversizeReason(size int) string {
	return fmt.Sprintf("image is %d bytes; the limit is %d", size, maxInlineImageBytes)
}
