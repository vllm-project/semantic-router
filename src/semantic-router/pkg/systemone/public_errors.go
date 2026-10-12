package systemone

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// runtimeCodes are the model runtime's documented request-level error codes
// and their statuses. Runtime messages may contain local model paths or
// deployment keys, so a client gets one of these codes with words written
// here, never the runtime's message.
var runtimeCodes = map[api.ErrorBodyCode]int{
	api.InvalidRequest:     400,
	api.ModelNotFound:      404,
	api.RequestTooLarge:    413,
	api.UnsupportedSurface: 422,
	api.Overloaded:         429,
	api.NotReady:           503,
}

// mediaReasons word the runtime's image and video rejection reasons; %s takes
// the limit the runtime reports with the reason.
var mediaReasons = map[api.ErrorBodyReason]string{
	api.ImageFormat:      "image must be a base64 PNG, JPEG or WebP data URL",
	api.ImageTooLarge:    "image exceeds %s",
	api.ImagePixels:      "image exceeds %s pixels",
	api.ImageAspectRatio: "image aspect ratio exceeds %s:1",
	api.ImageMalformed:   "malformed image",
	api.ImageUnsupported: "model does not read images",
	api.VideoFormat:      "video must be a base64 MP4, WebM, QuickTime or Matroska data URL",
	api.VideoTooLarge:    "video exceeds %s",
	api.VideoTooLong:     "video longer than %s s",
	api.VideoPixels:      "video frames exceed %s pixels",
	api.VideoAspectRatio: "video aspect ratio exceeds %s:1",
	api.VideoMalformed:   "malformed video",
	api.VideoTokens:      "videos exceed %s input tokens",
	api.VideoUnsupported: "model does not read videos",
}

// runtimeFailure is the public status, code and message for a runtime's
// non-2xx answer: its documented code, and for a rejected image or video the
// reason in the words above.
func runtimeFailure(status int, body []byte) (int, string, string) {
	var answer api.ErrorResponse
	if json.Unmarshal(body, &answer) != nil {
		return status, "systemone_request_failed", "systemone request failed"
	}
	code := answer.Error.Code
	documented, ok := runtimeCodes[code]
	if !ok {
		return status, "systemone_request_failed", "systemone request failed"
	}
	if code == api.InvalidRequest {
		return documented, string(code), mediaReason(answer.Error.Reason, answer.Error.Limit)
	}
	return documented, string(code), strings.ReplaceAll(string(code), "_", " ")
}

func mediaReason(reason *api.ErrorBodyReason, limit *int) string {
	if reason == nil {
		return "invalid request"
	}
	words, ok := mediaReasons[*reason]
	if !ok {
		return "invalid request"
	}
	if !strings.Contains(words, "%s") {
		return words
	}
	if limit == nil || *limit <= 0 {
		return "invalid request"
	}
	if *reason == api.ImageTooLarge || *reason == api.VideoTooLarge {
		return fmt.Sprintf(words, byteSize(*limit))
	}
	return fmt.Sprintf(words, thousands(*limit))
}

// byteSize writes whole megabytes as the runtime's limits are set, and other
// sizes in bytes.
func byteSize(n int) string {
	if n%1_000_000 == 0 {
		return strconv.Itoa(n/1_000_000) + " MB"
	}
	return thousands(n) + " bytes"
}

func thousands(n int) string {
	digits := strconv.Itoa(n)
	var out strings.Builder
	for index, digit := range digits {
		if index > 0 && (len(digits)-index)%3 == 0 {
			out.WriteByte(',')
		}
		out.WriteRune(digit)
	}
	return out.String()
}
