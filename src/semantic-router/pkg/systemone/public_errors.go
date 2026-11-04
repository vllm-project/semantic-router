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
	api.InvalidRequest:       400,
	api.ModelNotFound:        404,
	api.RequestTooLarge:      413,
	api.UnsupportedMediaType: 415,
	api.UnsupportedSurface:   422,
	api.Overloaded:           429,
	api.NotReady:             503,
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

// limitedReasons are the request reasons whose public error carries the
// runtime's limit, as System One's request-level token error does.
var limitedReasons = map[string]bool{"request_tokens": true}

// runtimeFailure is the public status, code, message, reason and limit for a
// runtime's non-2xx answer: its documented code, for a rejected image or video
// the reason in the words above, and the limit of a limitedReasons refusal.
func runtimeFailure(status int, body []byte) (int, string, string, string, *int) {
	var answer api.ErrorResponse
	if json.Unmarshal(body, &answer) != nil {
		return status, "systemone_request_failed", "systemone request failed", "", nil
	}
	code := answer.Error.Code
	documented, ok := runtimeCodes[code]
	if !ok {
		return status, "systemone_request_failed", "systemone request failed", "", nil
	}
	if code == api.InvalidRequest || code == api.UnsupportedMediaType {
		reason := publicReason(answer.Error.Reason)
		var limit *int
		if limitedReasons[reason] && answer.Error.Limit != nil && *answer.Error.Limit > 0 {
			limit = answer.Error.Limit
		}
		return documented, string(code), mediaReason(answer.Error.Reason, answer.Error.Limit), reason, limit
	}
	return documented, string(code), strings.ReplaceAll(string(code), "_", " "), "", nil
}

// publicReason is a runtime reason a client may read: one of the fixed words
// worded here.
func publicReason(reason *api.ErrorBodyReason) string {
	if reason == nil {
		return ""
	}
	word := string(*reason)
	if _, known := mediaReasons[*reason]; known {
		return word
	}
	if _, known := requestReasons[word]; known {
		return word
	}
	if _, known := questionReasons[word]; known {
		return word
	}
	return ""
}

func mediaReason(reason *api.ErrorBodyReason, limit *int) string {
	if reason == nil {
		return "invalid request"
	}
	words, ok := mediaReasons[*reason]
	if !ok {
		if words, ok = requestReasons[string(*reason)]; ok {
			if limitedReasons[string(*reason)] && limit != nil && *limit > 0 {
				return fmt.Sprintf("%s (at most %s: the state once and every question's own)", words, thousands(*limit))
			}
			return words
		}
		if words, ok = questionReasons[string(*reason)]; ok {
			return "no question is valid: " + words
		}
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
