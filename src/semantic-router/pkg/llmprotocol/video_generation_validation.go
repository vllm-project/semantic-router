package llmprotocol

import "strings"

func validateGeneratedVideoContent(content Content, limits Limits) error {
	if generatedVideoContentHasForeignFields(content) {
		return NewError(ErrorInvalidRequest, "invalid_generated_video", "generated video content contains fields from another content kind", nil)
	}
	return ValidateGeneratedVideo(content.GeneratedVideo, limits)
}

func generatedVideoContentHasForeignFields(content Content) bool {
	return content.Text != "" || content.ToolCall != nil || content.ToolResult != nil || content.URL != "" ||
		content.Data != "" || content.FileID != "" || content.Filename != "" || content.MediaType != "" ||
		content.Detail != "" || content.Signature != "" || content.Reasoning != "" || content.Cache != nil ||
		len(content.Citations) != 0 || content.GeneratedImage != nil || content.ReasoningDetail != nil
}

// ValidateGeneratedVideo applies the neutral video-job lifecycle contract.
func ValidateGeneratedVideo(video *GeneratedVideo, limits Limits) error {
	if video == nil {
		return NewError(ErrorInvalidRequest, "generated_video_required", "generated video content requires video-job state", nil)
	}
	if !ValidVideoGenerationStatus(video.Status) {
		return NewError(ErrorInvalidRequest, "invalid_video_generation_status", "video generation status is invalid", nil)
	}
	if strings.TrimSpace(video.ID) == "" && !video.Deleted {
		return NewError(ErrorInvalidRequest, "video_id_required", "video generation requires an id", nil)
	}
	for _, value := range []string{video.ID, video.Seconds, video.Size} {
		if exceeds(value, limits.MediaReferenceBytes) {
			return NewError(ErrorInvalidRequest, "generated_video_field_limit", "generated video metadata exceeds the configured limit", nil)
		}
	}
	return nil
}

// ValidVideoGenerationStatus reports whether a job state is one this contract
// models, so a codec can refuse an unknown one before building an artifact.
func ValidVideoGenerationStatus(status VideoGenerationStatus) bool {
	switch status {
	case VideoGenerationQueued, VideoGenerationInProgress, VideoGenerationCompleted, VideoGenerationFailed:
		return true
	default:
		return false
	}
}
