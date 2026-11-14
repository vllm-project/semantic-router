package llmprotocol

import (
	"errors"
	"testing"
)

func TestVideoGenerationCapabilityAndValidation(t *testing.T) {
	set, err := ParseCapabilities([]string{"video_generation"})
	if err != nil || !set.Supports(CapabilityVideoGeneration) || !set.TaskCapabilities().Supports(CapabilityVideoGeneration) {
		t.Fatalf("video generation capability = %v, %v", set.Names(), err)
	}
	required := RequiredCapabilities(Request{VideoGeneration: &VideoGenerationOptions{}})
	if !required.Supports(CapabilityVideoGeneration) {
		t.Fatalf("video request requirements = %v", required.Names())
	}
	for _, test := range []struct {
		name  string
		video *GeneratedVideo
		code  string
	}{
		{name: "missing job", code: "generated_video_required"},
		{name: "unknown status", video: &GeneratedVideo{ID: "video_1", Status: "pending"}, code: "invalid_video_generation_status"},
		{name: "missing id", video: &GeneratedVideo{Status: VideoGenerationQueued}, code: "video_id_required"},
	} {
		err := ValidateGeneratedVideo(test.video, DefaultPolicy().Limits)
		var protocolErr *ProtocolError
		ok := errors.As(err, &protocolErr)
		if !ok || protocolErr.Code != test.code {
			t.Fatalf("%s: error = %v, want %s", test.name, err, test.code)
		}
	}
	if err := ValidateGeneratedVideo(&GeneratedVideo{Deleted: true, Status: VideoGenerationCompleted}, DefaultPolicy().Limits); err != nil {
		t.Fatalf("deleted video may omit id: %v", err)
	}
}

func TestGeneratedVideoIsClosedToItsContentKind(t *testing.T) {
	request := Request{
		Generation: 1,
		Model:      "route",
		Messages: []Message{{Role: RoleUser, Content: []Content{{
			Kind: ContentText, Text: "hi", GeneratedVideo: &GeneratedVideo{ID: "video_1", Status: VideoGenerationQueued},
		}}}},
	}
	if err := ValidateRequest(request, DefaultPolicy().Limits); err == nil {
		t.Fatal("text content carrying a generated video must fail validation")
	}
}
