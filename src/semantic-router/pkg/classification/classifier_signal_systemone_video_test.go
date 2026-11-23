package classification

import (
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A full-coverage video rule that saw a video it cannot decode reports the
// input unread, which on_unscanned resolves; a truncated rule never does.
func TestUnreadVideo(t *testing.T) {
	full := config.SystemOneRule{Coverage: config.SystemOneCoverageFull, Input: config.SystemOneInput{Video: true}}
	reading := systemOneReading{videos: []string{"", "data:video/mp4;base64,QUJD"}, videosIncomplete: true}
	if got := reading.unread(full); got != unreadVideo {
		t.Fatalf("unread = %q, want %q", got, unreadVideo)
	}
	if got := reading.unread(config.SystemOneRule{Input: config.SystemOneInput{Video: true}}); got != "" {
		t.Fatalf("a truncated rule reported %q", got)
	}

	// A data URL the codec reads decodes whatever its case or parameters; one it
	// rejects is reported unread instead of failing the rule.
	videos, incomplete := decodedVideos([]string{
		"",
		"data:video/mp4;base64,QUJD",
		"https://cdn.example/clip.mp4",
		"data:VIDEO/MP4;base64,QUJD",
		"data:video/mp4;codecs=avc1;base64,QUJD",
		"data:video/mp2t;base64,QUJD",
		"data:video/avi;base64,QUJD",
	})
	want := []string{"data:video/mp4;base64,QUJD", "data:VIDEO/MP4;base64,QUJD", "data:video/mp2t;base64,QUJD"}
	if !slices.Equal(videos, want) || !incomplete {
		t.Fatalf("decodedVideos = %q, %v; want %q", videos, incomplete, want)
	}
}
