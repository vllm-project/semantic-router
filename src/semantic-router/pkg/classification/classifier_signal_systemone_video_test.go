package classification

import (
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

	videos, incomplete := decodedVideos([]string{"", "data:video/mp4;base64,QUJD", "https://cdn.example/clip.mp4"})
	if len(videos) != 1 || videos[0] != "data:video/mp4;base64,QUJD" || !incomplete {
		t.Fatalf("decodedVideos = %q, %v", videos, incomplete)
	}
}
