package classification

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// TestEveryWindowOfAMediaStateKeepsItsMedia covers issue #6340: a guard rule
// that reads image or video input sees it on the one-window path, but each
// window of a longer input was built without those fields, so the media
// silently stopped being evaluated as soon as the text crossed the scan
// window.
func TestEveryWindowOfAMediaStateKeepsItsMedia(t *testing.T) {
	images := []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="}
	videos := []string{"data:video/mp4;base64,AAAAIGZ0eXBpc29t"}
	text := strings.Repeat("The clerks counted 412 barges this spring. ", 60)
	answer := strings.Repeat("The story ends where it began, at the river. ", 120)
	budget := scanBudget{size: 256, most: 256, overlap: 32, parallel: 2}

	tests := []struct {
		name       string
		state      modelservice.Request
		wantImages []string
		wantVideos []string
	}{
		{
			name:       "a state of text with an image",
			state:      modelservice.Request{State: text, Images: images},
			wantImages: images,
		},
		{
			name:       "a state of text with a video",
			state:      modelservice.Request{State: text, Videos: videos},
			wantVideos: videos,
		},
		{
			name: "a state of parts with an image and a video",
			state: modelservice.Request{
				Parts:  map[string]string{"history": text, "request": "Summarize the chapter above."},
				Images: images, Videos: videos,
			},
			wantImages: images, wantVideos: videos,
		},
		{
			name: "a response answer with an image",
			state: modelservice.Request{
				Parts:  map[string]string{"answer": answer, "request": "Tell me a story."},
				Images: images,
			},
			wantImages: images,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scan := newGuardScan("decision", test.state, budget)
			if scan == nil {
				t.Fatal("a state larger than one window must be scanned")
			}
			calls := scan.windowCalls(context.Background())
			if len(calls) < 2 {
				t.Fatalf("windows = %d, want the input cut into several", len(calls))
			}
			for _, call := range calls {
				if !slices.Equal(call.request.Images, test.wantImages) {
					t.Fatalf("window %d images = %v, want %v", call.window, call.request.Images, test.wantImages)
				}
				if !slices.Equal(call.request.Videos, test.wantVideos) {
					t.Fatalf("window %d videos = %v, want %v", call.window, call.request.Videos, test.wantVideos)
				}
				// The media travels with its text: a window still reads text.
				if call.request.State == "" && len(call.request.Parts) == 0 {
					t.Fatalf("window %d carries media but no text", call.window)
				}
			}
		})
	}
}

// TestAWindowOfTextOnlyKeepsNoMedia pins the other direction: windowing must
// not invent media for a rule that reads none.
func TestAWindowOfTextOnlyKeepsNoMedia(t *testing.T) {
	text := strings.Repeat("The committee reviewed the quarterly report. ", 60)
	scan := newGuardScan("decision", modelservice.Request{State: text}, scanBudget{size: 256, most: 256, overlap: 32, parallel: 1})
	if scan == nil {
		t.Fatal("a state larger than one window must be scanned")
	}
	for _, call := range scan.windowCalls(context.Background()) {
		if len(call.request.Images) != 0 || len(call.request.Videos) != 0 {
			t.Fatalf("window %d invented media: %v %v", call.window, call.request.Images, call.request.Videos)
		}
	}
}

// TestAStateWithinOneWindowIsNotWindowed records that the non-windowed path is
// unchanged: a media-bearing state that fits its deployment's window is handed
// to the model as it is, media included.
func TestAStateWithinOneWindowIsNotWindowed(t *testing.T) {
	media := modelservice.Request{
		State:  "A short turn.",
		Images: []string{"data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="},
	}
	if scan := newGuardScan("decision", media, scanBudget{size: 1024, most: 1024, overlap: 16, parallel: 1}); scan != nil {
		t.Fatal("a state within one window must not be windowed")
	}
}
