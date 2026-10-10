package extproc

import (
	"encoding/base64"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A 1x1 PNG; small enough to inline and a valid image for the safety gate.
var tinyPNG = base64.StdEncoding.EncodeToString([]byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
	0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4,
	0x89, 0x00, 0x00, 0x00, 0x0a, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
	0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae,
	0x42, 0x60, 0x82,
})

func userImageRequest(content llmprotocol.Content) *llmprotocol.Request {
	return &llmprotocol.Request{
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleUser,
			Content: []llmprotocol.Content{
				{Kind: llmprotocol.ContentText, Text: "What is shown in this image?"},
				content,
			},
		}},
	}
}

// Inline data-URI images arrive from the codec as MediaType+Data with an empty
// URL; the snapshot must rebuild the data URI or image-modality rules never run.
func TestExtractSemanticRequestSignals_InlineImageBecomesDataURL(t *testing.T) {
	snapshot := extractSemanticRequestSignals(userImageRequest(llmprotocol.Content{
		Kind: llmprotocol.ContentImage, MediaType: "image/png", Data: tinyPNG,
	}))
	want := "data:image/png;base64," + tinyPNG
	if snapshot.FirstImageURL != want {
		t.Fatalf("FirstImageURL = %q, want inline data URI", snapshot.FirstImageURL)
	}
	if snapshot.ImageContentCount != 1 {
		t.Fatalf("ImageContentCount = %d, want 1", snapshot.ImageContentCount)
	}
}

// Remote URLs must never reach the image encoder (SSRF / local file guard).
func TestExtractSemanticRequestSignals_RemoteImageURLStaysUnavailable(t *testing.T) {
	snapshot := extractSemanticRequestSignals(userImageRequest(llmprotocol.Content{
		Kind: llmprotocol.ContentImage, URL: "https://example.com/passport.png",
	}))
	if snapshot.FirstImageURL != "" {
		t.Fatalf("FirstImageURL = %q, remote URL should remain unavailable to embeddings", snapshot.FirstImageURL)
	}
	if snapshot.ImageContentCount != 1 {
		t.Fatalf("ImageContentCount = %d, want 1 (still counted for input_modality)", snapshot.ImageContentCount)
	}
}

// Non-image inline payloads are rejected by the media-type allowlist.
func TestExtractSemanticRequestSignals_NonImageInlinePayloadRejected(t *testing.T) {
	snapshot := extractSemanticRequestSignals(userImageRequest(llmprotocol.Content{
		Kind: llmprotocol.ContentImage, MediaType: "text/html", Data: base64.StdEncoding.EncodeToString([]byte("<html></html>")),
	}))
	if snapshot.FirstImageURL != "" {
		t.Fatalf("FirstImageURL = %q, non-image payload should be rejected", snapshot.FirstImageURL)
	}
}
