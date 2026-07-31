package extproc

import "testing"

func TestExtractContentFastMetadataAndImageCount(t *testing.T) {
	body := []byte(`{
		"model":"vllm-sr/test",
		"metadata":{"consent":"denied","cohort":"canary"},
		"messages":[{"role":"user","content":[
			{"type":"text","text":"hello"},
			{"type":"image_url","image_url":{"url":"https://example.com/image.png"}}
		]}]
	}`)

	result, err := extractContentFast(body)
	if err != nil {
		t.Fatalf("extractContentFast() error = %v", err)
	}
	if result.Metadata["consent"] != "denied" {
		t.Fatalf("metadata = %v", result.Metadata)
	}
	if result.ImageContentCount != 1 {
		t.Fatalf("ImageContentCount = %d, want 1", result.ImageContentCount)
	}
	if result.FirstImageURL != "" {
		t.Fatalf("FirstImageURL = %q, remote URL should remain unavailable to embeddings", result.FirstImageURL)
	}
}

func TestExtractContentFastRejectsNonStringMetadata(t *testing.T) {
	body := []byte(`{"model":"m","metadata":{"cohort":1},"messages":[]}`)
	if _, err := extractContentFast(body); err == nil {
		t.Fatal("extractContentFast() expected metadata type error")
	}
}
