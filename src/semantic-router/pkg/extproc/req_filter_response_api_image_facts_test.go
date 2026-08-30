package extproc

import (
	"context"
	"encoding/base64"
	"fmt"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// fakeImageFileResolver stands in for the router file store.
type fakeImageFileResolver struct {
	files map[string][]byte
}

func (f fakeImageFileResolver) ResolveImageFile(fileID string) ([]byte, error) {
	data, ok := f.files[fileID]
	if !ok {
		return nil, fmt.Errorf("file not found: %s", fileID)
	}
	return data, nil
}

var responseAPIPNGBytes = []byte("\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

var _ = Describe("Response API image inputs", func() {
	var filter *ResponseAPIFilter

	fileIDRequest := `{
		"model": "vision-model",
		"input": [{
			"type": "message",
			"role": "user",
			"content": [
				{"type": "input_text", "text": "what is shown here?"},
				{"type": "input_image", "file_id": "file-abc123"}
			]
		}]
	}`

	BeforeEach(func() {
		filter = NewResponseAPIFilter(NewMockResponseStore())
	})

	It("inlines file_id images from the file store and counts them once", func() {
		filter.SetImageFileResolver(fakeImageFileResolver{files: map[string][]byte{"file-abc123": responseAPIPNGBytes}})

		respCtx, translatedBody, err := filter.TranslateRequest(context.Background(), []byte(fileIDRequest))
		Expect(err).NotTo(HaveOccurred())
		Expect(respCtx.NativeImageContentCount).To(Equal(1))
		Expect(string(translatedBody)).To(ContainSubstring(`"url":"data:image/png;base64,`))
		Expect(string(translatedBody)).NotTo(ContainSubstring("file-abc123"), "file_id must not leak upstream")

		fast, err := extractContentFast(translatedBody)
		Expect(err).NotTo(HaveOccurred())
		Expect(fast.ImageContentCount).To(Equal(1), "inlined image should survive translation")

		ctx := &RequestContext{ResponseAPICtx: respCtx}
		mergeResponseAPINativeFacts(fast, ctx)
		Expect(fast.ImageContentCount).To(Equal(1), "native and translated counts must not add up")
	})

	It("rejects file_id images that no file store can resolve", func() {
		_, _, err := filter.TranslateRequest(context.Background(), []byte(fileIDRequest))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("file-abc123"))
	})

	It("rejects file_id images missing from the file store", func() {
		filter.SetImageFileResolver(fakeImageFileResolver{})

		_, _, err := filter.TranslateRequest(context.Background(), []byte(fileIDRequest))
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("file not found: file-abc123"))
	})

	It("inlines file_data images without a file store", func() {
		payload := base64.StdEncoding.EncodeToString(responseAPIPNGBytes)
		responseAPIReq := `{
			"model": "vision-model",
			"input": [{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_image", "file_data": "` + payload + `"}
				]
			}]
		}`

		respCtx, translatedBody, err := filter.TranslateRequest(context.Background(), []byte(responseAPIReq))
		Expect(err).NotTo(HaveOccurred())
		Expect(respCtx.NativeImageContentCount).To(Equal(1))
		Expect(string(translatedBody)).To(ContainSubstring(`"url":"data:image/png;base64,` + payload + `"`))

		fast, err := extractContentFast(translatedBody)
		Expect(err).NotTo(HaveOccurred())
		Expect(fast.ImageContentCount).To(Equal(1))
	})

	It("does not double-count URL images that survive translation", func() {
		responseAPIReq := `{
			"model": "vision-model",
			"input": [{
				"type": "message",
				"role": "user",
				"content": [
					{"type": "input_image", "image_url": "https://example.com/a.png"}
				]
			}]
		}`

		respCtx, translatedBody, err := filter.TranslateRequest(context.Background(), []byte(responseAPIReq))
		Expect(err).NotTo(HaveOccurred())
		Expect(respCtx.NativeImageContentCount).To(Equal(1))

		fast, err := extractContentFast(translatedBody)
		Expect(err).NotTo(HaveOccurred())
		Expect(fast.ImageContentCount).To(Equal(1))

		ctx := &RequestContext{ResponseAPICtx: respCtx}
		mergeResponseAPINativeFacts(fast, ctx)
		Expect(fast.ImageContentCount).To(Equal(1))
	})

	It("leaves non-Response-API requests untouched", func() {
		fast := &FastExtractResult{ImageContentCount: 2}
		mergeResponseAPINativeFacts(fast, &RequestContext{})
		Expect(fast.ImageContentCount).To(Equal(2))
		mergeResponseAPINativeFacts(fast, nil)
		Expect(fast.ImageContentCount).To(Equal(2))
	})
})
