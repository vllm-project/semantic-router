package extproc

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Response API native image facts", func() {
	var filter *ResponseAPIFilter

	BeforeEach(func() {
		filter = NewResponseAPIFilter(NewMockResponseStore())
	})

	It("counts file_id images that translation cannot represent", func() {
		responseAPIReq := `{
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

		respCtx, translatedBody, err := filter.TranslateRequest(context.Background(), []byte(responseAPIReq))
		Expect(err).NotTo(HaveOccurred())
		Expect(respCtx.NativeImageContentCount).To(Equal(1))

		fast, err := extractContentFast(translatedBody)
		Expect(err).NotTo(HaveOccurred())
		Expect(fast.ImageContentCount).To(Equal(0), "file_id image should not survive translation")

		ctx := &RequestContext{ResponseAPICtx: respCtx}
		mergeResponseAPINativeFacts(fast, ctx)
		Expect(fast.ImageContentCount).To(Equal(1), "native count should restore the dropped image")
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
