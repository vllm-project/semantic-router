package extproc

import (
	"context"
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Response API Stream Flag Injection", func() {
	var (
		filter    *ResponseAPIFilter
		mockStore *MockResponseStore
	)

	BeforeEach(func() {
		mockStore = NewMockResponseStore()
		filter = NewResponseAPIFilter(mockStore)
	})

	It("should include stream flag in translated request when streaming", func() {
		responseAPIReq := `{
			"model": "gpt-4",
			"input": "Hello",
			"stream": true
		}`

		_, translatedBody, err := filter.TranslateRequest(context.Background(), []byte(responseAPIReq))
		Expect(err).NotTo(HaveOccurred())

		var chatReq map[string]interface{}
		err = json.Unmarshal(translatedBody, &chatReq)
		Expect(err).NotTo(HaveOccurred())
		Expect(chatReq["stream"]).To(Equal(true))
		Expect(chatReq).To(HaveKey("stream_options"))
	})

	It("should omit stream flag when not streaming", func() {
		responseAPIReq := `{
			"model": "gpt-4",
			"input": "Hello"
		}`

		_, translatedBody, err := filter.TranslateRequest(context.Background(), []byte(responseAPIReq))
		Expect(err).NotTo(HaveOccurred())

		var chatReq map[string]interface{}
		err = json.Unmarshal(translatedBody, &chatReq)
		Expect(err).NotTo(HaveOccurred())
		Expect(chatReq).NotTo(HaveKey("stream"))
		Expect(chatReq).NotTo(HaveKey("stream_options"))
	})
})
