package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

var _ = Describe("emitToolObservability", func() {
	var response *ext_proc.ProcessingResponse

	BeforeEach(func() {
		response = &ext_proc.ProcessingResponse{}
	})

	It("sets all three x-vsr-tools-* headers", func() {
		emitToolObservability(&response, "default", 0.87, 4*time.Millisecond)

		common := response.GetRequestBody().GetResponse()
		Expect(common).NotTo(BeNil())

		headerMap := make(map[string]string)
		for _, opt := range common.HeaderMutation.SetHeaders {
			headerMap[opt.Header.Key] = opt.Header.Value
		}

		Expect(headerMap[headers.VSRToolsStrategy]).To(Equal("default"))
		Expect(headerMap[headers.VSRToolsConfidence]).To(Equal("0.8700"))
		Expect(headerMap[headers.VSRToolsLatencyMs]).To(Equal("4"))
	})

	It("does nothing when strategy is empty", func() {
		var nilResponse *ext_proc.ProcessingResponse
		emitToolObservability(&nilResponse, "", 0, 0)
		Expect(nilResponse).To(BeNil())
	})
})
