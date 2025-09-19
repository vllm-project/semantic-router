package extproc

import (
	"encoding/json"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
)

func getHistogramSampleCount(metricName, model string) uint64 {
	mf, _ := prometheus.DefaultGatherer.Gather()
	for _, fam := range mf {
		if fam.GetName() != metricName || fam.GetType() != dto.MetricType_HISTOGRAM {
			continue
		}
		for _, m := range fam.GetMetric() {
			labels := m.GetLabel()
			match := false
			for _, l := range labels {
				if l.GetName() == "model" && l.GetValue() == model {
					match = true
					break
				}
			}
			if match {
				h := m.GetHistogram()
				if h != nil && h.SampleCount != nil {
					return h.GetSampleCount()
				}
			}
		}
	}
	return 0
}

var _ = Describe("Metrics recording", func() {
	var router *OpenAIRouter

	BeforeEach(func() {
		// Use a minimal router that doesn't require external models
		router = &OpenAIRouter{
			Cache: cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false}),
		}
	})

	It("records TTFT on response headers", func() {
		ctx := &RequestContext{
			RequestModel:        "model-a",
			ProcessingStartTime: time.Now().Add(-75 * time.Millisecond),
		}

		before := getHistogramSampleCount("llm_model_ttft_seconds", ctx.RequestModel)

		respHeaders := &ext_proc.ProcessingRequest_ResponseHeaders{
			ResponseHeaders: &ext_proc.HttpHeaders{
				Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: "content-type", Value: "application/json"}}},
			},
		}

		response, err := router.handleResponseHeaders(respHeaders, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(response.GetResponseHeaders()).NotTo(BeNil())

		after := getHistogramSampleCount("llm_model_ttft_seconds", ctx.RequestModel)
		Expect(after).To(BeNumerically(">", before))
		Expect(ctx.TTFTRecorded).To(BeTrue())
		Expect(ctx.TTFTSeconds).To(BeNumerically(">", 0))
	})

	It("records TPOT on response body", func() {
		ctx := &RequestContext{
			RequestID:    "tpot-test-1",
			RequestModel: "model-a",
			StartTime:    time.Now().Add(-1 * time.Second),
		}

		beforeTPOT := getHistogramSampleCount("llm_model_tpot_seconds", ctx.RequestModel)

		beforePrompt := getHistogramSampleCount("llm_prompt_tokens_per_request", ctx.RequestModel)
		beforeCompletion := getHistogramSampleCount("llm_completion_tokens_per_request", ctx.RequestModel)

		openAIResponse := map[string]interface{}{
			"id":      "chatcmpl-xyz",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   ctx.RequestModel,
			"usage": map[string]interface{}{
				"prompt_tokens":     10,
				"completion_tokens": 5,
				"total_tokens":      15,
			},
			"choices": []map[string]interface{}{
				{
					"message":       map[string]interface{}{"role": "assistant", "content": "Hello"},
					"finish_reason": "stop",
				},
			},
		}
		respBodyJSON, err := json.Marshal(openAIResponse)
		Expect(err).NotTo(HaveOccurred())

		respBody := &ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: respBodyJSON},
		}

		response, err := router.handleResponseBody(respBody, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(response.GetResponseBody()).NotTo(BeNil())

		afterTPOT := getHistogramSampleCount("llm_model_tpot_seconds", ctx.RequestModel)
		Expect(afterTPOT).To(BeNumerically(">", beforeTPOT))

		// New per-request token histograms should also be recorded
		afterPrompt := getHistogramSampleCount("llm_prompt_tokens_per_request", ctx.RequestModel)
		afterCompletion := getHistogramSampleCount("llm_completion_tokens_per_request", ctx.RequestModel)
		Expect(afterPrompt).To(BeNumerically(">", beforePrompt))
		Expect(afterCompletion).To(BeNumerically(">", beforeCompletion))
	})
})
