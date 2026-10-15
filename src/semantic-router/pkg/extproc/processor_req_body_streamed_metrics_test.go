package extproc

import (
	"strings"
	"testing"
	"time"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

var streamedBodyArrivalFamilies = []string{
	"llm_streamed_body_arrival_seconds",
	"llm_streamed_body_bytes",
	"llm_streamed_body_chunks",
}

const streamedBodyMetricsTestBody = `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"measure how long this streamed body takes to arrive"}]}`

// fakeStreamedBodyClock advances by step on every read so the arrival time is
// exactly (reads - 1) * step between the first chunk and end of stream.
func fakeStreamedBodyClock(t *testing.T, step time.Duration) {
	t.Helper()
	now := time.Unix(1_700_000_000, 0)
	previous := streamedBodyNow
	streamedBodyNow = func() time.Time {
		now = now.Add(step)
		return now
	}
	t.Cleanup(func() { streamedBodyNow = previous })
}

func streamedBodyArrivalSamples(t *testing.T, recipe string) map[string]float64 {
	t.Helper()
	samples := make(map[string]float64, len(streamedBodyArrivalFamilies))
	for _, name := range streamedBodyArrivalFamilies {
		samples[name] = quorumMetricSamples(t, name, map[string]string{"recipe": recipe})
	}
	return samples
}

func dispatchStreamedBodyChunks(t *testing.T, router *OpenAIRouter, ctx *RequestContext, chunks [][]byte) {
	t.Helper()
	for i, chunk := range chunks {
		_, err := router.handleRequestBodyDispatch(&ext_proc.ProcessingRequest_RequestBody{
			RequestBody: &ext_proc.HttpBody{Body: chunk, EndOfStream: i == len(chunks)-1},
		}, ctx)
		require.NoError(t, err)
	}
}

func TestStreamedBodyDispatchRecordsArrivalStats(t *testing.T) {
	for _, fullDuplex := range []bool{false, true} {
		name := "streamed"
		if fullDuplex {
			name = "full_duplex_streamed"
		}
		t.Run(name, func(t *testing.T) {
			fakeStreamedBodyClock(t, 10*time.Millisecond)
			body := []byte(streamedBodyMetricsTestBody)
			chunks := splitTestChunks(body, len(body)/3+1)
			require.Len(t, chunks, 3)

			before := streamedBodyArrivalSamples(t, string(config.DefaultRecipeName))
			ctx := &RequestContext{Headers: make(map[string]string), FullDuplexRequestBody: fullDuplex}
			dispatchStreamedBodyChunks(t, makeTestRouter("vllm-sr/auto"), ctx, chunks)

			stats := ctx.StreamedBodyStats
			assert.True(t, stats.Present)
			assert.True(t, stats.Observed)
			assert.Equal(t, len(body), stats.Bytes)
			assert.Equal(t, len(chunks), stats.Chunks)
			// One clock read at the first chunk, one at end of stream.
			assert.Equal(t, 10*time.Millisecond, stats.Arrival)
			assert.Equal(t, config.DefaultRecipeName, ctx.Routing.RecipeName())

			after := streamedBodyArrivalSamples(t, string(config.DefaultRecipeName))
			for _, family := range streamedBodyArrivalFamilies {
				assert.Equal(t, before[family]+1, after[family], family)
			}
		})
	}
}

func TestFullDuplexTrailersEndRecordArrivalStats(t *testing.T) {
	fakeStreamedBodyClock(t, 10*time.Millisecond)
	body := fullDuplexTestBody
	before := streamedBodyArrivalSamples(t, consts.UnknownLabel)
	ctx := &RequestContext{Headers: make(map[string]string)}

	runProcessRequests(t, fullDuplexRoutingRouter(), ctx, NewMockStream(nil),
		fullDuplexHeadersRequest(false), bodyRequest(body[:30], false), bodyRequest(body[30:], false), trailersRequest())

	stats := ctx.StreamedBodyStats
	assert.True(t, stats.Present)
	assert.True(t, stats.Observed)
	assert.Equal(t, len(body), stats.Bytes)
	assert.Equal(t, 2, stats.Chunks)
	// One clock read at the first chunk, one when the trailers end the body.
	assert.Equal(t, 10*time.Millisecond, stats.Arrival)
	after := streamedBodyArrivalSamples(t, consts.UnknownLabel)
	for _, family := range streamedBodyArrivalFamilies {
		assert.Equal(t, before[family]+1, after[family], family)
	}
}

func TestFullDuplexTrailersWithoutBodyRecordNoArrivalStats(t *testing.T) {
	before := streamedBodyArrivalSamples(t, consts.UnknownLabel)
	ctx := &RequestContext{Headers: make(map[string]string)}

	runProcessRequests(t, fullDuplexRoutingRouter(), ctx, NewMockStream(nil),
		fullDuplexHeadersRequest(false), trailersRequest())

	assert.Equal(t, StreamedBodyStats{}, ctx.StreamedBodyStats)
	assert.Equal(t, before, streamedBodyArrivalSamples(t, consts.UnknownLabel))
}

func TestBufferedBodyRecordsNoArrivalStats(t *testing.T) {
	body := []byte(streamedBodyMetricsTestBody)
	t.Run("router streamed_body disabled", func(t *testing.T) {
		router := makeTestRouter("vllm-sr/auto")
		router.Config.StreamedBodyMode = false

		before := streamedBodyArrivalSamples(t, string(config.DefaultRecipeName))
		ctx := &RequestContext{Headers: make(map[string]string)}
		dispatchStreamedBodyChunks(t, router, ctx, [][]byte{body})

		assert.Equal(t, StreamedBodyStats{}, ctx.StreamedBodyStats)
		assert.Equal(t, before, streamedBodyArrivalSamples(t, string(config.DefaultRecipeName)))
	})
	// streamed_body stays enabled while the data plane negotiates a buffered
	// mode, which still reaches the streamed handler as one EOS message.
	for _, mode := range []http_ext.ProcessingMode_BodySendMode{
		http_ext.ProcessingMode_BUFFERED,
		http_ext.ProcessingMode_BUFFERED_PARTIAL,
	} {
		t.Run("data plane "+mode.String(), func(t *testing.T) {
			before := streamedBodyArrivalSamples(t, string(config.DefaultRecipeName))
			ctx := &RequestContext{Headers: make(map[string]string)}
			req := &ext_proc.ProcessingRequest{
				ProtocolConfig: &ext_proc.ProtocolConfiguration{RequestBodyMode: mode},
				Request: &ext_proc.ProcessingRequest_RequestBody{
					RequestBody: &ext_proc.HttpBody{Body: body, EndOfStream: true},
				},
			}
			require.NoError(t, makeTestRouter("vllm-sr/auto").handleProcessRequest(NewMockStream(nil), req, ctx))

			assert.True(t, ctx.BufferedRequestBody)
			assert.Equal(t, StreamedBodyStats{}, ctx.StreamedBodyStats)
			assert.Equal(t, before, streamedBodyArrivalSamples(t, string(config.DefaultRecipeName)))
		})
	}
}

func TestStreamedBodyPoolReuseResetsArrivalStats(t *testing.T) {
	first := newStreamedBodyHandler(makeTestRouter("vllm-sr/auto"), &RequestContext{})
	_, err := first.HandleChunk(&ext_proc.HttpBody{Body: []byte("partial")}, first.ctx)
	require.NoError(t, err)
	require.Equal(t, 1, first.chunkCount)
	require.False(t, first.firstChunkAt.IsZero())
	first.Release()

	second := newStreamedBodyHandler(makeTestRouter("vllm-sr/auto"), &RequestContext{})
	defer second.Release()
	assert.Zero(t, second.chunkCount)
	assert.True(t, second.firstChunkAt.IsZero())
}

func TestPromptCompressionRecordsDurationAndOutcome(t *testing.T) {
	longText := strings.Repeat("The router reads this sentence while it measures prompt compression. ", 60)
	tests := []struct {
		name        string
		compression config.PromptCompressionConfig
		text        string
		outcome     string
	}{
		{
			name:        "compressed",
			compression: config.PromptCompressionConfig{Enabled: true, MaxTokens: 64},
			text:        longText,
			outcome:     metrics.PromptCompressionCompressed,
		},
		{
			name:        "under min_length",
			compression: config.PromptCompressionConfig{Enabled: true, MaxTokens: 64, MinLength: 10_000},
			text:        longText,
			outcome:     metrics.PromptCompressionSkippedMinLength,
		},
		{
			name:        "within max_tokens",
			compression: config.PromptCompressionConfig{Enabled: true, MaxTokens: 10_000},
			text:        longText,
			outcome:     metrics.PromptCompressionSkippedMaxTokens,
		},
		{
			name:        "disabled",
			compression: config.PromptCompressionConfig{Enabled: false, MaxTokens: 64},
			text:        longText,
			outcome:     metrics.PromptCompressionSkippedDisabled,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			const recipe = "prompt-compression-metrics-test"
			router := &OpenAIRouter{Config: &config.RouterConfig{}}
			router.Config.PromptCompression = test.compression
			ctx := &RequestContext{}
			ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: recipe})
			ctx.TraceContext, ctx.RequestSpan = tracing.StartSpan(t.Context(), tracing.SpanRequest)
			defer ctx.RequestSpan.End()

			outcomeLabels := map[string]string{"recipe": recipe, "outcome": test.outcome}
			stageLabels := map[string]string{"recipe": recipe, "stage": metrics.RoutingStagePromptCompression}
			outcomesBefore := quorumMetricSamples(t, "llm_prompt_compression_total", outcomeLabels)
			durationsBefore := quorumMetricSamples(t, "llm_routing_stage_duration_seconds", stageLabels)

			compressed, _, run := router.compressSignalEvaluationText(test.text)
			observePromptCompression(ctx, run.outcome, run.elapsed)

			assert.Equal(t, test.outcome, run.outcome)
			assert.Equal(t, outcomesBefore+1, quorumMetricSamples(t, "llm_prompt_compression_total", outcomeLabels))
			wantDurations := durationsBefore
			if test.outcome == metrics.PromptCompressionCompressed {
				wantDurations++
				assert.Less(t, len(compressed), len(test.text))
			} else {
				assert.Equal(t, test.text, compressed)
			}
			assert.Equal(t, wantDurations, quorumMetricSamples(t, "llm_routing_stage_duration_seconds", stageLabels))
		})
	}
}

func TestDecisionEvaluationRecordsPromptCompressionOutcome(t *testing.T) {
	const recipe = "prompt-compression-decision-test"
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	router.Config.PromptCompression = config.PromptCompressionConfig{Enabled: true, MaxTokens: 10_000}
	router.Config.Decisions = []config.Decision{{Name: "only"}}
	ctx := &RequestContext{}
	ctx.Routing.SelectRecipe(&config.RoutingRecipe{Name: recipe})
	ctx.TraceContext, ctx.RequestSpan = tracing.StartSpan(t.Context(), tracing.SpanRequest)
	defer ctx.RequestSpan.End()

	labels := map[string]string{"recipe": recipe, "outcome": metrics.PromptCompressionSkippedMaxTokens}
	before := quorumMetricSamples(t, "llm_prompt_compression_total", labels)
	// No classifier is configured, so evaluation stops after compression.
	_, _, _, _, err := router.performDecisionEvaluation("vllm-sr/auto", signalConversationHistory{currentUserMessage: "short routing text"}, ctx)
	require.Error(t, err)
	assert.Equal(t, before+1, quorumMetricSamples(t, "llm_prompt_compression_total", labels))
}
