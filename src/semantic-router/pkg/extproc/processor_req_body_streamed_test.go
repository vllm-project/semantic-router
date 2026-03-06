package extproc

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ---------- helpers ----------

func makeTestRouter(autoModel string) *OpenAIRouter {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "fallback-model",
		},
		RouterOptions: config.RouterOptions{
			StreamedBodyMode: true,
		},
	}
	if autoModel != "" {
		cfg.AutoModelName = autoModel
	}
	return &OpenAIRouter{Config: cfg}
}

// makeTestBody builds an OpenAI-style JSON body with "model" at the start
// (matching real client SDK behavior, not Go's alphabetical map ordering).
func makeTestBody(model string, content string, stream bool) []byte {
	contentJSON, _ := json.Marshal(content)
	streamStr := "false"
	if stream {
		streamStr = "true"
	}
	return []byte(fmt.Sprintf(
		`{"model":%q,"stream":%s,"messages":[{"role":"user","content":%s}]}`,
		model, streamStr, string(contentJSON),
	))
}

func splitTestChunks(data []byte, chunkSize int) [][]byte {
	var chunks [][]byte
	for len(data) > 0 {
		end := chunkSize
		if end > len(data) {
			end = len(data)
		}
		chunks = append(chunks, data[:end])
		data = data[end:]
	}
	return chunks
}

// ---------- Dispatch: BUFFERED vs STREAMED detection ----------

func TestDispatch_BufferedMode_NoHandler(t *testing.T) {
	ctx := &RequestContext{}

	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body:        []byte(`{"model":"gpt-4"}`),
			EndOfStream: true,
		},
	}

	assert.Nil(t, ctx.StreamedBody)

	eos := v.RequestBody.GetEndOfStream()
	assert.True(t, eos, "single body msg should be EOS")
	assert.Nil(t, ctx.StreamedBody, "no handler for BUFFERED")
}

func TestDispatch_StreamedMode_CreatesHandler(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body:        []byte(`{"mod`),
			EndOfStream: false,
		},
	}

	eos := v.RequestBody.GetEndOfStream()
	assert.False(t, eos, "first streamed chunk should not be EOS")

	// Simulate what dispatch does: create handler for non-EOS first message
	ctx.StreamedBody = newStreamedBodyHandler(router, ctx)
	assert.NotNil(t, ctx.StreamedBody)
	ctx.StreamedBody.Release()
	ctx.StreamedBody = nil
}

// ---------- StreamedBodyHandler state machine ----------

func TestHandler_InitState_AccumulatesSmallChunks(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 100

	resp, err := h.HandleChunk(&ext_proc.HttpBody{
		Body:        []byte(`{"mod`),
		EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state, "should stay in init until chunkSize reached")
	assert.NotNil(t, resp)

	bm := resp.GetRequestBody().GetResponse().GetBodyMutation()
	require.NotNil(t, bm, "init state eats chunks with empty body mutation")
	assert.Empty(t, bm.GetBody())
}

func TestHandler_InitState_TransitionsOnChunkSize(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32

	body := makeTestBody("gpt-4", "Hello world", false)
	require.Greater(t, len(body), 32, "body must exceed chunkSize")

	// Feed a chunk larger than chunkSize — should transition out of init
	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body:        body[:64],
		EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assert.NotEqual(t, stateInit, h.state, "should have left init state")
}

func TestHandler_DetectsAutoModel(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32

	body := makeTestBody("auto", "test content", false)

	// Feed enough data to detect model
	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body:        body,
		EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assert.Equal(t, "auto", h.model)
	assert.True(t, h.isAuto)
	assert.Equal(t, stateAccumulate, h.state)
}

func TestHandler_DetectsSpecifiedModel(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32

	body := makeTestBody("gpt-4o", "describe this", false)

	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body:        body,
		EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assert.Equal(t, "gpt-4o", h.model)
	assert.False(t, h.isAuto)
	assert.Equal(t, statePassthrough, h.state)
}

func TestHandler_StreamFieldDetected(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32

	body := makeTestBody("gpt-4", "hi", true)

	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body:        body,
		EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assert.True(t, ctx.ExpectStreamingResponse)
}

func TestHandler_Passthrough_NoBodyMutation(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32
	h.state = statePassthrough
	h.model = "gpt-4"

	resp, err := h.HandleChunk(&ext_proc.HttpBody{
		Body:        []byte("more data"),
		EndOfStream: false,
	}, ctx)
	require.NoError(t, err)

	bm := resp.GetRequestBody().GetResponse().GetBodyMutation()
	assert.Nil(t, bm, "passthrough should not mutate body")
}

func TestHandler_Accumulate_EatsChunks(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32
	h.state = stateAccumulate
	h.preproc = &incrementalPreprocessor{}
	h.buf.Write(makeTestBody("auto", "initial content", false))

	resp, err := h.HandleChunk(&ext_proc.HttpBody{
		Body:        []byte(" more content appended"),
		EndOfStream: false,
	}, ctx)
	require.NoError(t, err)

	bm := resp.GetRequestBody().GetResponse().GetBodyMutation()
	require.NotNil(t, bm, "accumulate should eat chunks with empty body mutation")
	assert.Empty(t, bm.GetBody())
}

// ---------- Incremental preprocessing ----------

func TestHandler_IncrementalPreprocess_SplitsSentences(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32
	h.state = stateAccumulate
	h.preproc = &incrementalPreprocessor{}

	longContent := "This is the first sentence. This is the second sentence. This is the third sentence."
	body := makeTestBody("auto", longContent, false)
	h.buf.Write(body)

	h.runIncrementalPreprocess()

	assert.GreaterOrEqual(t, len(h.preproc.sentences), 2, "should split at least 2 sentences, got %d", len(h.preproc.sentences))
	assert.Len(t, h.preproc.tfVectors, len(h.preproc.sentences), "TF vectors should match sentence count")
}

func TestHandler_IncrementalPreprocess_IdempotentOnSameData(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32
	h.state = stateAccumulate
	h.preproc = &incrementalPreprocessor{}

	body := makeTestBody("auto", "One sentence. Two sentences.", false)
	h.buf.Write(body)

	h.runIncrementalPreprocess()
	count1 := len(h.preproc.sentences)

	h.runIncrementalPreprocess()
	count2 := len(h.preproc.sentences)

	assert.Equal(t, count1, count2, "calling preprocess again on same data should be idempotent")
}

// ---------- continueEmptyBody / continueNoMutation ----------

func TestContinueEmptyBody_StructureCorrect(t *testing.T) {
	resp := continueEmptyBody()
	require.NotNil(t, resp)

	br := resp.GetRequestBody()
	require.NotNil(t, br)
	assert.Equal(t, ext_proc.CommonResponse_CONTINUE, br.Response.Status)

	bm := br.Response.BodyMutation
	require.NotNil(t, bm, "body mutation should be set")
	assert.Empty(t, bm.GetBody(), "body should be empty")
}

func TestContinueNoMutation_StructureCorrect(t *testing.T) {
	resp := continueNoMutation()
	require.NotNil(t, resp)

	br := resp.GetRequestBody()
	require.NotNil(t, br)
	assert.Equal(t, ext_proc.CommonResponse_CONTINUE, br.Response.Status)
	assert.Nil(t, br.Response.BodyMutation, "should have no body mutation")
}

// ---------- Pool reuse ----------

func TestHandler_PoolReuse_ResetsFields(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h1 := newStreamedBodyHandler(router, ctx)
	h1.model = "dirty"
	h1.isAuto = true
	h1.state = stateAccumulate
	h1.buf.WriteString("leftover data")
	h1.preproc = &incrementalPreprocessor{sentences: []string{"s1"}}
	h1.Release()

	h2 := newStreamedBodyHandler(router, ctx)
	defer h2.Release()

	assert.Empty(t, h2.model, "pool reuse must reset model")
	assert.False(t, h2.isAuto, "pool reuse must reset isAuto")
	assert.Equal(t, stateInit, h2.state, "pool reuse must reset state")
	assert.Equal(t, 0, h2.buf.Len(), "pool reuse must reset buffer")
	assert.Nil(t, h2.preproc, "pool reuse must reset preproc")
}

// ---------- Multi-chunk flow (state machine only, no full pipeline) ----------

func TestHandler_MultiChunk_AutoModel_StateTransitions(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32

	body := makeTestBody("auto", strings.Repeat("word ", 200), false)
	chunks := splitTestChunks(body, 50)
	require.Greater(t, len(chunks), 2, "need multiple chunks")

	// Process all non-final chunks
	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{
			Body:        chunk,
			EndOfStream: false,
		}, ctx)
		require.NoError(t, err)
		require.NotNil(t, resp)
	}

	assert.Equal(t, stateAccumulate, h.state)
	assert.Equal(t, "auto", h.model)
	assert.True(t, h.isAuto)
	assert.NotNil(t, h.preproc)
}

func TestHandler_MultiChunk_SpecifiedModel_StateTransitions(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32

	body := makeTestBody("gpt-4", strings.Repeat("word ", 200), false)
	chunks := splitTestChunks(body, 50)
	require.Greater(t, len(chunks), 2, "need multiple chunks")

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{
			Body:        chunk,
			EndOfStream: false,
		}, ctx)
		require.NoError(t, err)
		require.NotNil(t, resp)
	}

	assert.Equal(t, statePassthrough, h.state)
	assert.Equal(t, "gpt-4", h.model)
	assert.False(t, h.isAuto)
}

func TestHandler_AccumulatesFullBody(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.chunkSize = 32

	body := makeTestBody("auto", "Hello world test", false)
	chunks := splitTestChunks(body, 20)

	for _, chunk := range chunks[:len(chunks)-1] {
		_, err := h.HandleChunk(&ext_proc.HttpBody{
			Body:        chunk,
			EndOfStream: false,
		}, ctx)
		require.NoError(t, err)
	}

	assert.Equal(t, len(body)-len(chunks[len(chunks)-1]), h.buf.Len(),
		"buffer should contain all non-final chunks")
}

// ---------- Benchmarks ----------

func BenchmarkHandler_ChunkProcessing_16K(b *testing.B) {
	router := makeTestRouter("auto")
	words := make([]string, 16000)
	for i := range words {
		words[i] = fmt.Sprintf("word%d", i%1000)
	}
	body := makeTestBody("auto", strings.Join(words, " "), false)
	chunks := splitTestChunks(body, 4096)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx := &RequestContext{Headers: make(map[string]string)}
		h := newStreamedBodyHandler(router, ctx)
		for _, chunk := range chunks[:len(chunks)-1] {
			h.HandleChunk(&ext_proc.HttpBody{ //nolint:errcheck
				Body:        chunk,
				EndOfStream: false,
			}, ctx)
		}
		h.Release()
	}
}
