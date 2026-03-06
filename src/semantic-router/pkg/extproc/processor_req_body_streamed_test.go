package extproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

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

// makeTestBodyModelLast builds a body where "model" appears after "messages",
// simulating Go's alphabetical json.Marshal ordering.
func makeTestBodyModelLast(model string, content string) []byte {
	contentJSON, _ := json.Marshal(content)
	return []byte(fmt.Sprintf(
		`{"messages":[{"role":"user","content":%s}],"model":%q,"stream":false}`,
		string(contentJSON), model,
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

// assertChunkEaten verifies a non-EOS response has CONTINUE + empty body mutation.
func assertChunkEaten(t *testing.T, resp *ext_proc.ProcessingResponse, msg string) {
	t.Helper()
	require.NotNil(t, resp, msg)
	br := resp.GetRequestBody()
	require.NotNil(t, br, msg)
	assert.Equal(t, ext_proc.CommonResponse_CONTINUE, br.GetResponse().GetStatus(), msg)
	bm := br.GetResponse().GetBodyMutation()
	require.NotNil(t, bm, msg+": body mutation must be set")
	assert.Empty(t, bm.GetBody(), msg+": body must be empty")
}

// =====================================================================
// Dispatch: BUFFERED vs STREAMED detection
// =====================================================================

func TestDispatch_BufferedMode_NoHandler(t *testing.T) {
	ctx := &RequestContext{}
	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{
			Body:        []byte(`{"model":"gpt-4"}`),
			EndOfStream: true,
		},
	}
	assert.Nil(t, ctx.StreamedBody)
	assert.True(t, v.RequestBody.GetEndOfStream())
}

func TestDispatch_StreamedMode_CreatesHandler(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	ctx.StreamedBody = newStreamedBodyHandler(router, ctx)
	assert.NotNil(t, ctx.StreamedBody)
	ctx.StreamedBody.Release()
	ctx.StreamedBody = nil
}

// =====================================================================
// State machine: init → passthrough / accumulate transitions
// =====================================================================

func TestHandler_InitState_AccumulatesSmallChunks(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	resp, err := h.HandleChunk(&ext_proc.HttpBody{
		Body:        []byte(`{"mod`),
		EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state)
	assertChunkEaten(t, resp, "init partial chunk")
}

func TestHandler_InitState_TransitionsOnModelDetection(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", "Hello world", false)
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.NotEqual(t, stateInit, h.state)
}

func TestHandler_DetectsAutoModel(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: makeTestBody("auto", "test content", false), EndOfStream: false,
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

	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: makeTestBody("gpt-4o", "describe this", false), EndOfStream: false,
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

	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: makeTestBody("gpt-4", "hi", true), EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assert.True(t, ctx.ExpectStreamingResponse)
}

// =====================================================================
// Both paths eat every chunk (the core correctness invariant)
// =====================================================================

func TestHandler_Passthrough_EatsChunks(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.state = statePassthrough
	h.model = "gpt-4"

	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: []byte("more data"), EndOfStream: false}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "passthrough non-EOS")
}

func TestHandler_Accumulate_EatsChunks(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	h.state = stateAccumulate
	h.buf.Write(makeTestBody("auto", "initial content", false))

	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: []byte(" more"), EndOfStream: false}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "accumulate non-EOS")
}

// =====================================================================
// Model detection split across chunks (the bug that comment #3 found)
// =====================================================================

func TestHandler_ModelSplitAcrossChunks_AutoModel(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", "some content here", false)
	// Split right in the middle of "model":"auto" so first chunk can't detect it
	splitAt := 5 // `{"mod` — not enough for gjson to extract "model"

	resp1, err := h.HandleChunk(&ext_proc.HttpBody{Body: body[:splitAt], EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state, "model not yet visible — should stay in init")
	assertChunkEaten(t, resp1, "init chunk before model visible")

	resp2, err := h.HandleChunk(&ext_proc.HttpBody{Body: body[splitAt:], EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateAccumulate, h.state, "model now visible — should transition to accumulate")
	assertChunkEaten(t, resp2, "transition chunk must also be eaten")
}

func TestHandler_ModelSplitAcrossChunks_SpecifiedModel(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", "some content", false)
	splitAt := 5

	resp1, err := h.HandleChunk(&ext_proc.HttpBody{Body: body[:splitAt], EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state)
	assertChunkEaten(t, resp1, "init chunk")

	resp2, err := h.HandleChunk(&ext_proc.HttpBody{Body: body[splitAt:], EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, statePassthrough, h.state)
	assertChunkEaten(t, resp2, "passthrough chunk after transition")
}

func TestHandler_ModelFieldAtEndOfJSON(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBodyModelLast("gpt-4o-mini", "test data")
	// Send first chunk with only the messages part (model at end)
	splitAt := bytes.Index(body, []byte(`"model"`))
	require.Positive(t, splitAt, "model field must exist in body")

	resp1, err := h.HandleChunk(&ext_proc.HttpBody{Body: body[:splitAt], EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state, "model not in first chunk")
	assertChunkEaten(t, resp1, "chunk without model")

	resp2, err := h.HandleChunk(&ext_proc.HttpBody{Body: body[splitAt:], EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, statePassthrough, h.state)
	assert.Equal(t, "gpt-4o-mini", h.model)
	assertChunkEaten(t, resp2, "chunk with model")
}

// =====================================================================
// Single-chunk EOS (entire body arrives in one message)
// =====================================================================

// Single-chunk EOS triggers the full handleRequestBody pipeline which requires
// a fully initialized router (caching, classifiers, etc.). We test the state
// transition by sending a non-EOS chunk first, then verifying state + buffer,
// which is the critical invariant. The EOS→handleRequestBody path is exercised
// by integration tests with a full router.

func TestHandler_SingleChunk_AutoModel_StateAndBuffer(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", "single chunk content", false)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "auto single chunk non-EOS")
	assert.Equal(t, stateAccumulate, h.state)
	assert.Equal(t, body, h.buf.Bytes())
	assert.Equal(t, "auto", ctx.RequestModel)
}

func TestHandler_SingleChunk_SpecifiedModel_StateAndBuffer(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", "single chunk content", false)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "specified single chunk non-EOS")
	assert.Equal(t, statePassthrough, h.state)
	assert.Equal(t, body, h.buf.Bytes())
	assert.Equal(t, "gpt-4", ctx.RequestModel)
}

// =====================================================================
// Empty and zero-length chunks
// =====================================================================

func TestHandler_EmptyChunk_StaysInInit(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: []byte{}, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state)
	assertChunkEaten(t, resp, "empty chunk")
	assert.Equal(t, 0, h.buf.Len())
}

func TestHandler_EmptyChunks_ThenNonEOS_NoModel(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// Multiple empty non-EOS chunks — should stay in init with empty model
	for i := 0; i < 5; i++ {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: []byte{}, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assert.Equal(t, stateInit, h.state, "empty chunk %d: should stay in init", i)
		assertChunkEaten(t, resp, fmt.Sprintf("empty chunk %d", i))
	}
	assert.Empty(t, h.model)
	assert.Equal(t, 0, h.buf.Len())
}

func TestHandler_EmptyIntermediateChunks_PassthroughPath(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", "content", false)
	// First chunk has the full body (triggers passthrough transition)
	resp1, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, statePassthrough, h.state)
	assertChunkEaten(t, resp1, "first chunk")

	// Send several empty intermediate chunks
	for i := 0; i < 3; i++ {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: []byte{}, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, fmt.Sprintf("empty chunk %d", i))
	}

	assert.Equal(t, len(body), h.buf.Len(), "empty chunks should not change buffer size")
}

// =====================================================================
// Tiny chunk sizes (1-byte streaming)
// =====================================================================

func TestHandler_ByteByByte_AutoModel(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", "hello", false)
	chunks := splitTestChunks(body, 1)
	require.Greater(t, len(chunks), 10, "should have many 1-byte chunks")

	var transitioned bool
	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "1-byte chunk")
		if h.state == stateAccumulate {
			transitioned = true
		}
	}
	assert.True(t, transitioned, "should eventually detect auto and move to accumulate")
	assert.Equal(t, len(body)-1, h.buf.Len())
}

func TestHandler_ByteByByte_SpecifiedModel(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", "hello", false)
	chunks := splitTestChunks(body, 1)

	var transitioned bool
	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "1-byte chunk")
		if h.state == statePassthrough {
			transitioned = true
		}
	}
	assert.True(t, transitioned, "should detect specified model and move to passthrough")
}

// =====================================================================
// Large payloads (16K, 64K, 256K)
// =====================================================================

func TestHandler_LargePayload_16K_Accumulate(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", strings.Repeat("A long sentence for testing. ", 2000), false)
	chunks := splitTestChunks(body, 4096)
	require.Greater(t, len(chunks), 3, "need 4+ chunks at 4K each for 16K payload")

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "16K chunk")
	}
	assert.Equal(t, stateAccumulate, h.state)
	expected := 0
	for _, c := range chunks[:len(chunks)-1] {
		expected += len(c)
	}
	assert.Equal(t, expected, h.buf.Len())
}

func TestHandler_LargePayload_64K_Passthrough(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", strings.Repeat("X", 64*1024), false)
	chunks := splitTestChunks(body, 8192)

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "64K passthrough chunk")
	}
	assert.Equal(t, statePassthrough, h.state)
}

func TestHandler_LargePayload_256K_BufferIntegrity(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	content := strings.Repeat("word ", 50000) // ~250KB
	body := makeTestBody("auto", content, false)
	chunks := splitTestChunks(body, 4096)

	for _, chunk := range chunks[:len(chunks)-1] {
		_, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
	}

	var expectedLen int
	for _, c := range chunks[:len(chunks)-1] {
		expectedLen += len(c)
	}
	assert.Equal(t, expectedLen, h.buf.Len(), "buffer must contain all non-EOS chunk data")
}

// =====================================================================
// Varying chunk sizes within a single request
// =====================================================================

func TestHandler_IrregularChunkSizes(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", strings.Repeat("data ", 500), false)
	sizes := []int{3, 1, 50, 7, 200, 1024, 13, 500}
	var chunks [][]byte
	remaining := body
	for _, sz := range sizes {
		if len(remaining) == 0 {
			break
		}
		if sz > len(remaining) {
			sz = len(remaining)
		}
		chunks = append(chunks, remaining[:sz])
		remaining = remaining[sz:]
	}
	if len(remaining) > 0 {
		chunks = append(chunks, remaining)
	}

	totalSent := 0
	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "irregular chunk")
		totalSent += len(chunk)
	}
	assert.Equal(t, totalSent, h.buf.Len())
	assert.Equal(t, stateAccumulate, h.state)
}

// =====================================================================
// Unicode / multibyte content
// =====================================================================

func TestHandler_UnicodeContent(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", "你好世界! 🌍 Ünïcödé tëxt with ñ and ß", false)
	chunks := splitTestChunks(body, 10) // split mid-codepoint

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "unicode chunk")
	}

	var expected int
	for _, c := range chunks[:len(chunks)-1] {
		expected += len(c)
	}
	assert.Equal(t, expected, h.buf.Len())
	assert.Contains(t, h.buf.String(), `你好世界`)
}

// =====================================================================
// Malformed / edge-case JSON
// =====================================================================

func TestHandler_MalformedJSON_NoModelField(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// Valid JSON but no "model" field — stays in init until EOS
	body := []byte(`{"messages":[{"role":"user","content":"no model"}],"stream":false}`)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state, "no model field → stay in init")
	assertChunkEaten(t, resp, "no model field chunk")
}

func TestHandler_MalformedJSON_InvalidJSON(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := []byte(`{this is not valid json at all!!!}`)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state, "invalid JSON → no model → stay in init")
	assertChunkEaten(t, resp, "invalid JSON chunk")
}

func TestHandler_MalformedJSON_NonEOS_StaysInit(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := []byte(`not json at all`)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state, "invalid JSON without model → stay in init")
	assert.Empty(t, h.model)
	assertChunkEaten(t, resp, "malformed JSON chunk")
}

func TestHandler_EmptyModelString(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := []byte(`{"model":"","messages":[{"role":"user","content":"test"}]}`)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	// gjson returns "" for model:"" — model is empty, stays in init
	// (extractModelFast returns "" for empty string values)
	assertChunkEaten(t, resp, "empty model string")
}

// =====================================================================
// Malformed / edge-case chunks (Envoy protocol robustness)
// =====================================================================

func TestHandler_NilBodyChunk(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// nil Body — GetBody() returns nil, buf.Write(nil) is a no-op
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: nil, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state)
	assert.Equal(t, 0, h.buf.Len(), "nil write must not grow buffer")
	assertChunkEaten(t, resp, "nil body chunk")
}

func TestHandler_NilBodyThenRealChunk(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// Several nil chunks followed by a real chunk
	for i := 0; i < 3; i++ {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: nil, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, fmt.Sprintf("nil chunk %d", i))
	}

	body := makeTestBody("gpt-4", "hello", false)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "real chunk after nils")
	assert.Equal(t, statePassthrough, h.state)
	assert.Equal(t, "gpt-4", h.model)
}

func TestHandler_BinaryContent_StaysInInit(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// Pure binary — no JSON structure, no model key
	binary := []byte{0x00, 0x01, 0xFF, 0xFE, 0x89, 0x50, 0x4E, 0x47}
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: binary, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state, "binary data has no model key")
	assertChunkEaten(t, resp, "binary chunk")
	assert.Equal(t, len(binary), h.buf.Len())
}

func TestHandler_BinaryThenJSON_Recovers(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// Binary garbage first — stays in init
	binary := []byte{0x00, 0xFF, 0xFE}
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: binary, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state)

	// Then valid JSON with model — gjson scans the accumulated buffer and
	// may or may not find "model" depending on binary interference. The key
	// invariant: no crash, no panic, handler stays functional.
	json := makeTestBody("gpt-4", "test", false)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: json, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "json after binary")
	// gjson is resilient — it scans for "model": in byte stream
	// With leading binary bytes the key may or may not be found, but no crash
}

func TestHandler_PartialJSON_ModelSplitMidQuote(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// Split right in the middle of the model value string: "gpt|-4"
	part1 := []byte(`{"model":"gpt`)
	part2 := []byte(`-4","stream":false,"messages":[]}`)

	resp1, err := h.HandleChunk(&ext_proc.HttpBody{Body: part1, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateInit, h.state, "unterminated string — gjson can't extract model")
	assertChunkEaten(t, resp1, "partial model value")

	resp2, err := h.HandleChunk(&ext_proc.HttpBody{Body: part2, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, statePassthrough, h.state, "model now complete")
	assert.Equal(t, "gpt-4", h.model)
	assertChunkEaten(t, resp2, "completed model value")
}

func TestHandler_DuplicateModelKeys(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// gjson returns first match; json.Unmarshal uses last.
	// With "auto" first, handler enters accumulate; the full pipeline
	// (extractContentFast) will see "auto" too since it also uses gjson.
	body := []byte(`{"model":"auto","model":"gpt-4","messages":[]}`)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "duplicate model keys")
	assert.Equal(t, "auto", h.model, "gjson returns first match")
	assert.Equal(t, stateAccumulate, h.state)
}

func TestHandler_ChunkAfterRelease_DispatchCreatesNew(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h := newStreamedBodyHandler(router, ctx)
	ctx.StreamedBody = h

	body := makeTestBody("gpt-4", "test", false)
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)

	// Simulate EOS release (as handleRequestBodyDispatch does)
	h.Release()
	ctx.StreamedBody = nil

	// After release, ctx.StreamedBody is nil. A hypothetical extra body
	// message would go through buffered path (ctx.StreamedBody == nil).
	assert.Nil(t, ctx.StreamedBody, "handler must be nil after release")
}

// =====================================================================
// Multimodal content (images in OpenAI content parts)
// =====================================================================

// makeMultimodalBody builds a realistic OpenAI multimodal request with both
// text and an inline base64 image of the given size.
func makeMultimodalBody(model string, text string, imageBytes int) []byte {
	// Build a fake base64 string of the requested size
	b64 := strings.Repeat("ABCDEFGH", imageBytes/8+1)
	b64 = b64[:imageBytes]
	imageURL := "data:image/png;base64," + b64

	contentJSON, _ := json.Marshal(text)
	imageURLJSON, _ := json.Marshal(imageURL)
	return []byte(fmt.Sprintf(
		`{"model":%q,"stream":false,"messages":[{"role":"user","content":[{"type":"text","text":%s},{"type":"image_url","image_url":{"url":%s}}]}]}`,
		model, string(contentJSON), string(imageURLJSON),
	))
}

func TestHandler_Multimodal_SmallImage_ChunkedDelivery(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeMultimodalBody("gpt-4o", "What is in this image?", 200)
	chunks := splitTestChunks(body, 50)
	require.Greater(t, len(chunks), 3, "need multiple chunks")

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "multimodal chunk")
	}

	assert.Equal(t, statePassthrough, h.state)
	assert.Equal(t, "gpt-4o", h.model)

	var totalNonEOS int
	for _, c := range chunks[:len(chunks)-1] {
		totalNonEOS += len(c)
	}
	assert.Equal(t, totalNonEOS, h.buf.Len())
}

func TestHandler_Multimodal_LargeImage_100KB(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// 100KB base64 image — realistic for a high-res photo
	body := makeMultimodalBody("auto", "Describe this photo", 100_000)
	chunks := splitTestChunks(body, 4096)
	require.Greater(t, len(chunks), 20, "100KB image should produce 20+ chunks at 4K each")

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "large image chunk")
	}

	assert.Equal(t, stateAccumulate, h.state, "auto model → accumulate")
	assert.Equal(t, "auto", h.model)
}

func TestHandler_Multimodal_LargeImage_500KB(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeMultimodalBody("gpt-4o", "What are the objects?", 500_000)
	chunks := splitTestChunks(body, 16384)
	require.Greater(t, len(chunks), 5)

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "500KB image chunk")
	}

	assert.Equal(t, statePassthrough, h.state)

	var totalNonEOS int
	for _, c := range chunks[:len(chunks)-1] {
		totalNonEOS += len(c)
	}
	assert.Equal(t, totalNonEOS, h.buf.Len(), "buffer must contain all image data")
}

func TestHandler_Multimodal_ImageSplitAcrossChunks_Base64Boundary(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeMultimodalBody("gpt-4o", "Analyze", 10_000)

	// Split at 1-byte chunks through the base64 data region
	// This tests that mid-base64 splits don't confuse the handler
	chunks := splitTestChunks(body, 7) // prime number to hit awkward boundaries

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "base64 boundary chunk")
	}

	assert.Equal(t, statePassthrough, h.state)
	assert.Equal(t, "gpt-4o", h.model)
}

func TestHandler_Multimodal_MultipleImages(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := []byte(`{"model":"gpt-4o","stream":false,"messages":[{"role":"user","content":[` +
		`{"type":"text","text":"Compare these images"},` +
		`{"type":"image_url","image_url":{"url":"data:image/png;base64,` + strings.Repeat("A", 1000) + `"}},` +
		`{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,` + strings.Repeat("B", 2000) + `"}}` +
		`]}]}`)

	chunks := splitTestChunks(body, 200)

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "multi-image chunk")
	}

	assert.Equal(t, statePassthrough, h.state)
	assert.Equal(t, "gpt-4o", h.model)
}

func TestHandler_Multimodal_HTTPImageURL_NotDataURI(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// HTTP URL images (not inline base64) — still a valid OpenAI request
	body := []byte(`{"model":"gpt-4o","stream":false,"messages":[{"role":"user","content":[` +
		`{"type":"text","text":"What is this?"},` +
		`{"type":"image_url","image_url":{"url":"https://example.com/photo.png"}}` +
		`]}]}`)

	chunks := splitTestChunks(body, 30)

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "http image url chunk")
	}

	assert.Equal(t, statePassthrough, h.state)
	assert.Equal(t, "gpt-4o", h.model)
}

func TestHandler_Multimodal_ModelAfterImage_AlphabeticalJSON(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// Go's json.Marshal alphabetizes keys: "messages" before "model".
	// With a large image, most chunks arrive before model is visible.
	imageData := strings.Repeat("X", 50_000)
	body := []byte(fmt.Sprintf(
		`{"messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]}],"model":"gpt-4o","stream":false}`,
		imageData,
	))

	chunks := splitTestChunks(body, 4096)

	// Send ALL chunks as non-EOS (model is in the last few bytes)
	initCount := 0
	for _, chunk := range chunks {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "alphabetical json chunk")
		if h.state == stateInit {
			initCount++
		}
	}

	assert.Greater(t, initCount, 5,
		"many chunks should arrive before model is visible in alphabetical JSON")
	assert.Equal(t, statePassthrough, h.state, "should eventually find model")
	assert.Equal(t, "gpt-4o", h.model)
	assert.Equal(t, len(body), h.buf.Len(), "buffer must contain entire body")
}

func TestHandler_Multimodal_AutoModel_LargeImage(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeMultimodalBody("auto", "Classify this image", 200_000)
	chunks := splitTestChunks(body, 8192)

	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "auto+image chunk")
	}

	assert.Equal(t, stateAccumulate, h.state)
	assert.Equal(t, "auto", h.model)
	assert.True(t, h.isAuto)
}

func TestHandler_Multimodal_MaxBytes_RejectsLargeImage(t *testing.T) {
	// 50KB limit but image is 100KB
	router := makeTestRouterWithLimits("auto", 50_000, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeMultimodalBody("gpt-4o", "Describe", 100_000)
	chunks := splitTestChunks(body, 4096)

	var hitLimit bool
	for _, chunk := range chunks {
		_, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		if err != nil {
			assert.Contains(t, err.Error(), "too large")
			hitLimit = true
			break
		}
	}
	assert.True(t, hitLimit, "should reject before fully accumulating 100KB image")
}

// =====================================================================
// Context field propagation
// =====================================================================

func TestHandler_SetsRequestModel_OnDetection(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("claude-3-opus", "test", false)
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, "claude-3-opus", ctx.RequestModel)
}

func TestHandler_SetsExpectStreamingResponse(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", "hi", true)
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.True(t, ctx.ExpectStreamingResponse)
	assert.True(t, h.isStream)
}

func TestHandler_NoStreamParam_DefaultsFalse(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", "hi", false)
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.False(t, ctx.ExpectStreamingResponse)
	assert.False(t, h.isStream)
}

// =====================================================================
// Pool reuse safety
// =====================================================================

func TestHandler_PoolReuse_ResetsFields(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}

	h1 := newStreamedBodyHandler(router, ctx)
	h1.model = "dirty"
	h1.isAuto = true
	h1.isStream = true
	h1.state = stateAccumulate
	h1.buf.WriteString("leftover data")
	h1.Release()

	h2 := newStreamedBodyHandler(router, ctx)
	defer h2.Release()

	assert.Empty(t, h2.model)
	assert.False(t, h2.isAuto)
	assert.False(t, h2.isStream)
	assert.Equal(t, stateInit, h2.state)
	assert.Equal(t, 0, h2.buf.Len())
}

func TestHandler_PoolReuse_MultipleRoundTrips(t *testing.T) {
	router := makeTestRouter("auto")

	for round := 0; round < 10; round++ {
		ctx := &RequestContext{Headers: make(map[string]string)}
		h := newStreamedBodyHandler(router, ctx)

		assert.Equal(t, stateInit, h.state, "round %d: must start in init", round)
		assert.Empty(t, h.model, "round %d: model must be empty", round)
		assert.Equal(t, 0, h.buf.Len(), "round %d: buffer must be empty", round)

		model := "gpt-4"
		if round%2 == 0 {
			model = "auto"
		}
		body := makeTestBody(model, fmt.Sprintf("round %d content", round), false)
		_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assert.Equal(t, model, h.model)
		h.Release()
	}
}

// =====================================================================
// Buffer clone safety (the aliasing fix)
// =====================================================================

func TestHandler_BufferClone_NoAliasing_Accumulate(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)

	body := makeTestBody("auto", "test content for clone safety", false)
	h.state = stateAccumulate
	h.buf.Write(body)

	h.ctx.ProcessingStartTime = time.Now()
	h.ctx.OriginalRequestBody = bytes.Clone(h.buf.Bytes())
	snapshot := make([]byte, len(h.ctx.OriginalRequestBody))
	copy(snapshot, h.ctx.OriginalRequestBody)

	h.Release()

	assert.Equal(t, snapshot, ctx.OriginalRequestBody,
		"OriginalRequestBody must survive handler release")
}

func TestHandler_BufferClone_NoAliasing_Passthrough(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)

	body := makeTestBody("gpt-4", "passthrough clone test", false)
	h.state = statePassthrough
	h.buf.Write(body)

	h.ctx.ProcessingStartTime = time.Now()
	h.ctx.OriginalRequestBody = bytes.Clone(h.buf.Bytes())
	snapshot := make([]byte, len(h.ctx.OriginalRequestBody))
	copy(snapshot, h.ctx.OriginalRequestBody)

	h.Release()

	assert.Equal(t, snapshot, ctx.OriginalRequestBody,
		"OriginalRequestBody must survive handler release")
}

func TestHandler_BufferClone_SurvivesPoolReuse(t *testing.T) {
	router := makeTestRouter("auto")
	ctx1 := &RequestContext{Headers: make(map[string]string)}
	h1 := newStreamedBodyHandler(router, ctx1)

	body1 := makeTestBody("auto", "first request data", false)
	h1.buf.Write(body1)
	ctx1.OriginalRequestBody = bytes.Clone(h1.buf.Bytes())
	h1.Release()

	// Reuse the pool — h2 may get the same underlying handler
	ctx2 := &RequestContext{Headers: make(map[string]string)}
	h2 := newStreamedBodyHandler(router, ctx2)
	h2.buf.WriteString("OVERWRITTEN DATA THAT SHOULD NOT LEAK")
	h2.Release()

	// ctx1's body must still be intact
	assert.Equal(t, body1, ctx1.OriginalRequestBody,
		"first request's body must not be corrupted by pool reuse")
}

// =====================================================================
// Multi-chunk complete flows
// =====================================================================

func TestHandler_MultiChunk_AutoModel_AllChunksEaten(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", strings.Repeat("word ", 200), false)
	chunks := splitTestChunks(body, 50)
	require.Greater(t, len(chunks), 2)

	for i, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, fmt.Sprintf("auto chunk %d", i))
	}
	assert.Equal(t, stateAccumulate, h.state)
}

func TestHandler_MultiChunk_SpecifiedModel_AllChunksEaten(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", strings.Repeat("word ", 200), false)
	chunks := splitTestChunks(body, 50)
	require.Greater(t, len(chunks), 2)

	for i, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, fmt.Sprintf("specified chunk %d", i))
	}
	assert.Equal(t, statePassthrough, h.state)
}

func TestHandler_AccumulatesFullBody(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", "Hello world test", false)
	chunks := splitTestChunks(body, 20)

	for _, chunk := range chunks[:len(chunks)-1] {
		_, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
	}

	expectedLen := len(body) - len(chunks[len(chunks)-1])
	assert.Equal(t, expectedLen, h.buf.Len())
}

// =====================================================================
// Full flow with feedChunks helper (init → transition → all eaten)
// =====================================================================

func TestHandler_FullFlow_TinyChunks_AutoModel(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", "Complete flow test with tiny chunks", false)
	chunks := splitTestChunks(body, 3)

	// feedChunks sends all chunks including EOS on the last one.
	// EOS triggers handleRequestBody which may fail without full router,
	// so we only feed non-EOS chunks here.
	for _, chunk := range chunks[:len(chunks)-1] {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: chunk, EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, "tiny chunk")
	}

	var totalBytes int
	for _, c := range chunks[:len(chunks)-1] {
		totalBytes += len(c)
	}
	assert.Equal(t, totalBytes, h.buf.Len())
}

// =====================================================================
// sharedContinueEmptyBody (pooled singleton)
// =====================================================================

func TestSharedContinueEmptyBody_StructureCorrect(t *testing.T) {
	resp := sharedContinueEmptyBody
	require.NotNil(t, resp)

	br := resp.GetRequestBody()
	require.NotNil(t, br)
	assert.Equal(t, ext_proc.CommonResponse_CONTINUE, br.Response.Status)

	bm := br.Response.BodyMutation
	require.NotNil(t, bm)
	assert.Empty(t, bm.GetBody())
}

func TestSharedContinueEmptyBody_IsSingleton(t *testing.T) {
	r1 := sharedContinueEmptyBody
	r2 := sharedContinueEmptyBody
	assert.Same(t, r1, r2, "shared response must be the same pointer")
}

func TestSharedContinueEmptyBody_ReturnedByHandler(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: []byte(`{"mod`), EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Same(t, sharedContinueEmptyBody, resp,
		"non-EOS chunk should return the pooled singleton, not a fresh allocation")
}

// =====================================================================
// Table-driven: chunk sizes × model types × payload sizes
// =====================================================================

func TestHandler_Matrix_ChunkSizes_Models_PayloadSizes(t *testing.T) {
	chunkSizes := []int{1, 7, 50, 512, 4096}
	models := []struct {
		name      string
		wantState streamedBodyState
	}{
		{"auto", stateAccumulate},
		{"gpt-4", statePassthrough},
		{"claude-3-sonnet", statePassthrough},
	}
	payloadSizes := []int{50, 500, 5000, 50000}

	router := makeTestRouter("auto")

	for _, cs := range chunkSizes {
		for _, m := range models {
			for _, ps := range payloadSizes {
				name := fmt.Sprintf("chunk=%d/model=%s/payload=%d", cs, m.name, ps)
				t.Run(name, func(t *testing.T) {
					ctx := &RequestContext{Headers: make(map[string]string)}
					h := newStreamedBodyHandler(router, ctx)
					defer h.Release()

					content := strings.Repeat("x", ps)
					body := makeTestBody(m.name, content, false)
					chunks := splitTestChunks(body, cs)

					// When chunk size >= body size, there's only 1 chunk.
					// Send it as non-EOS to test the state transition without
					// triggering handleRequestBody.
					if len(chunks) == 1 {
						resp, err := h.HandleChunk(&ext_proc.HttpBody{
							Body: chunks[0], EndOfStream: false,
						}, ctx)
						require.NoError(t, err)
						assertChunkEaten(t, resp, "single chunk must be eaten")
						assert.Equal(t, m.wantState, h.state)
						assert.Equal(t, m.name, h.model)
						assert.Equal(t, len(body), h.buf.Len())
						return
					}

					var totalNonEOS int
					for _, chunk := range chunks[:len(chunks)-1] {
						resp, err := h.HandleChunk(&ext_proc.HttpBody{
							Body: chunk, EndOfStream: false,
						}, ctx)
						require.NoError(t, err)
						assertChunkEaten(t, resp, "chunk must be eaten")
						totalNonEOS += len(chunk)
					}

					assert.Equal(t, m.wantState, h.state,
						"expected state %v", m.wantState)
					assert.Equal(t, m.name, h.model)
					assert.Equal(t, totalNonEOS, h.buf.Len(),
						"buffer must match total non-EOS bytes")
				})
			}
		}
	}
}

// =====================================================================
// Multiple init chunks before model detection
// =====================================================================

func TestHandler_MultipleInitChunks_BeforeModelDetection(t *testing.T) {
	router := makeTestRouter("auto")
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// Simulate many tiny init chunks that gradually reveal the JSON
	body := makeTestBodyModelLast("gpt-4", strings.Repeat("data ", 100))
	modelOffset := bytes.Index(body, []byte(`"model"`))
	require.Greater(t, modelOffset, 20, "model should be deep in the JSON")

	// Send 1-byte chunks up to just before "model" — all should stay in init
	for i := 0; i < modelOffset; i++ {
		resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body[i : i+1], EndOfStream: false}, ctx)
		require.NoError(t, err)
		assertChunkEaten(t, resp, fmt.Sprintf("init byte %d", i))
		assert.Equal(t, stateInit, h.state, "byte %d: should still be in init", i)
	}

	// Now send the rest — should detect model and transition
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: body[modelOffset:], EndOfStream: false}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "model detection chunk")
	assert.Equal(t, statePassthrough, h.state)
	assert.Equal(t, "gpt-4", h.model)
	assert.Equal(t, len(body), h.buf.Len())
}

// =====================================================================
// Benchmarks
// =====================================================================

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

func BenchmarkHandler_PoolAllocation(b *testing.B) {
	router := makeTestRouter("auto")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx := &RequestContext{Headers: make(map[string]string)}
		h := newStreamedBodyHandler(router, ctx)
		h.buf.WriteString("some data to exercise buffer")
		h.Release()
	}
}

func BenchmarkHandler_ByteByByte_1K(b *testing.B) {
	router := makeTestRouter("auto")
	body := makeTestBody("auto", strings.Repeat("word ", 200), false)
	chunks := splitTestChunks(body, 1)

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

// =====================================================================
// Guard: max body size
// =====================================================================

func makeTestRouterWithLimits(autoModel string, maxBytes int64, timeoutSec int) *OpenAIRouter {
	cfg := &config.RouterConfig{
		BackendModels: config.BackendModels{
			DefaultModel: "fallback-model",
		},
		RouterOptions: config.RouterOptions{
			StreamedBodyMode:       true,
			MaxStreamedBodyBytes:   maxBytes,
			StreamedBodyTimeoutSec: timeoutSec,
		},
	}
	if autoModel != "" {
		cfg.AutoModelName = autoModel
	}
	return &OpenAIRouter{Config: cfg}
}

func TestGuard_MaxBytes_RejectsOversizedBody(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 100, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// First chunk: 60 bytes — under limit
	resp, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: bytes.Repeat([]byte("A"), 60), EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "under limit")

	// Second chunk: 60 more bytes — total 120 > limit 100
	_, err = h.HandleChunk(&ext_proc.HttpBody{
		Body: bytes.Repeat([]byte("B"), 60), EndOfStream: false,
	}, ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too large")
}

func TestGuard_MaxBytes_ExactLimitAllowed(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 100, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	resp, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: bytes.Repeat([]byte("X"), 100), EndOfStream: false,
	}, ctx)
	require.NoError(t, err, "exactly at limit should be allowed")
	assertChunkEaten(t, resp, "at limit")
}

func TestGuard_MaxBytes_OneByteOverRejects(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 100, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: bytes.Repeat([]byte("X"), 101), EndOfStream: false,
	}, ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too large")
}

func TestGuard_MaxBytes_ZeroMeansUnlimited(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 0, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	bigChunk := bytes.Repeat([]byte("X"), 1_000_000)
	resp, err := h.HandleChunk(&ext_proc.HttpBody{Body: bigChunk, EndOfStream: false}, ctx)
	require.NoError(t, err, "no limit → should accept any size")
	assertChunkEaten(t, resp, "unlimited")
}

func TestGuard_MaxBytes_ErrorOnInitState(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 50, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: bytes.Repeat([]byte("Z"), 60), EndOfStream: false,
	}, ctx)
	require.Error(t, err, "should reject even in init state")
	assert.Equal(t, stateInit, h.state, "state should not have advanced past init")
}

func TestGuard_MaxBytes_ErrorOnPassthroughState(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 200, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("gpt-4", "hello", false)
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, statePassthrough, h.state)

	// Push over limit
	_, err = h.HandleChunk(&ext_proc.HttpBody{
		Body: bytes.Repeat([]byte("X"), 200), EndOfStream: false,
	}, ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too large")
}

func TestGuard_MaxBytes_ErrorOnAccumulateState(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 200, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", "hello", false)
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)
	assert.Equal(t, stateAccumulate, h.state)

	_, err = h.HandleChunk(&ext_proc.HttpBody{
		Body: bytes.Repeat([]byte("X"), 200), EndOfStream: false,
	}, ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too large")
}

func TestGuard_MaxBytes_IncrementalGrowthToLimit(t *testing.T) {
	limit := int64(500)
	router := makeTestRouterWithLimits("auto", limit, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", "test", false)
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.NoError(t, err)

	chunkSize := 50
	for int64(h.buf.Len()+chunkSize) <= limit {
		_, err = h.HandleChunk(&ext_proc.HttpBody{
			Body: bytes.Repeat([]byte("A"), chunkSize), EndOfStream: false,
		}, ctx)
		require.NoError(t, err, "at %d bytes, still under limit %d", h.buf.Len(), limit)
	}

	// One more push over
	_, err = h.HandleChunk(&ext_proc.HttpBody{
		Body: bytes.Repeat([]byte("B"), chunkSize), EndOfStream: false,
	}, ctx)
	require.Error(t, err)
}

// =====================================================================
// Guard: accumulation timeout
// =====================================================================

func TestGuard_Timeout_RejectsStaleAccumulation(t *testing.T) {
	// Use 1-second timeout
	router := makeTestRouterWithLimits("auto", 0, 1)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	// First chunk is fine
	resp, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: []byte(`{"mod`), EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "before timeout")

	// Simulate time passing by moving the deadline into the past
	h.deadline = time.Now().Add(-1 * time.Second)

	_, err = h.HandleChunk(&ext_proc.HttpBody{
		Body: []byte(`el":`), EndOfStream: false,
	}, ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "timed out")
}

func TestGuard_Timeout_ZeroMeansNoDeadline(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 0, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	assert.True(t, h.deadline.IsZero(), "zero timeout should produce zero deadline")

	// Should not error even after writing data
	resp, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: []byte(`{"mod`), EndOfStream: false,
	}, ctx)
	require.NoError(t, err)
	assertChunkEaten(t, resp, "no deadline")
}

func TestGuard_Timeout_DeadlineSetCorrectly(t *testing.T) {
	before := time.Now()
	router := makeTestRouterWithLimits("auto", 0, 30)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()
	after := time.Now()

	assert.False(t, h.deadline.IsZero())
	assert.True(t, h.deadline.After(before.Add(29*time.Second)),
		"deadline should be ~30s from now")
	assert.True(t, h.deadline.Before(after.Add(31*time.Second)),
		"deadline should be ~30s from now")
}

func TestGuard_Timeout_InAllStates(t *testing.T) {
	states := []struct {
		name  string
		state streamedBodyState
	}{
		{"init", stateInit},
		{"passthrough", statePassthrough},
		{"accumulate", stateAccumulate},
	}

	for _, s := range states {
		t.Run(s.name, func(t *testing.T) {
			router := makeTestRouterWithLimits("auto", 0, 1)
			ctx := &RequestContext{Headers: make(map[string]string)}
			h := newStreamedBodyHandler(router, ctx)
			defer h.Release()
			h.state = s.state
			h.deadline = time.Now().Add(-1 * time.Second) // expired

			_, err := h.HandleChunk(&ext_proc.HttpBody{
				Body: []byte("data"), EndOfStream: false,
			}, ctx)
			require.Error(t, err, "state %s should be guarded by timeout", s.name)
			assert.Contains(t, err.Error(), "timed out")
		})
	}
}

// =====================================================================
// Guard: combined max bytes + timeout
// =====================================================================

func TestGuard_Combined_MaxBytesTriggersFirst(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 50, 300)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: bytes.Repeat([]byte("X"), 60), EndOfStream: false,
	}, ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "too large",
		"max bytes should trigger before timeout")
}

func TestGuard_Combined_TimeoutTriggersFirst(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 1_000_000, 1)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	h.deadline = time.Now().Add(-1 * time.Second) // expired

	_, err := h.HandleChunk(&ext_proc.HttpBody{
		Body: []byte("tiny"), EndOfStream: false,
	}, ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "timed out",
		"timeout should trigger before max bytes")
}

// =====================================================================
// Guard: errors stop processing (no state advance after error)
// =====================================================================

func TestGuard_ErrorPreventsStateAdvance(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 10, 0)
	ctx := &RequestContext{Headers: make(map[string]string)}
	h := newStreamedBodyHandler(router, ctx)
	defer h.Release()

	body := makeTestBody("auto", "too much content that exceeds the tiny limit", false)
	_, err := h.HandleChunk(&ext_proc.HttpBody{Body: body, EndOfStream: false}, ctx)
	require.Error(t, err)
	assert.Equal(t, stateInit, h.state,
		"on guard error, state machine should not advance")
}

// =====================================================================
// Pool reuse clears guard fields
// =====================================================================

func TestGuard_PoolReuse_ClearsGuardFields(t *testing.T) {
	router := makeTestRouterWithLimits("auto", 500, 60)
	ctx := &RequestContext{Headers: make(map[string]string)}

	h1 := newStreamedBodyHandler(router, ctx)
	assert.Equal(t, int64(500), h1.maxBytes)
	assert.False(t, h1.deadline.IsZero())
	h1.Release()

	// Get a new handler from pool with no-limit router
	routerNoLimit := makeTestRouter("auto")
	h2 := newStreamedBodyHandler(routerNoLimit, ctx)
	defer h2.Release()

	assert.Equal(t, int64(0), h2.maxBytes,
		"pool-recycled handler must reset maxBytes from new config")
	assert.True(t, h2.deadline.IsZero(),
		"pool-recycled handler must reset deadline from new config")
}
