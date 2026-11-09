package extproc

import (
	"bytes"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A reply the backend produced names the model the backend reports, whatever
// the client and backend formats and whether it streams. Three names differ
// here: the logical model (target-*), the one dispatch asks the backend for
// (provider-model) and the one the backend answers with (source-model).
func TestBackendReplyNamesTheModelTheBackendReports(t *testing.T) {
	forEachExtProcMatrixPair(t, func(t *testing.T, client, backend llmprotocol.WireFormat) {
		dispatched := func(stream bool) (*OpenAIRouter, *RequestContext) {
			router, model := routingTestRouterForFormat(backend)
			request := testNeutralRequest(model, "hello")
			request.Stream = stream
			ctx := routingTestContext(client, request)
			ctx.ExpectStreamingResponse = stream
			if _, err := router.prepareProviderDispatch(request, model, "", false, ctx); err != nil {
				t.Fatal(err)
			}
			if ctx.SemanticRequest.Model != "provider-model" || ctx.RequestModel != model {
				t.Fatalf("dispatch left request model %q and logical model %q", ctx.SemanticRequest.Model, ctx.RequestModel)
			}
			return router, ctx
		}
		router, ctx := dispatched(true)
		streamed := extProcStreamFixture(backend)
		if mutation := router.handleSemanticStreamingResponseBody(streamed, true, ctx).
			GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			streamed = mutation.GetBody()
		}
		assertNamesOnlyBackendModel(t, "streamed", streamed)
		if ctx.SemanticResponse == nil || ctx.SemanticResponse.Model != "source-model" {
			t.Errorf("reconstructed stream names %+v, want source-model", ctx.SemanticResponse)
		}
		router, ctx = dispatched(false)
		buffered := extProcResponseFixture(backend)
		if mutation := router.handleNonStreamingResponseBody(buffered, ctx, 0).
			GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
			buffered = mutation.GetBody()
		}
		assertNamesOnlyBackendModel(t, "buffered", buffered)
	})
}

func assertNamesOnlyBackendModel(t *testing.T, path string, body []byte) {
	t.Helper()
	if !bytes.Contains(body, []byte("source-model")) || bytes.Contains(body, []byte("provider-model")) || bytes.Contains(body, []byte("target-")) {
		t.Errorf("%s reply does not name only the backend's model:\n%s", path, body)
	}
}
