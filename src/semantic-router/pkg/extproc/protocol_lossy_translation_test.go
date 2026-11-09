package extproc

import (
	"bytes"
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// A coding agent reports a command that exited nonzero as a failed tool
// result. Without the opt-in the dispatch to a Chat backend fails closed, and
// with it the backend gets the command output and the caller gets the warning.
func TestAllowLossyTranslationDispatchesFailedToolResultWithWarning(t *testing.T) {
	for _, allow := range []bool{false, true} {
		router, model := routingTestRouterForFormat(llmprotocol.OpenAIChatV1)
		if allow {
			router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldToolResultError}
		}
		body := []byte(`{"model":"` + model + `","max_tokens":64,"messages":[
			{"role":"user","content":"List /nope"},
			{"role":"assistant","content":[{"type":"tool_use","id":"call_1","name":"Bash","input":{"command":"ls /nope"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_1","is_error":true,"content":"Exit code 2\nls: cannot access '/nope': No such file or directory"}]}]}`)
		request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.AnthropicMessagesV1, body)
		if err != nil {
			t.Fatal(err)
		}
		ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, &request)
		ctx.ProtocolEnvelope = envelope
		if _, err = router.prepareProviderDispatch(&request, model, "", false, ctx); err != nil {
			t.Fatal(err)
		}
		encoded, err := router.encodeDispatchRequest(ctx)
		if !allow {
			var protocolErr *llmprotocol.ProtocolError
			if !errors.As(err, &protocolErr) || protocolErr.Code != "lossy_translation" {
				t.Fatalf("default dispatch error = %v, want lossy_translation", err)
			}
			continue
		}
		if err != nil || !bytes.Contains(encoded, []byte("No such file or directory")) || bytes.Contains(encoded, []byte("is_error")) {
			t.Fatalf("opted-in dispatch: body=%s error=%v", encoded, err)
		}
		builder := newResponseHeaderMutationBuilder()
		builder.addProtocolDiagnostics(ctx, ctx.ProtocolDiagnostics)
		warning := ""
		for _, header := range builder.setHeaders {
			if header.Header.Key == headers.VSRProtocolWarnings {
				warning = string(header.Header.RawValue)
			}
		}
		if !strings.HasPrefix(warning, "approximated;") || !strings.HasSuffix(warning, ";tool_result.is_error") {
			t.Fatalf("opted-in loss is not visible to the caller: %q", warning)
		}
	}
}
