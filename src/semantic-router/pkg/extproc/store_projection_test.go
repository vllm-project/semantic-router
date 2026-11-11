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

// A Chat Completions client that sets store reaches a Messages backend:
// store: false asks for nothing such a backend does not already do, and
// store: true fails closed until the operator accepts dropping it. A Chat
// backend receives store as the client sent it.
func TestStoreReachesBackendsWithoutRequestStorage(t *testing.T) {
	for _, tc := range []struct {
		name    string
		target  llmprotocol.WireFormat
		store   string
		allow   bool
		refused bool
		keeps   bool
		warns   bool
	}{
		{name: "false to Messages", target: llmprotocol.AnthropicMessagesV1, store: "false"},
		{name: "true to Messages", target: llmprotocol.AnthropicMessagesV1, store: "true", refused: true},
		{name: "true to Messages, loss accepted", target: llmprotocol.AnthropicMessagesV1, store: "true", allow: true, warns: true},
		{name: "true to Chat, loss accepted", target: llmprotocol.OpenAIChatV1, store: "true", allow: true, keeps: true},
		{name: "false to Chat", target: llmprotocol.OpenAIChatV1, store: "false", keeps: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, model := routingTestRouterForFormat(tc.target)
			if tc.allow {
				router.Config.AllowLossyTranslation = []string{llmprotocol.LossyFieldStore}
			}
			body := []byte(`{"model":"` + model + `","store":` + tc.store + `,"max_tokens":16,"messages":[{"role":"user","content":"Say OK"}]}`)
			request, envelope, _, err := protocolcodec.NewBuiltinEngine().DecodeRequest(llmprotocol.OpenAIChatV1, body)
			if err != nil {
				t.Fatal(err)
			}
			ctx := routingTestContext(llmprotocol.OpenAIChatV1, &request)
			ctx.ProtocolEnvelope = envelope
			_, err = router.prepareProviderDispatch(&request, model, "", false, ctx)
			if tc.refused {
				var protocolErr *llmprotocol.ProtocolError
				if !errors.As(err, &protocolErr) || protocolErr.Code != "unsupported_capability" || !strings.Contains(protocolErr.Message, "request_storage") {
					t.Fatalf("dispatch error = %v, want the request_storage refusal", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("dispatch error = %v", err)
			}
			encoded, err := router.encodeDispatchRequest(ctx)
			if err != nil {
				t.Fatalf("encode error = %v", err)
			}
			if kept := bytes.Contains(encoded, []byte(`"store":`+tc.store)); kept != tc.keeps {
				t.Fatalf("dispatched body keeps store = %t, want %t: %s", kept, tc.keeps, encoded)
			}
			builder := newResponseHeaderMutationBuilder()
			builder.addWarnings(ctx)
			warning := ""
			for _, header := range builder.setHeaders {
				if header.Header.Key == headers.VSRWarnings {
					warning = string(header.Header.RawValue)
				}
			}
			warned := strings.HasPrefix(warning, "protocol;action=dropped;") && strings.HasSuffix(warning, `;field="store"`)
			if warned != tc.warns {
				t.Fatalf("x-vsr-warnings = %q, want a store warning: %t", warning, tc.warns)
			}
		})
	}
}
