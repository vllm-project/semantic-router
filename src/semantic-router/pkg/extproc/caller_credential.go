package extproc

import (
	"bytes"
	"errors"
	"net/http"
	"slices"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// The caller credential is the provider login a client sent the Router, as
// Claude Code sends its Claude login, which a backend whose auth is
// credential: caller takes in place of a key. It is kept on the request
// context only, never on the principal, so an in-process hop or a call the
// Router makes on its own behalf has none to present, and it reaches the
// backend in the header the client sent it in: an OAuth bearer stays an
// Authorization header, a key stays x-api-key.

// callerCredentialHeaders are the headers a client sends its provider login
// in, the first found taken.
var callerCredentialHeaders = []string{headers.Authorization, headers.XAPIKey}

// callerCredentialFormats are the client APIs whose login each caller
// provider issued. A login reaches only the provider of the API the client
// spoke, so a bearer that came in on Responses never reaches Anthropic.
var callerCredentialFormats = map[string][]llmprotocol.WireFormat{
	"anthropic": {llmprotocol.AnthropicMessagesV1},
}

const (
	callerCredentialMessage = "Authentication failed. The selected upstream takes the client's own login for its provider, which the request does not carry."
	callerCredentialCode    = "caller_credential_missing" // #nosec G101 -- error code, not a credential
)

// errCallerCredentialMissing refuses a call to a caller backend that has no
// client login to present: a request that sent none, or a hop.
var errCallerCredentialMissing = errors.New("the upstream takes the client's own provider login, which the call does not carry")

// callerCredential is the client's login, the header it came in, and the
// client's anthropic-beta values, which a fallback attempt builds its headers
// without. Its fields are unexported so no serialization of the context
// carries it.
type callerCredential struct {
	header string
	value  string
	beta   string
}

// captureCallerCredential keeps the client's login when some backend takes
// it. The header itself stays where it was: every provider-bound request
// drops the client's credentials, and only a caller backend gets this back.
func (r *OpenAIRouter) captureCallerCredential(ctx *RequestContext) {
	if ctx == nil || ctx.Hop != nil || r == nil || !r.Config.HasCallerBackends() {
		return
	}
	for _, name := range callerCredentialHeaders {
		if value := strings.TrimSpace(headerValueCI(ctx, name)); value != "" {
			ctx.callerCredential = callerCredential{header: name, value: value, beta: headerValueCI(ctx, "anthropic-beta")}
			return
		}
	}
}

// callerCredentialFor returns the client's login for a caller backend when
// the client spoke the API of the backend's provider, and whether it did.
func callerCredentialFor(ctx *RequestContext, profile *config.ProviderProfile) (callerCredential, bool) {
	if ctx == nil || ctx.callerCredential.value == "" || profile == nil {
		return callerCredential{}, false
	}
	formats := callerCredentialFormats[strings.ToLower(strings.TrimSpace(profile.Type))]
	if !slices.Contains(formats, ctx.SourceFormat) {
		return callerCredential{}, false
	}
	return ctx.callerCredential, true
}

// redactCallerCredential takes the client's login out of a body the Router
// passes on or keeps, in case an upstream echoed it: the token itself, with
// or without its scheme.
func redactCallerCredential(ctx *RequestContext, body []byte) []byte {
	if ctx == nil || ctx.callerCredential.value == "" || len(body) == 0 {
		return body
	}
	value := ctx.callerCredential.value
	if _, token, ok := strings.Cut(value, " "); ok && strings.TrimSpace(token) != "" {
		value = strings.TrimSpace(token)
	}
	return bytes.ReplaceAll(body, []byte(value), []byte("[redacted]"))
}

// appendCallerCredential presents the client's login to a caller backend,
// in the header the client sent it in, and nothing else beside it.
func (r *OpenAIRouter) appendCallerCredential(state *routeHeaderState, model, backendName string, ctx *RequestContext) *ext_proc.ProcessingResponse {
	if r.CredentialResolver != nil {
		state.removeHeaders = append(state.removeHeaders, r.CredentialResolver.HeadersToStrip()...)
	}
	credential, ok := callerCredentialFor(ctx, state.profile)
	if !ok {
		logging.ComponentWarnEvent("extproc", "caller_credential_missing", map[string]interface{}{
			"request_id": ctx.RequestID, "model": model, "backend": backendName, "client_api": ctx.SourceFormat,
		})
		return r.callerCredentialMissing(ctx)
	}
	for _, name := range config.CredentialCarrierHeaders() {
		if !strings.EqualFold(name, credential.header) {
			state.removeHeaders = append(state.removeHeaders, name)
		}
	}
	removeClientCredentials(state, credential.header, true)
	state.setHeaders = append(state.setHeaders, overwriteRequestHeader(credential.header, credential.value))
	return nil
}

// callerCredentialAuthorizer presents the client's login on a fallback
// attempt to a caller backend, as primary dispatch does.
func callerCredentialAuthorizer(ctx *RequestContext, profile *config.ProviderProfile) (func(*http.Request), error) {
	credential, ok := callerCredentialFor(ctx, profile)
	if !ok {
		return nil, errCallerCredentialMissing
	}
	return func(request *http.Request) {
		for _, name := range config.CredentialCarrierHeaders() {
			if !strings.EqualFold(name, credential.header) {
				request.Header.Del(name)
			}
		}
		request.Header.Set(credential.header, credential.value)
		if credential.beta != "" {
			request.Header.Set("anthropic-beta", credential.beta)
		}
	}, nil
}

// callerCredentialMissing answers the client 401 before a call to a caller
// backend that the request carries no login for.
func (r *OpenAIRouter) callerCredentialMissing(ctx *RequestContext) *ext_proc.ProcessingResponse {
	ctx.ImmediateProtocolError = llmprotocol.NewError(
		llmprotocol.ErrorAuthentication, callerCredentialCode, callerCredentialMessage, nil,
	)
	return r.createErrorResponse(401, callerCredentialMessage)
}
