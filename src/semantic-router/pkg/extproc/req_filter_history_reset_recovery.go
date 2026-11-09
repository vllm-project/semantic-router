package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// historyResetRecoveryWriter stores removed turns under the request's trusted
// scope. It is request-local: the scope and TTL are fixed when the action is
// prepared, so the policy cannot influence where content lands. Its store is
// the request's recovery store, which charges the shared byte budget.
type historyResetRecoveryWriter struct {
	store contextcompression.RecoveryStore
	scope string
	ttl   time.Duration
}

func (w *historyResetRecoveryWriter) Store(ctx context.Context, payload string) (string, error) {
	key, err := contextcompression.StoreRecoveryContent(ctx, w.store, w.scope, payload, w.ttl)
	if err != nil {
		return "", fmt.Errorf("store removed history: %w", err)
	}
	return key, nil
}

// historyResetRecovery prepares recoverable removal. It returns a terminal
// reason instead of a writer when the policy requires recovery that cannot be
// provided, so the action preserves history rather than removing content it
// could not store.
func (r *OpenAIRouter) historyResetRecovery(
	ctx *RequestContext,
	request *llmprotocol.Request,
) (historyreset.RecoveryWriter, map[int]llmprotocol.Message, string) {
	if !ctx.HistoryResetPolicy.RequiresRecovery() {
		return nil, nil, ""
	}
	// A removal that must stay recoverable cannot be served by a stream: the
	// retrieval follow-up has no place to run. Preserve or reject explicitly
	// instead of silently downgrading to irreversible removal.
	if ctx.ExpectStreamingResponse {
		return nil, nil, historyreset.ReasonStreamingUnsupported
	}
	if contextRecoveryToolConflict(ctx, request) {
		return nil, nil, historyreset.ReasonReservedToolConflict
	}
	scope := r.contextRecoveryScope(ctx)
	if ctx.ContextRecovery == nil || ctx.ContextRecoveryStore == nil || scope == "" {
		return nil, nil, historyreset.ReasonRecoveryUnavailable
	}
	writer := &historyResetRecoveryWriter{
		store: ctx.ContextRecoveryStore,
		scope: scope,
		ttl:   ctx.ContextRecovery.TTL,
	}
	return writer, detachedRequestMessages(request), ""
}

// detachedRequestMessages copies the complete neutral messages before any
// transformation runs, keyed by the stable ID the shared layer will assign.
// Reset is the first history step, so those IDs are the current positions.
// The copies keep tool calls, results, and media that the policy's text-only
// view cannot express.
func detachedRequestMessages(request *llmprotocol.Request) map[int]llmprotocol.Message {
	if request == nil {
		return nil
	}
	detached := make(map[int]llmprotocol.Message, len(request.Messages))
	for index, message := range request.Messages {
		copied := message
		copied.Content = append([]llmprotocol.Content(nil), message.Content...)
		detached[index] = copied
	}
	return detached
}

// historyResetBindingSchema identifies the binding representation so it can
// evolve without silently comparing two different encodings.
const historyResetBindingSchema = "vsr.history-reset.binding.v1"

// historyResetEvidenceBinding identifies the resolved original history and the
// live turn this request presents. Evidence must carry the same binding to be
// accepted, which is what makes stale or replayed evidence detectable; a
// timestamp alone could not.
//
// The digest covers the complete canonical semantic history, not just role and
// text: two requests that differ only in a tool call's arguments, a tool-result
// identifier, or a content kind must not share a binding. The snapshot's own
// deterministic encoding supplies that canonical form, and the schema label
// plus a length prefix keep distinct inputs from colliding.
func historyResetEvidenceBinding(ctx *RequestContext) string {
	if ctx == nil {
		return ""
	}
	history, ok := originalConversation(ctx)
	if !ok {
		return ""
	}
	canonical, err := json.Marshal(history)
	if err != nil {
		// Without a canonical form no evidence can be bound to this request,
		// so report no binding and let the action reject unbound evidence.
		return ""
	}
	var length [8]byte
	binary.BigEndian.PutUint64(length[:], uint64(len(canonical)))
	sum := sha256.New()
	sum.Write([]byte(historyResetBindingSchema))
	sum.Write([]byte{0})
	sum.Write(length[:])
	sum.Write(canonical)
	return hex.EncodeToString(sum.Sum(nil)[:16])
}

// finalizeHistoryResetRecovery publishes the issued key once the executor has
// committed the removal, so a rejected plan never leaves a retrieval tool
// advertising content that is still present in the request.
func finalizeHistoryResetRecovery(
	ctx *RequestContext,
	request *llmprotocol.Request,
) error {
	if ctx == nil || ctx.HistoryResetAction == nil {
		return nil
	}
	key := ctx.HistoryResetAction.RecoveryKey()
	if key == "" || ctx.HistoryResetDiagnostics == nil ||
		ctx.HistoryResetDiagnostics.Outcome != historyreset.OutcomeApplied {
		return nil
	}
	return registerContextRecoveryKeys(ctx, request, key)
}
