package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/historyreset"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// defaultHistoryResetRecoveryTTL matches the compression default so one store
// does not hold two different retention contracts.
const defaultHistoryResetRecoveryTTL = 15 * time.Minute

// historyResetRecoveryWriter stores removed turns under the request's trusted
// scope. It is request-local: the scope and TTL are fixed when the action is
// prepared, so the policy cannot influence where content lands.
type historyResetRecoveryWriter struct {
	store contextcompression.RecoveryStore
	scope string
	ttl   time.Duration
}

func (w *historyResetRecoveryWriter) Store(
	ctx context.Context,
	payload string,
) (string, error) {
	if w == nil || w.store == nil || w.scope == "" {
		return "", fmt.Errorf("context recovery store or trusted scope is unavailable")
	}
	key := historyResetRecoveryKey(w.scope, payload)
	now := time.Now().UTC()
	entry := contextcompression.RecoveryEntry{
		Key:       key,
		Scope:     w.scope,
		Content:   payload,
		CreatedAt: now,
		ExpiresAt: now.Add(w.ttl),
	}
	if err := w.store.Put(ctx, entry, w.ttl); err != nil {
		return "", fmt.Errorf("store removed history: %w", err)
	}
	return key, nil
}

func historyResetRecoveryKey(scope string, payload string) string {
	sum := sha256.Sum256([]byte(scope + "\x00" + payload))
	return hex.EncodeToString(sum[:12])
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
	settings, err := contextRecoverySettingsForRequest(ctx)
	if err != nil || settings == nil {
		return nil, nil, historyreset.ReasonRecoveryUnavailable
	}
	store := r.contextRecoveryStore(settings)
	scope := r.contextCompressionScope(ctx)
	if store == nil || scope == "" {
		return nil, nil, historyreset.ReasonRecoveryUnavailable
	}
	writer := &historyResetRecoveryWriter{
		store: store,
		scope: scope,
		ttl:   historyResetRecoveryTTL(settings),
	}
	return writer, detachedRequestMessages(request), ""
}

func historyResetRecoveryTTL(settings *config.ContextCompressionRecoveryConfig) time.Duration {
	if settings.TTLSeconds > 0 {
		return time.Duration(settings.TTLSeconds) * time.Second
	}
	return defaultHistoryResetRecoveryTTL
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
	if ctx == nil || ctx.OriginalContextHistory == nil {
		return ""
	}
	canonical, err := json.Marshal(ctx.OriginalContextHistory.Conversation())
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
