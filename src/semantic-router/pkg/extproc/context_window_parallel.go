package extproc

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount"
)

// errContextWindowExceeded cancels the signals of a request whose count
// rules out every model it can reach.
var errContextWindowExceeded = errors.New("the request exceeds the context window of every model it can reach")

// contextAdmission is the window check of a request whose bytes could exceed
// a window it can reach. Its counts run beside the signals, which never wait
// for them; a count that rules out every reachable model cancels the
// signals, and the request is refused as soon as they return. Otherwise the
// counts serve the window checks after the decision.
type contextAdmission struct {
	done       chan struct{}
	started    time.Time
	generation uint64
	removable  int
	models     []admissionModel
	// parts and reserve are the prompt the counts read, taken on the
	// request's goroutine: later stages may change the request in place.
	parts   []string
	reserve int

	// Written by the count before done closes; read after it.
	elapsed time.Duration
	refusal *contextLengthError
	err     error

	mu sync.Mutex
	// refused is set once the counts refuse the request, and cancel, while
	// the signals run, ends them.
	refused bool
	cancel  context.CancelCauseFunc

	// joined is set, on the request's goroutine, once the counts are kept.
	joined bool
}

// admissionModel is one model the request can reach, and its count.
type admissionModel struct {
	model     string
	window    int
	tokenizer modelTokenizer
	tokens    int
}

// startContextAdmission starts the window check of a request whose UTF-8
// bytes could exceed the widest window it can reach; a request whose bytes
// fit, or that carries media, is never counted. A recipe that may compress
// the request leaves the check to the window checks after the decision.
func (r *OpenAIRouter) startContextAdmission(originalModel string, ctx *RequestContext) {
	if ctx == nil || ctx.SemanticRequest == nil || ctx.contextAdmission != nil {
		return
	}
	models, shrink := r.reachableContextModels(originalModel, ctx)
	if len(models) == 0 || shrink.unbounded {
		return
	}
	widest := 0
	for _, model := range models {
		window := r.contextWindowSize(model)
		if window <= 0 {
			return
		}
		widest = max(widest, window)
	}
	if requestContextByteBound(ctx, nil) < widest || requestCarriesMedia(ctx.SemanticRequest) {
		return
	}
	admission := &contextAdmission{
		done:       make(chan struct{}),
		started:    time.Now(),
		generation: ctx.SemanticRequest.Generation,
		removable:  shrink.removableBytes(ctx.SemanticRequest),
	}
	admission.parts, admission.reserve = promptParts(ctx.SemanticRequest)
	for _, model := range models {
		admission.models = append(admission.models, admissionModel{
			model: model, window: r.contextWindowSize(model), tokenizer: r.requestTokenizer(ctx, model),
		})
	}
	ctx.contextAdmission = admission
	go admission.run(r, selectionRequestContext(ctx))
}

// run counts the request once per tokenizer and decides it. A request that
// ends stops the count.
func (a *contextAdmission) run(r *OpenAIRouter, ctx context.Context) {
	finished := false
	finish := func(err error) {
		if !finished {
			finished = true
			a.err, a.elapsed = err, time.Since(a.started)
			close(a.done)
		}
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			finish(fmt.Errorf("the context window count failed: %v", recovered))
		}
	}()
	counts := map[*tokencount.Encoding]int{}
	for i := range a.models {
		model := &a.models[i]
		tokens, counted := counts[model.tokenizer.encoding]
		if !counted {
			var err error
			if tokens, err = countPromptParts(ctx, model.tokenizer.encoding, a.parts, a.reserve); err != nil {
				finish(err)
				return
			}
			counts[model.tokenizer.encoding] = tokens
		}
		model.tokens = tokens
	}
	a.refusal = r.admissionRefusal(a.models, a.removable)
	finish(nil)
	if a.refusal != nil {
		a.cancelSignals()
	}
}

// admissionRefusal is the refusal for counts that exceed every model's band,
// even without what the request's decisions may drop; nil when one of them
// does not.
func (r *OpenAIRouter) admissionRefusal(models []admissionModel, removable int) *contextLengthError {
	var refusal *contextLengthError
	for _, model := range models {
		kind := model.tokenizer.kind
		if r.bandVerdict(model.window, model.tokens, kind) != contextExceeds ||
			r.bandVerdict(model.window, model.tokens-removable, kind) != contextExceeds {
			return nil
		}
		if refusal == nil || model.window > refusal.window {
			refusal = &contextLengthError{window: model.window, tokens: model.tokens, kind: kind, scope: "every model the request can reach"}
		}
	}
	return refusal
}

// signals is the context the request's signals run under: parent, ended as
// soon as the counts refuse the request, and the function that releases it
// once the signals return.
func (a *contextAdmission) signals(parent context.Context) (context.Context, func()) {
	if a == nil {
		return parent, func() {}
	}
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithCancelCause(parent)
	a.mu.Lock()
	a.cancel = cancel
	refused := a.refused
	a.mu.Unlock()
	if refused {
		cancel(errContextWindowExceeded)
	}
	return ctx, func() {
		a.mu.Lock()
		a.cancel = nil
		a.mu.Unlock()
		cancel(context.Canceled)
	}
}

func (a *contextAdmission) cancelSignals() {
	a.mu.Lock()
	a.refused = true
	cancel := a.cancel
	a.mu.Unlock()
	if cancel != nil {
		cancel(errContextWindowExceeded)
	}
}

// finished reports counts that are done, without waiting for them.
func (a *contextAdmission) finished() bool {
	select {
	case <-a.done:
		return true
	default:
		return false
	}
}

// joinContextAdmission waits for the request's counts, if it has any, and
// keeps them for the window checks after the decision. It runs on the
// request's goroutine; the wait is the only time the counts add to the
// request's latency.
func (r *OpenAIRouter) joinContextAdmission(ctx *RequestContext) {
	if ctx == nil || ctx.contextAdmission == nil || ctx.contextAdmission.joined {
		return
	}
	admission := ctx.contextAdmission
	waited := time.Now()
	<-admission.done
	admission.joined = true
	r.keepContextAdmission(ctx, admission, time.Since(waited))
}

// contextAdmissionRefusal joins the request's counts and returns the refusal
// when they rule out every model the request can reach.
func (r *OpenAIRouter) contextAdmissionRefusal(ctx *RequestContext) *contextLengthError {
	r.joinContextAdmission(ctx)
	if ctx == nil || ctx.contextAdmission == nil {
		return nil
	}
	return ctx.contextAdmission.refusal
}

func (r *OpenAIRouter) keepContextAdmission(ctx *RequestContext, admission *contextAdmission, wait time.Duration) {
	metrics.RecordContextWindowWait(wait.Seconds())
	fields := map[string]interface{}{
		"request_id": ctx.RequestID,
		"count_ms":   admission.elapsed.Milliseconds(),
		"wait_ms":    float64(wait.Microseconds()) / 1000,
		"refused":    admission.refusal != nil,
	}
	if admission.err != nil {
		fields["error"] = admission.err.Error()
		logging.ComponentWarnEvent("extproc", "context_window_counted", fields)
		return
	}
	logging.ComponentEvent("extproc", "context_window_counted", fields)
	if ctx.contextCounts == nil {
		ctx.contextCounts = make(map[contextCountKey]contextCount)
	}
	logged := map[string]bool{}
	for _, model := range admission.models {
		name := model.tokenizer.encoding.Name()
		key := contextCountKey{tokenizer: name, generation: admission.generation}
		ctx.contextCounts[key] = contextCount{tokens: model.tokens, ok: true}
		if !logged[name] {
			logged[name] = true
			logContextCount(ctx, key, ctx.contextCounts[key], nil, admission.elapsed)
		}
		if admission.refusal != nil {
			r.noteContextVerdict(ctx, nil, model.model, model.window, model.tokens, model.tokenizer.kind, contextExceeds)
		}
	}
}
