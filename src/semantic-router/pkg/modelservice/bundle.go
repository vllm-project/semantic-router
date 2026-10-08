package modelservice

import (
	"context"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/admission"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DefaultBundleWindow caps how long a parked classify, embeddings or rerank
// call waits for the other participants of its bundle. Decisions calls never
// wait on a timer.
const DefaultBundleWindow = 2 * time.Millisecond

type bundleKey struct{}

type participantKey struct{}

// Bundle coalesces the runtime calls of one request stage into one
// /v1/bundle call per runtime process, or several when the stage has more
// calls than the process takes in one bundle. Classify calls that differ only
// in their inputs share one task, and the questions a stage asks one served
// model share one decisions task (see fuse and fuseDecisions).
//
// The goroutines of the stage Join the bundle. A runtime call made with the
// bundle's context parks in it, and each caller gets its own result.
//
// Decisions calls are sent per deployment, and never on a timer. A
// participant that declares the deployments it asks (JoinAsking) is an asker
// of each of them; one that joins without declaring (Join) may ask any. A
// deployment's questions are sent once every asker of it has asked (is parked
// on a decisions call, to it or to another deployment, or waits for
// admission), is fanning out, or has left, and every undeclared participant
// is blocked on a call or has left. So a stage asks a deployment its
// questions in one call whatever the timing of its participants, and that
// call never waits for a declared participant that does not ask it. A
// participant asks each deployment once, before any other runtime call. Once
// its questions are sent, a caller waits for them until the latest deadline
// of the exchange that carries them: the stage waits for that exchange anyway.
//
// Other calls are sent when no participant can still add one (every one is
// blocked on a call or on its own fan-out, or has left), or once the window
// since the first of them parked has passed. Their callers stop waiting at
// their own deadline, and the task carries the latest of them.
type Bundle struct {
	base    context.Context
	window  time.Duration
	mu      sync.Mutex
	active  int
	blocked int
	// open counts the participants that declared no deployment, and
	// openBlocked those of them that are blocked.
	open        int
	openBlocked int
	// askers counts per deployment the participants that may ask it, not
	// counting one while it fans out; asked counts those that have asked.
	askers  map[string]int
	asked   map[string]int
	pending map[*Client][]*bundleCall
	first   time.Time
	timer   *time.Timer
	// questions holds the parked decisions calls by deployment, and since
	// the first of them has waited.
	questions map[string][]*bundleCall
	since     map[string]time.Time
	// decisionCalls counts the decisions tasks sent per deployment.
	decisionCalls map[string]int
	seq           int
	flushes       int
}

type bundleCall struct {
	ctx    context.Context
	client *Client
	task   api.BundleTask
	done   chan struct{}
	result api.BundleResult
	timing exchangeTiming
	err    error
	// decision is set for a decisions call, and sent is closed once the task
	// that carries it is sent, in the exchange whose context is carrier.
	decision *decisionCall
	sent     chan struct{}
	carrier  context.Context
}

// decisionCall is a decisions call parked in a bundle: the request it asks,
// the served model's result cache (nil when caching is off), the deployment
// its cache metrics count against, and, once answered, its own answers.
type decisionCall struct {
	request    Request
	cache      *resultCache
	deployment string
	response   Response
}

// participant is how one participant's calls count in its bundle: as an
// asker of the deployments it declared.
type participant struct {
	bundle      *Bundle
	deployments []string
}

// WithBundle returns a context whose runtime calls are bundled, and the bundle.
// A window of zero or less uses DefaultBundleWindow.
func WithBundle(ctx context.Context, window time.Duration) (context.Context, *Bundle) {
	if window <= 0 {
		window = DefaultBundleWindow
	}
	bundle := &Bundle{
		base: ctx, window: window, pending: make(map[*Client][]*bundleCall),
		askers: make(map[string]int), asked: make(map[string]int),
		questions: make(map[string][]*bundleCall), since: make(map[string]time.Time), decisionCalls: make(map[string]int),
	}
	ctx = context.WithValue(ctx, bundleKey{}, bundle)
	return admission.WithWaiter(ctx, func() func() { return bundle.waiting(nil) }), bundle
}

func bundleFrom(ctx context.Context) *Bundle {
	bundle, _ := ctx.Value(bundleKey{}).(*Bundle)
	return bundle
}

// InBundle reports whether runtime calls made with ctx park in a bundle.
func InBundle(ctx context.Context) bool {
	return bundleFrom(ctx) != nil
}

// participantOf returns the declared participant whose context ctx is, or
// nil for an undeclared one.
func (b *Bundle) participantOf(ctx context.Context) *participant {
	p, _ := ctx.Value(participantKey{}).(*participant)
	if p == nil || p.bundle != b {
		return nil
	}
	return p
}

// Join adds a participant that may ask any deployment; call the returned
// function when it has finished.
func (b *Bundle) Join() (leave func()) {
	b.mu.Lock()
	b.enterLocked(nil, 1)
	b.mu.Unlock()
	return b.leaver(nil)
}

// JoinAsking adds a participant that asks only the given deployments (none
// for one that asks no decision model) and returns the context its calls use
// and the function to call when it has finished.
func (b *Bundle) JoinAsking(ctx context.Context, deployments ...string) (context.Context, func()) {
	deployments = slices.Clone(deployments)
	slices.Sort(deployments)
	p := &participant{bundle: b, deployments: slices.Compact(deployments)}
	b.mu.Lock()
	b.enterLocked(p, 1)
	b.mu.Unlock()
	ctx = context.WithValue(ctx, participantKey{}, p)
	return admission.WithWaiter(ctx, func() func() { return b.waiting(p) }), b.leaver(p)
}

func (b *Bundle) leaver(p *participant) func() {
	var once sync.Once
	return func() {
		once.Do(func() {
			b.mu.Lock()
			b.enterLocked(p, -1)
			b.sendReadyLocked()
			b.mu.Unlock()
		})
	}
}

// enterLocked adds n participants like p (removes them for a negative n).
func (b *Bundle) enterLocked(p *participant, n int) {
	b.active += n
	if p == nil {
		b.open += n
		return
	}
	for _, deployment := range p.deployments {
		b.askers[deployment] += n
	}
}

// blockLocked marks a participant like p blocked (unblocked for a negative
// n); asked says whether a declared one has asked by it: it parked a
// decisions call or waits for admission.
func (b *Bundle) blockLocked(p *participant, asked bool, n int) {
	b.blocked += n
	if p == nil {
		b.openBlocked += n
		return
	}
	if asked {
		for _, deployment := range p.deployments {
			b.asked[deployment] += n
		}
	}
}

// fanLocked marks a participant like p fanning out (back from it for a
// negative n): it is blocked, and its work asks in its place.
func (b *Bundle) fanLocked(p *participant, n int) {
	b.blocked += n
	if p == nil {
		b.openBlocked += n
		return
	}
	for _, deployment := range p.deployments {
		b.askers[deployment] -= n
	}
}

// waiting counts a participant like p as blocked while it waits for an
// admission slot, which a call parked in the bundle may hold.
func (b *Bundle) waiting(p *participant) func() {
	b.mu.Lock()
	b.blockLocked(p, true, 1)
	b.sendReadyLocked()
	b.mu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			b.mu.Lock()
			b.blockLocked(p, true, -1)
			b.mu.Unlock()
		})
	}
}

// Fan runs work(i) for every i in [0, n) in its own goroutine, each a
// participant of the context's bundle like the caller, and waits for all of
// them. While it waits, the caller cannot add a call, so it counts as blocked,
// and as an asker its work asks in its place.
func Fan(ctx context.Context, n int, work func(i int)) {
	bundle := bundleFrom(ctx)
	if n <= 0 {
		return
	}
	var p *participant
	if bundle != nil {
		p = bundle.participantOf(ctx)
		bundle.mu.Lock()
		bundle.enterLocked(p, n)
		bundle.fanLocked(p, 1)
		bundle.mu.Unlock()
	}
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if bundle != nil {
				defer bundle.leaver(p)()
			}
			work(i)
		}(i)
	}
	wg.Wait()
	if bundle != nil {
		bundle.mu.Lock()
		bundle.fanLocked(p, -1)
		bundle.mu.Unlock()
	}
}

// Flushes reports how many times the bundle has sent calls (tests, metrics).
func (b *Bundle) Flushes() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.flushes
}

func (b *Bundle) submit(ctx context.Context, client *Client, task api.BundleTask) (api.BundleResult, exchangeTiming, error) {
	call := &bundleCall{ctx: ctx, task: task, done: make(chan struct{})}
	if err := b.park(client, call); err != nil {
		return api.BundleResult{}, exchangeTiming{}, err
	}
	return call.result, call.timing, call.err
}

// decide parks a decisions call and returns its own answers.
func (b *Bundle) decide(ctx context.Context, client *Client, decision *decisionCall, body api.DecisionRequest) (Response, exchangeTiming, error) {
	call := &bundleCall{ctx: ctx, task: api.BundleTask{Decisions: &body}, done: make(chan struct{}), decision: decision, sent: make(chan struct{})}
	if err := b.park(client, call); err != nil {
		return Response{}, exchangeTiming{}, err
	}
	return decision.response, call.timing, call.err
}

// park adds a call to the bundle and waits until it is answered or its
// caller stops waiting.
func (b *Bundle) park(client *Client, call *bundleCall) error {
	p := b.participantOf(call.ctx)
	asks := call.decision != nil
	b.mu.Lock()
	b.seq++
	call.task.Id = strconv.Itoa(b.seq)
	call.client = client
	if asks {
		deployment := call.decision.deployment
		if len(b.questions[deployment]) == 0 {
			b.since[deployment] = time.Now()
		}
		b.questions[deployment] = append(b.questions[deployment], call)
	} else {
		if len(b.pending) == 0 {
			b.first = time.Now()
		}
		b.pending[client] = append(b.pending[client], call)
	}
	b.blockLocked(p, asks, 1)
	b.sendReadyLocked()
	if len(b.pending) > 0 && b.timer == nil {
		b.timer = time.AfterFunc(b.window, b.flushOnTimer)
	}
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		b.blockLocked(p, asks, -1)
		b.mu.Unlock()
	}()
	return call.wait()
}

// wait returns once the call is answered or its caller stops waiting: at its
// own deadline until the call is sent and, for a decisions call, at the
// deadline of the exchange that carries it once it is.
func (c *bundleCall) wait() error {
	select {
	case <-c.done:
		return nil
	case <-c.sent:
		return c.waitCarrier()
	case <-c.ctx.Done():
	}
	select {
	case <-c.done:
		return nil
	case <-c.sent:
		return c.waitCarrier()
	default:
		return c.ctx.Err()
	}
}

func (c *bundleCall) waitCarrier() error {
	select {
	case <-c.done:
		return nil
	case <-c.carrier.Done():
	}
	select {
	case <-c.done:
		return nil
	default:
		return c.carrier.Err()
	}
}

// sendReadyLocked sends the calls whose senders are ready: the other calls
// when no participant is still running, and each deployment's questions once
// all of its askers have asked.
func (b *Bundle) sendReadyLocked() {
	others := len(b.pending) > 0 && b.blocked >= b.active
	var deployments []string
	if b.openBlocked >= b.open {
		for deployment, calls := range b.questions {
			if len(calls) > 0 && b.asked[deployment] >= b.askers[deployment] {
				deployments = append(deployments, deployment)
			}
		}
	}
	if others || len(deployments) > 0 {
		slices.Sort(deployments)
		b.sendLocked(others, deployments)
	}
}

func (b *Bundle) flushOnTimer() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.timer = nil
	if len(b.pending) > 0 {
		b.sendLocked(true, nil)
	}
}

// sendLocked sends the pending other calls (when others is set) and the
// questions to the given deployments; the caller holds b.mu.
func (b *Bundle) sendLocked(others bool, deployments []string) {
	byClient := make(map[*Client][]*bundleTask)
	var order []*Client
	add := func(client *Client, tasks ...*bundleTask) {
		if _, seen := byClient[client]; !seen {
			order = append(order, client)
		}
		byClient[client] = append(byClient[client], tasks...)
	}
	if others {
		if b.timer != nil {
			b.timer.Stop()
			b.timer = nil
		}
		bundleWait.Observe(time.Since(b.first).Seconds())
		for client, calls := range b.pending {
			add(client, fuse(client, calls)...)
		}
		b.pending = make(map[*Client][]*bundleCall)
	}
	questions := make(map[*Client][]*bundleCall)
	var asked []*Client
	for _, deployment := range deployments {
		questionWait.Observe(time.Since(b.since[deployment]).Seconds())
		for _, call := range b.questions[deployment] {
			if _, seen := questions[call.client]; !seen {
				asked = append(asked, call.client)
			}
			questions[call.client] = append(questions[call.client], call)
		}
		delete(b.questions, deployment)
		delete(b.since, deployment)
	}
	for _, client := range asked {
		tasks := fuseDecisions(client, questions[client])
		for _, task := range tasks {
			b.countLocked(task)
		}
		add(client, tasks...)
	}
	b.flushes++
	for _, client := range order {
		tasks := answerCached(byClient[client])
		limit := max(1, int(client.bundleTasks.Load()))
		for len(tasks) > 0 {
			part := tasks[:min(len(tasks), limit)]
			tasks = tasks[len(part):]
			ctx, cancel := b.carrier(part)
			for _, task := range part {
				for _, call := range task.calls {
					if call.sent != nil {
						call.carrier = ctx
						close(call.sent)
					}
				}
			}
			go b.send(ctx, cancel, client, part)
		}
	}
}

// countLocked counts a decisions task against each deployment it asks: a
// stage's first call to a deployment, or a later one, which means the
// stage's questions to it did not all travel in one call.
func (b *Bundle) countLocked(task *bundleTask) {
	if task.decision == nil {
		return
	}
	var deployments []string
	for _, call := range task.calls {
		deployments = append(deployments, call.decision.deployment)
	}
	slices.Sort(deployments)
	for _, deployment := range slices.Compact(deployments) {
		b.decisionCalls[deployment]++
		if b.decisionCalls[deployment] == 1 {
			stageDecisionCalls.WithLabelValues(deployment, "first").Inc()
			continue
		}
		stageDecisionCalls.WithLabelValues(deployment, "later").Inc()
		logging.ComponentDebugEvent("model_runtime", "stage_decisions_split", map[string]interface{}{
			"deployment": deployment, "call": b.decisionCalls[deployment],
		})
	}
}

// carrier is the context of one /v1/bundle exchange: it lasts as long as the
// latest of its callers waits (no deadline if one of them sets none).
func (b *Bundle) carrier(tasks []*bundleTask) (context.Context, context.CancelFunc) {
	var latest time.Time
	for _, task := range tasks {
		for _, call := range task.calls {
			deadline, ok := call.ctx.Deadline()
			if !ok {
				return context.WithCancel(b.base)
			}
			if deadline.After(latest) {
				latest = deadline
			}
		}
	}
	return context.WithDeadline(b.base, latest)
}

// send makes one /v1/bundle call in its exchange's context.
func (b *Bundle) send(ctx context.Context, cancel context.CancelFunc, client *Client, tasks []*bundleTask) {
	defer cancel()
	request := make([]api.BundleTask, len(tasks))
	for index, task := range tasks {
		request[index] = task.task
	}
	bundleTasks.Observe(float64(len(request)))
	results, timing, err := client.sendBundle(ctx, request)
	for index, task := range tasks {
		for _, call := range task.calls {
			call.timing = timing
		}
		switch {
		case err != nil:
			task.answer(api.BundleResult{}, err)
		case results[index].Id != task.task.Id:
			task.answer(api.BundleResult{}, ErrFailed)
		default:
			task.answer(results[index], nil)
		}
	}
}
