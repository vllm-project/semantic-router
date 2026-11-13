package classification

import (
	"context"
	"errors"
	"maps"
	"math/rand/v2"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// With global.model_catalog.on_signal_overload.action retry, every call of a
// request stage that the decision runtime refuses as overloaded is asked
// again, a rule that fails open and a routing question alike, so that once
// the runtime has capacity the request is read as it is without the overload.
// The wait starts at the stage's first overloaded answer and lasts up to
// wait_ms, within the signals' deadline; a rule's own timeout_ms bounds each
// attempt, not the wait. Three things bound what the waiting requests ask an
// overloaded runtime:
//   - A stage's calls to a deployment ask again together, in rounds, so the
//     questions one call carried travel in one call again.
//   - A stage has at most overloadRounds rounds for a deployment, spread over
//     its wait: at a quarter of it, at half, and at its end; sooner once the
//     deployment answers again. It takes them one at a time, a step apart at
//     least: its calls that reach a round just after it took one ride that
//     one.
//   - While a deployment refuses calls as overloaded, one stage's round at a
//     time asks it, a jittered step apart, from 100 ms doubling to 1 s,
//     whatever the number of requests waiting; the first answer to any call
//     opens the deployment again, and every stage waiting for it has its
//     round then.
//   - A round that finds the deployment still refusing calls, of a stage
//     with more than overloadProbeAbove calls waiting for it, such as a long
//     input's windows, sends only one of them, on its own: the probe. It
//     does not count as one of the stage's rounds.
//
// A rule still unanswered fails as before (reason overloaded), and
// OverloadedRefusal and OverloadedUnread tell the Router when such rules
// decide the request.

const (
	overloadBackoffFirst = 100 * time.Millisecond
	overloadBackoffMost  = time.Second
	// overloadJoin is how soon after a stage's round for a deployment
	// another of its waiting calls rides that round instead of taking one.
	overloadJoin = 50 * time.Millisecond
	// overloadRounds is how many times a stage's calls to a deployment ask
	// again while it waits for capacity.
	overloadRounds = 3
	// overloadProbeAbove is how many of a stage's calls may wait for a
	// deployment before the stage probes it with one call of theirs.
	overloadProbeAbove = 8
)

// signalErrorOverloaded is the signal error of a rule whose deployment
// refused its call as overloaded.
var signalErrorOverloaded = systemOneSignalErrorPrefix + modelservice.ErrorReason(modelservice.ErrOverloaded)

// overloadStep is the jittered step before a busy deployment's steps-th
// round: half to all of 100 ms doubling to 1 s.
func overloadStep(steps int) time.Duration {
	step := min(overloadBackoffFirst<<min(steps, 4), overloadBackoffMost)
	return step/2 + rand.N(step/2+1)
}

// overloadGate is one deployment's state while its decision runtime refuses
// calls as overloaded, which every request that waits for it shares: busy
// since a call was refused and until one is answered, when its next round
// may come, and the channel closed once it answers again.
type overloadGate struct {
	mu     sync.Mutex
	busy   bool
	next   time.Time
	steps  int
	opened chan struct{}
}

// refused records a call the deployment refused as overloaded at now.
func (g *overloadGate) refused(now time.Time) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.busy {
		g.busy, g.steps, g.opened = true, 0, make(chan struct{})
		g.next = now.Add(overloadStep(0))
	}
}

// answered records a call the deployment answered.
func (g *overloadGate) answered() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.busy {
		g.busy = false
		close(g.opened)
	}
}

// due is when the deployment's next round may come, now when it is not busy,
// and the channel closed once it answers again (nil when it is not busy).
func (g *overloadGate) due(now time.Time) (time.Time, <-chan struct{}) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.busy {
		return now, nil
	}
	return g.next, g.opened
}

// claim reports whether a round may ask the deployment at now, and whether
// the deployment is busy: a round may when it is not, or when its next round
// is due, which the claim takes, so the one after comes a longer step later.
func (g *overloadGate) claim(now time.Time) (claimed, busy bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if !g.busy {
		return true, false
	}
	if now.Before(g.next) {
		return false, true
	}
	g.steps++
	g.next = now.Add(overloadStep(g.steps))
	return true, true
}

// overloadGates are the deployments' gates.
type overloadGates struct {
	mu    sync.Mutex
	gates map[string]*overloadGate
}

func (g *overloadGates) of(deployment string) *overloadGate {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.gates == nil {
		g.gates = map[string]*overloadGate{}
	}
	gate := g.gates[deployment]
	if gate == nil {
		gate = &overloadGate{}
		g.gates[deployment] = gate
	}
	return gate
}

type overloadWaitKey struct{}

// overloadWait is a request stage's wait for the decision runtime to have
// capacity: how long it lasts, when it began, once a call met an overloaded
// runtime, and each deployment's rounds.
type overloadWait struct {
	wait   time.Duration
	mu     sync.Mutex
	start  time.Time
	rounds map[string]*deploymentRounds
}

// deploymentRounds are a stage's rounds for one deployment: how many it has
// had and when the last of them was, how many of its calls wait for one, and
// the one they are to ask again in.
type deploymentRounds struct {
	taken   int
	last    time.Time
	waiting int
	next    *overloadRound
}

// overloadRound is when a stage's waiting calls to one deployment ask
// again, and, once the first of them reached it, whether they do.
type overloadRound struct {
	at      time.Time
	decided bool
	asks    bool
}

func withOverloadWait(ctx context.Context, wait time.Duration) context.Context {
	if wait <= 0 {
		return ctx
	}
	return context.WithValue(ctx, overloadWaitKey{}, &overloadWait{wait: wait, rounds: map[string]*deploymentRounds{}})
}

// round is the round a waiting call of the stage to deployment joins at now,
// and the channel the deployment's gate closes once it answers again; false
// once the stage's wait has run out or its rounds are spent. The wait lasts
// wait_ms from the stage's first overloaded answer, or less when the call's
// deadline comes first, and the rounds come at a quarter of it, at half and
// at its end, each a little before; later when the gate holds them back or
// the stage's last round was less than a step before, and within 50 ms when
// the deployment answers again.
func (w *overloadWait) round(ctx context.Context, deployment string, gate *overloadGate, now time.Time) (*overloadRound, <-chan struct{}, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.start.IsZero() {
		w.start = now
	}
	wait := w.wait
	if deadline, ok := ctx.Deadline(); ok {
		wait = min(wait, deadline.Sub(w.start))
	}
	until := w.start.Add(wait)
	rounds := w.roundsOfLocked(deployment)
	if !now.Before(until) || rounds.taken >= overloadRounds {
		return nil, nil, false
	}
	due, opened := gate.due(now)
	if round := rounds.next; round != nil && !round.decided && !due.After(round.at) && (opened != nil || !round.at.After(now)) {
		return round, opened, true
	}
	at := now
	if opened != nil {
		share := wait >> (overloadRounds - 1 - rounds.taken)
		at = w.start.Add(share - share/16 - rand.N(share/16+1))
		if at.Before(due) {
			at = due
		}
		if at.Before(now) {
			at = now
		}
	} else {
		// The deployment answers again: the stage asks at once, a little
		// apart from the other stages that waited for it.
		at = now.Add(rand.N(overloadBackoffFirst/2 + 1))
	}
	// A stage's rounds are a step apart at least, so a deployment that
	// answers some calls and refuses others does not spend them at once.
	if !rounds.last.IsZero() {
		if earliest := rounds.last.Add(overloadStep(rounds.taken)); at.Before(earliest) {
			at = earliest
		}
	}
	if at.After(until) {
		at = until
	}
	rounds.next = &overloadRound{at: at}
	return rounds.next, opened, true
}

func (w *overloadWait) roundsOfLocked(deployment string) *deploymentRounds {
	rounds := w.rounds[deployment]
	if rounds == nil {
		rounds = &deploymentRounds{}
		w.rounds[deployment] = rounds
	}
	return rounds
}

// join counts a call of the stage that starts (n 1) or stops (n -1) waiting
// for deployment.
func (w *overloadWait) join(deployment string, n int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.roundsOfLocked(deployment).waiting += n
}

// asks reports whether the stage's round lets a call ask its deployment
// again: together with the round's other calls, or alone, as its probe. The
// first call to reach the round decides it for all of them. Within
// overloadJoin of the round the stage took last, it rides that round, and
// sooner than half a step after it, it waits for a later one. Otherwise it
// claims the deployment's gate, and probes when the deployment is busy and
// more than overloadProbeAbove calls wait; a round the gate lets counts.
func (w *overloadWait) asks(deployment string, round *overloadRound, gate *overloadGate, now time.Time) (together, alone bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if round.decided {
		return round.asks, false
	}
	round.decided = true
	rounds := w.roundsOfLocked(deployment)
	// A stage takes one round for a deployment at a time: a call that
	// reaches another round of its stage just after one was taken rides it,
	// and one that comes sooner than half a step after it waits for a later
	// one.
	if !rounds.last.IsZero() {
		since := now.Sub(rounds.last)
		if since < overloadJoin {
			round.asks = true
			return true, false
		}
		if since < min(overloadBackoffFirst<<rounds.taken, overloadBackoffMost)/2 {
			return false, false
		}
	}
	claimed, busy := gate.claim(now)
	if !claimed {
		return false, false
	}
	if busy && rounds.waiting > overloadProbeAbove {
		return false, true
	}
	round.asks = true
	rounds.taken++
	rounds.last = now
	return true, false
}

// waitForCapacity waits, once deployment refused a call of the stage as
// overloaded, until a round of the stage's lets it ask again, and reports
// how: together with the round's other calls, or alone, as its probe;
// neither once the stage's wait has run out, its rounds are spent or the
// call's context ended.
func (c *Classifier) waitForCapacity(ctx context.Context, wait *overloadWait, deployment string) (together, alone bool) {
	if ctx.Err() != nil {
		return false, false
	}
	gate := c.overloadGates.of(deployment)
	wait.join(deployment, 1)
	defer wait.join(deployment, -1)
	for {
		round, opened, ok := wait.round(ctx, deployment, gate, time.Now())
		if !ok {
			return false, false
		}
		timer := time.NewTimer(time.Until(round.at))
		select {
		case <-ctx.Done():
			timer.Stop()
			return false, false
		case <-opened:
			timer.Stop()
		case <-timer.C:
		}
		if ctx.Err() != nil {
			return false, false
		}
		if together, alone = wait.asks(deployment, round, gate, time.Now()); together || alone {
			return together, alone
		}
	}
}

// askUntilCapacity asks a call that ask sends with the context it is given,
// and, while the decision runtime answers it overloaded and its stage waits
// for capacity, asks it again in the stage's rounds. It returns the last
// answer and whether the call waited. With on_signal_overload retry, every
// call tells its deployment's gate how it ended, a stage's that does not
// wait as well.
//
// While the call waits, its stage's bundle counts it as blocked, so it holds
// back no other work, but not as asked, so the questions it carries travel
// with those of the round's other calls. A probe goes out of the bundle, and
// meanwhile still holds back the stage's calls to its deployment.
func (c *Classifier) askUntilCapacity(ctx context.Context, deployment string, ask func(context.Context) (modelservice.Response, error)) (modelservice.Response, bool, error) {
	response, err := ask(ctx)
	wait, waits := ctx.Value(overloadWaitKey{}).(*overloadWait)
	if !waits && !c.Config.SignalOverloadRetries() {
		return response, false, err
	}
	gate := c.overloadGates.of(deployment)
	report := func() {
		switch {
		case err == nil:
			gate.answered()
		case errors.Is(err, modelservice.ErrOverloaded):
			gate.refused(time.Now())
		}
	}
	report()
	if !waits {
		return response, false, err
	}
	waited := false
	for errors.Is(err, modelservice.ErrOverloaded) {
		waited = true
		resume := modelservice.Pause(ctx)
		together, alone := c.waitForCapacity(ctx, wait, deployment)
		asking := ctx
		if alone {
			asking = modelservice.Unbundled(ctx)
		} else {
			resume()
		}
		if !together && !alone {
			break
		}
		again, failed := ask(asking)
		resume()
		if failed != nil && !errors.Is(failed, modelservice.ErrOverloaded) && ctx.Err() != nil {
			// The stage ended during the attempt: the rule stays unread for
			// the overload, not for the stage's deadline.
			break
		}
		response, err = again, failed
		report()
	}
	if waited && !answersGivenUp(ctx) {
		metrics.RecordSignalOverloadWait(deployment, modelservice.ErrorReason(err))
	}
	return response, waited, err
}

// overloadedFailures are the rules ("type:name") that failed because their
// deployment refused their calls as overloaded, and among them those that
// failed closed (matched by policy).
func overloadedFailures(signals *SignalResults) (unread, failedClosed []string) {
	for key, reason := range signals.SignalErrors {
		if reason != signalErrorOverloaded {
			continue
		}
		if signals.SignalErrorMatches[key] {
			failedClosed = append(failedClosed, key)
		} else {
			unread = append(unread, key)
		}
	}
	slices.Sort(unread)
	slices.Sort(failedClosed)
	return unread, failedClosed
}

// OverloadedRefusal is, when result refuses the request (its decision has a
// fast_response) only because rules failed closed on an overloaded decision
// runtime, those rules, by "type:name"; nil when it refuses anyway: when the
// decisions would still refuse had those rules stayed unknown, because
// another rule matched or a condition matches on error.
func (c *Classifier) OverloadedRefusal(signals *SignalResults, candidates []config.Decision, result *decision.DecisionResult) []string {
	if c == nil || c.Config == nil || signals == nil || result == nil || result.Decision == nil || result.Decision.GetFastResponseConfig() == nil {
		return nil
	}
	_, overloaded := overloadedFailures(signals)
	if len(overloaded) == 0 || c.refuses(signals.withoutPolicyMatches(overloaded), candidates) {
		return nil
	}
	return overloaded
}

// OverloadedUnread is, when result does not refuse the request, the rules
// that stayed unknown because their deployment refused their calls as
// overloaded and whose match would have refused it, by "type:name": the
// request is not forwarded with them unread. A rule that fails open, such as
// safety, does not match when it fails, but its overload says nothing of the
// request. nil when there are none.
func (c *Classifier) OverloadedUnread(signals *SignalResults, candidates []config.Decision, result *decision.DecisionResult) []string {
	if c == nil || c.Config == nil || signals == nil || result != nil && result.Decision != nil && result.Decision.GetFastResponseConfig() != nil {
		return nil
	}
	keys, _ := overloadedFailures(signals)
	var unread []string
	var rules []askedRule
	for _, key := range keys {
		if asked, ok := c.askedRuleOf(key); ok {
			unread, rules = append(unread, key), append(rules, asked)
		}
	}
	if len(rules) == 0 {
		return nil
	}
	// One rule's match can make a decision above the refusing one win, so
	// each rule is asked alone as well.
	if c.refuses(signals.withPolicyMatches(rules), candidates) || len(rules) > 1 && slices.ContainsFunc(rules, func(asked askedRule) bool {
		return c.refuses(signals.withPolicyMatches([]askedRule{asked}), candidates)
	}) {
		return unread
	}
	return nil
}

// refuses reports whether the decisions refuse a request with these
// results. The traced evaluation records no decision metrics: it asks what
// the decisions would say, not what the request gets.
func (c *Classifier) refuses(signals *SignalResults, candidates []config.Decision) bool {
	decisions := c.Config.Decisions
	if candidates != nil {
		decisions = candidates
	}
	engine := decision.NewDecisionEngine(decisions, c.Config.Strategy).WithRoutingScope(c.Config.RoutingScope)
	result, _, _, err := engine.EvaluateDecisionsWithTraceAndDiagnostics(signals.Matches())
	return err == nil && result != nil && result.Decision != nil && result.Decision.GetFastResponseConfig() != nil
}

// askedRuleOf is the request-stage rule key names ("type:name").
func (c *Classifier) askedRuleOf(key string) (askedRule, bool) {
	signalType, name, ok := strings.Cut(key, ":")
	if !ok {
		return askedRule{}, false
	}
	if signalType == config.SignalTypeSystemOne {
		for _, rule := range c.Config.SystemOneRules {
			if rule.Name == name {
				return systemOneAsked(rule), true
			}
		}
		return askedRule{}, false
	}
	for _, rule := range c.applicationRules {
		if rule.Type == signalType && rule.Rule.Name == name {
			return applicationAsked(rule), true
		}
	}
	return askedRule{}, false
}

// withPolicyMatches is a copy of the results in which rules match by policy,
// as a rule that fails closed does when it fails: by itself and by its
// counted labels.
func (r *SignalResults) withPolicyMatches(rules []askedRule) *SignalResults {
	copied := *r
	copied.SignalErrorMatches = maps.Clone(r.SignalErrorMatches)
	if copied.SignalErrorMatches == nil {
		copied.SignalErrorMatches = map[string]bool{}
	}
	copied.labelReferences = maps.Clone(r.labelReferences)
	for _, asked := range rules {
		copied.SignalErrorMatches[signalConfidenceKey(asked.signalType, asked.rule.Name)] = true
		matched := copied.matchedRulesOf(asked.signalType)
		if matched == nil {
			continue
		}
		*matched = slices.Clone(*matched)
		if asked.rule.MatchesItself() {
			copied.appendMatched(asked.signalType, asked.rule, asked.rule.Name)
		}
		for _, label := range asked.rule.CountedLabels() {
			copied.appendMatched(asked.signalType, asked.rule, asked.rule.Name+":"+label)
		}
	}
	return &copied
}

// withoutPolicyMatches is a copy of the results in which the rules keys
// names ("type:name") match neither by themselves nor by label: they failed
// and stayed unknown.
func (r *SignalResults) withoutPolicyMatches(keys []string) *SignalResults {
	copied := *r
	copied.SignalErrorMatches = make(map[string]bool, len(r.SignalErrorMatches))
	for key, matched := range r.SignalErrorMatches {
		if !slices.Contains(keys, key) {
			copied.SignalErrorMatches[key] = matched
		}
	}
	for _, key := range keys {
		signalType, name, ok := strings.Cut(key, ":")
		if !ok {
			continue
		}
		if matched := copied.matchedRulesOf(signalType); matched != nil {
			*matched = slices.DeleteFunc(slices.Clone(*matched), func(reference string) bool {
				return reference == name || strings.HasPrefix(reference, name+":")
			})
		}
	}
	return &copied
}
