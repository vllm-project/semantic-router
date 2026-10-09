package upstream

import (
	"container/heap"
	"math/rand/v2"
	"sync"
	"sync/atomic"
)

// Rand is the randomness a load balancer draws from. Tests inject a seeded
// source to make picks deterministic.
type Rand interface {
	Uint64() uint64
}

type globalRand struct{}

func (globalRand) Uint64() uint64 { return rand.Uint64() }

// lockedRand makes an injected source safe for concurrent picks.
type lockedRand struct {
	mu  sync.Mutex
	src Rand
}

func (r *lockedRand) Uint64() uint64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.src.Uint64()
}

// hostSet is the endpoint list one pick chooses from. A cluster replaces it
// whole when endpoint health changes, so its identity tells a balancer to
// rebuild a schedule derived from it.
type hostSet struct {
	hosts        []*endpoint
	equalWeights bool
}

func newHostSet(hosts []*endpoint) *hostSet {
	equal := true
	for _, host := range hosts[min(1, len(hosts)):] {
		if host.spec.Weight != hosts[0].spec.Weight {
			equal = false
			break
		}
	}
	return &hostSet{hosts: hosts, equalWeights: equal}
}

type balancer interface {
	pick(set *hostSet) *endpoint
}

func newBalancer(policy LBPolicy, rnd Rand) balancer {
	if policy == LBLeastRequest {
		return &leastRequest{rnd: rnd, choices: leastRequestChoices, schedule: edfCache{seed: rnd.Uint64()}}
	}
	b := &roundRobin{schedule: edfCache{seed: rnd.Uint64()}}
	b.next.Store(rnd.Uint64())
	return b
}

// roundRobin rotates over equally weighted hosts from a random starting
// index, and follows an earliest-deadline-first schedule when weights differ,
// as Envoy's round robin balancer does.
type roundRobin struct {
	next     atomic.Uint64
	schedule edfCache
}

func (b *roundRobin) pick(set *hostSet) *endpoint {
	switch n := len(set.hosts); {
	case n == 0:
		return nil
	case n == 1:
		return set.hosts[0]
	case set.equalWeights:
		return set.hosts[(b.next.Add(1)-1)%uint64(n)]
	}
	return b.schedule.pick(set, configuredWeight)
}

// leastRequestChoices is the choice_count the Envoy template configures.
const leastRequestChoices = 2

// leastRequest samples hosts at random (with replacement) and keeps the one
// with the fewest active requests when weights are equal. When they differ it
// follows an earliest-deadline-first schedule over weight / (active + 1), as
// Envoy's least request balancer does with its default active request bias.
type leastRequest struct {
	rnd      Rand
	choices  int
	schedule edfCache
}

func (b *leastRequest) pick(set *hostSet) *endpoint {
	switch n := len(set.hosts); {
	case n == 0:
		return nil
	case n == 1:
		return set.hosts[0]
	case set.equalWeights:
		var best *endpoint
		for range b.choices {
			candidate := set.hosts[b.rnd.Uint64()%uint64(n)]
			if best == nil || candidate.active.Load() < best.active.Load() {
				best = candidate
			}
		}
		return best
	}
	return b.schedule.pick(set, loadedWeight)
}

func configuredWeight(e *endpoint) float64 { return float64(e.spec.Weight) }

func loadedWeight(e *endpoint) float64 {
	return float64(e.spec.Weight) / float64(e.active.Load()+1)
}

// edfCache keeps the schedule built for the current host set.
type edfCache struct {
	mu       sync.Mutex
	seed     uint64
	set      *hostSet
	schedule *edfScheduler
}

func (c *edfCache) pick(set *hostSet, weight func(*endpoint) float64) *endpoint {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.set != set {
		// Envoy seeds the schedule with a 32-bit pick count.
		c.schedule = newEDFScheduler(set.hosts, weight, uint32(c.seed>>32))
		c.set = set
	}
	return c.schedule.pickAndAdd(weight)
}

// edfScheduler is Envoy's earliest-deadline-first weighted schedule: every
// entry is due at the time of its last pick plus 1 / weight, and ties go to
// the entry queued first.
type edfScheduler struct {
	now   float64
	order uint64
	queue edfQueue
}

type edfEntry struct {
	deadline float64
	order    uint64
	host     *endpoint
}

func (s *edfScheduler) pickAndAdd(weight func(*endpoint) float64) *endpoint {
	if len(s.queue) == 0 {
		return nil
	}
	entry := heap.Pop(&s.queue).(edfEntry)
	s.now = entry.deadline
	heap.Push(&s.queue, edfEntry{deadline: s.now + 1/weight(entry.host), order: s.order, host: entry.host})
	s.order++
	return entry.host
}

// newEDFScheduler builds the schedule as if picks picks had already been made
// from it, so balancers seeded differently start at different points of the
// rotation. It follows Envoy's EdfScheduler::createWithPicks, including the
// epsilon that keeps weights that are multiples of each other exact.
func newEDFScheduler(hosts []*endpoint, weight func(*endpoint) float64, picks uint32) *edfScheduler {
	const maxPicks = 429496729
	picks %= maxPicks
	weights := make([]float64, len(hosts))
	var sum float64
	for i, host := range hosts {
		weights[i] = weight(host) + 1e-13
		sum += weights[i]
	}
	s := &edfScheduler{order: uint64(len(hosts)), queue: make(edfQueue, 0, len(hosts))}
	var picked uint32
	for i, host := range hosts {
		w := weights[i]
		floor := uint32(w * float64(picks) / sum)
		if floor > 0 && float64(floor)/w >= float64(picks)/sum {
			floor--
		}
		s.now = max(s.now, float64(floor)/w)
		s.queue = append(s.queue, edfEntry{deadline: float64(floor+1) / w, order: uint64(i), host: host})
		picked += floor
	}
	heap.Init(&s.queue)
	for ; picked < picks; picked++ {
		s.pickAndAdd(weight)
	}
	return s
}

type edfQueue []edfEntry

func (q edfQueue) Len() int { return len(q) }

func (q edfQueue) Less(i, j int) bool {
	if q[i].deadline != q[j].deadline {
		return q[i].deadline < q[j].deadline
	}
	return q[i].order < q[j].order
}

func (q edfQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }

func (q *edfQueue) Push(x any) { *q = append(*q, x.(edfEntry)) }

func (q *edfQueue) Pop() any {
	old := *q
	entry := old[len(old)-1]
	*q = old[:len(old)-1]
	return entry
}
