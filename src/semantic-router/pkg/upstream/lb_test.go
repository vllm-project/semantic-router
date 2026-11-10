package upstream

import (
	"math/rand/v2"
	"testing"
)

// scriptedRand returns the given values in order, then repeats the last.
type scriptedRand struct {
	values []uint64
	next   int
}

func (r *scriptedRand) Uint64() uint64 {
	v := r.values[min(r.next, len(r.values)-1)]
	r.next++
	return v
}

func testEndpoints(weights ...int) []*endpoint {
	endpoints := make([]*endpoint, len(weights))
	for i, w := range weights {
		endpoints[i] = &endpoint{spec: EndpointSpec{Name: string(rune('a' + i)), Weight: w}}
	}
	return endpoints
}

func pickCounts(b balancer, set *hostSet, n int) map[string]int {
	counts := map[string]int{}
	for range n {
		counts[b.pick(set).spec.Name]++
	}
	return counts
}

func TestRoundRobinEqualWeightsRotatesFromSeededStart(t *testing.T) {
	set := newHostSet(testEndpoints(1, 1, 1))
	// The first draw seeds the weighted schedule, the second the rotation.
	b := newBalancer(LBRoundRobin, &scriptedRand{values: []uint64{0, 4}})
	var got []string
	for range 6 {
		got = append(got, b.pick(set).spec.Name)
	}
	want := []string{"b", "c", "a", "b", "c", "a"}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("picks = %v, want %v", got, want)
		}
	}
}

func TestRoundRobinFollowsWeightsExactly(t *testing.T) {
	for _, seed := range []uint64{0, 1, 7, 12345} {
		set := newHostSet(testEndpoints(3, 1, 2))
		b := newBalancer(LBRoundRobin, rand.New(rand.NewPCG(seed, seed)))
		// Every full cycle of the schedule holds each host weight times; a
		// seeded start may split one cycle across the ends of the window.
		counts := pickCounts(b, set, 600)
		for name, want := range map[string]int{"a": 300, "b": 100, "c": 200} {
			if diff := counts[name] - want; diff < -3 || diff > 3 {
				t.Fatalf("seed %d: counts = %v, want about a=300 b=100 c=200", seed, counts)
			}
		}
	}
}

func TestRoundRobinWeightedScheduleInterleaves(t *testing.T) {
	set := newHostSet(testEndpoints(3, 1))
	b := &roundRobin{schedule: edfCache{seed: 0}}
	var got string
	for range 8 {
		got += b.pick(set).spec.Name
	}
	// Envoy's EDF schedule from an unpicked start: ties go to the entry
	// queued first.
	if got != "aabaaaba" {
		t.Fatalf("schedule = %q, want %q", got, "aabaaaba")
	}
}

func TestLeastRequestEqualWeightsKeepsFewerActive(t *testing.T) {
	endpoints := testEndpoints(1, 1, 1)
	endpoints[0].active.Store(5)
	endpoints[1].active.Store(2)
	endpoints[2].active.Store(9)
	set := newHostSet(endpoints)
	tests := []struct {
		samples []uint64
		want    string
	}{
		{samples: []uint64{0, 1}, want: "b"},
		{samples: []uint64{2, 0}, want: "a"},
		{samples: []uint64{2, 2}, want: "c"},
		// Ties keep the first sample.
		{samples: []uint64{1, 1}, want: "b"},
	}
	for _, tt := range tests {
		b := &leastRequest{rnd: &scriptedRand{values: tt.samples}, choices: leastRequestChoices}
		if got := b.pick(set).spec.Name; got != tt.want {
			t.Fatalf("samples %v picked %s, want %s", tt.samples, got, tt.want)
		}
	}
}

func TestLeastRequestUnequalWeightsDiscountsActiveRequests(t *testing.T) {
	endpoints := testEndpoints(2, 1)
	// Effective weights: a = 2/(3+1) = 0.5, b = 1/(0+1) = 1.
	endpoints[0].active.Store(3)
	set := newHostSet(endpoints)
	b := newBalancer(LBLeastRequest, rand.New(rand.NewPCG(3, 4)))
	counts := pickCounts(b, set, 300)
	if diff := counts["b"] - 200; diff < -2 || diff > 2 {
		t.Fatalf("counts = %v, want about a=100 b=200", counts)
	}
}

func TestEDFSchedulerSeededPicksKeepProportions(t *testing.T) {
	hosts := testEndpoints(5, 3, 2)
	for _, picks := range []uint32{0, 1, 9, 10, 37, 1001} {
		s := newEDFScheduler(hosts, configuredWeight, picks)
		counts := map[string]int{}
		for range 100 {
			counts[s.pickAndAdd(configuredWeight).spec.Name]++
		}
		for name, want := range map[string]int{"a": 50, "b": 30, "c": 20} {
			if diff := counts[name] - want; diff < -2 || diff > 2 {
				t.Fatalf("picks %d: counts = %v, want about a=50 b=30 c=20", picks, counts)
			}
		}
	}
}

func TestBalancersHandleEmptyAndSingleHostSets(t *testing.T) {
	for _, policy := range []LBPolicy{LBRoundRobin, LBLeastRequest} {
		b := newBalancer(policy, globalRand{})
		if got := b.pick(newHostSet(nil)); got != nil {
			t.Fatalf("%s picked %v from an empty set", policy, got)
		}
		single := newHostSet(testEndpoints(4))
		if got := b.pick(single); got != single.hosts[0] {
			t.Fatalf("%s did not pick the only host", policy)
		}
	}
}
