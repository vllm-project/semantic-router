package kvtransfer

import (
	"context"
	"errors"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"testing"
)

type sourceLookupStub struct {
	source             *SourceCache
	err                error
	calls              int
	namespace, session string
}

func (s *sourceLookupStub) LookupSource(_ context.Context, namespace, session string) (*SourceCache, error) {
	s.calls++
	s.namespace, s.session = namespace, session
	return s.source, s.err
}

func TestCoordinatorScopedSwitch(t *testing.T) {
	r, s, m := testHandoff(t)
	l := &sourceLookupStub{source: &s}
	c := &Coordinator{Lookup: l, Policies: []PairPolicy{{Mapper: m, Enabled: true, MaxTransferTurn: 1}}}
	h, reason := c.Plan(context.Background(), r, 1)
	if reason != ReasonEligible || h == nil {
		t.Fatalf("Plan = %v, %s", h, reason)
	}
	if l.namespace != s.Namespace || l.session != s.SessionID {
		t.Fatal("incorrect lookup scope")
	}
	wire := h.Headers()
	if len(wire) != 3 || wire[headers.VSRKVSourcePod] != s.Endpoint || wire[headers.VSRKVCacheID] != s.CacheID || wire[headers.VSRKVMapperID] != m.ID {
		t.Fatalf("headers = %v", wire)
	}
}

func TestCoordinatorSafeFallbacks(t *testing.T) {
	cases := []struct {
		name   string
		change func(*Coordinator, *Request, *SourceCache, *sourceLookupStub)
		turn   int
		want   Reason
		calls  int
	}{
		{"disabled", func(c *Coordinator, _ *Request, _ *SourceCache, _ *sourceLookupStub) { c.Policies[0].Enabled = false }, 0, ReasonPairNotAllowed, 0},
		{"turn limit", func(*Coordinator, *Request, *SourceCache, *sourceLookupStub) {}, 1, ReasonTurnLimit, 0},
		{"untrusted", func(_ *Coordinator, r *Request, _ *SourceCache, _ *sourceLookupStub) {
			r.SessionProvenance = "message_hash"
		}, 0, ReasonUntrustedSession, 0},
		{"registry error", func(_ *Coordinator, _ *Request, _ *SourceCache, l *sourceLookupStub) { l.err = errors.New("offline") }, 0, ReasonSourceUnavailable, 1},
		{"registry miss", func(_ *Coordinator, _ *Request, _ *SourceCache, l *sourceLookupStub) { l.source = nil }, 0, ReasonSourceUnavailable, 1},
		{"namespace mismatch", func(_ *Coordinator, _ *Request, s *SourceCache, _ *sourceLookupStub) { s.Namespace = "other" }, 0, ReasonScopeMismatch, 1},
		{"expired", func(_ *Coordinator, r *Request, s *SourceCache, _ *sourceLookupStub) { s.ExpiresAt = r.Now }, 0, ReasonStaleSource, 1},
		{"same model", func(c *Coordinator, r *Request, s *SourceCache, _ *sourceLookupStub) {
			s.Model = r.Target
			c.Policies[0].Mapper.Source = r.Target
		}, 0, ReasonSameModel, 1},
		{"unknown pair", func(_ *Coordinator, _ *Request, s *SourceCache, _ *sourceLookupStub) { s.Model.Model = "other" }, 0, ReasonPairNotAllowed, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r, s, m := testHandoff(t)
			l := &sourceLookupStub{source: &s}
			c := &Coordinator{Lookup: l, Policies: []PairPolicy{{Mapper: m, Enabled: true}}}
			tc.change(c, &r, &s, l)
			h, reason := c.Plan(context.Background(), r, tc.turn)
			if h != nil || reason != tc.want || l.calls != tc.calls {
				t.Fatalf("Plan = %v, %s, calls %d", h, reason, l.calls)
			}
		})
	}
}
