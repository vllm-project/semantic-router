package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func retBool(b bool) *bool { return &b }
func retInt(i int) *int    { return &i }

func TestShouldSkipCacheWrite(t *testing.T) {
	cases := []struct {
		name       string
		ctx        *RequestContext
		wantSkip   bool
		wantReason string
	}{
		{name: "nil_ctx", ctx: nil, wantSkip: false},
		{name: "no_retention", ctx: &RequestContext{}, wantSkip: false},
		{
			name:     "retention_without_drop",
			ctx:      &RequestContext{EmittedRetention: &config.RetentionDirective{TTLTurns: intPtr(3)}},
			wantSkip: false,
		},
		{
			name:     "drop_false",
			ctx:      &RequestContext{EmittedRetention: &config.RetentionDirective{Drop: boolPtr(false)}},
			wantSkip: false,
		},
		{
			name:       "drop_true",
			ctx:        &RequestContext{EmittedRetention: &config.RetentionDirective{Drop: boolPtr(true)}},
			wantSkip:   true,
			wantReason: "retention.drop",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			skip, reason := ShouldSkipCacheWrite(tc.ctx)
			if skip != tc.wantSkip || reason != tc.wantReason {
				t.Fatalf("got (%v,%q), want (%v,%q)", skip, reason, tc.wantSkip, tc.wantReason)
			}
		})
	}
}

func TestApplyEmittedRetentionDeepClone(t *testing.T) {
	src := &config.Decision{
		Name: "r",
		Emits: []config.EmitDirective{
			{Kind: "retention", Retention: &config.RetentionDirective{
				Drop:                  boolPtr(true),
				TTLTurns:              intPtr(5),
				KeepCurrentModel:      boolPtr(true),
				PreferPrefixRetention: boolPtr(false),
			}},
		},
	}
	ctx := &RequestContext{}
	got := applyEmittedRetention(src, ctx)
	if got == nil || ctx.EmittedRetention != got {
		t.Fatalf("expected ctx.EmittedRetention to be set to returned clone")
	}
	if ctx.EmittedRetention.Drop == nil || !*ctx.EmittedRetention.Drop {
		t.Fatalf("drop missing on clone")
	}

	// Mutate the clone; source must remain untouched (no pointer aliasing).
	*ctx.EmittedRetention.Drop = false
	*ctx.EmittedRetention.TTLTurns = 99
	if !*src.Emits[0].Retention.Drop {
		t.Fatalf("source Drop polluted by clone mutation")
	}
	if *src.Emits[0].Retention.TTLTurns != 5 {
		t.Fatalf("source TTLTurns polluted by clone mutation, got %d", *src.Emits[0].Retention.TTLTurns)
	}
}

func TestApplyEmittedRetentionNoMatch(t *testing.T) {
	src := &config.Decision{Name: "r"} // no emits
	ctx := &RequestContext{}
	if got := applyEmittedRetention(src, ctx); got != nil {
		t.Fatalf("expected nil when decision has no emits")
	}
	if ctx.EmittedRetention != nil {
		t.Fatalf("expected EmittedRetention to stay nil")
	}
}

func TestApplyEmittedRetentionNilInputs(t *testing.T) {
	if got := applyEmittedRetention(nil, &RequestContext{}); got != nil {
		t.Fatalf("nil decision must yield nil")
	}
	if got := applyEmittedRetention(&config.Decision{}, nil); got != nil {
		t.Fatalf("nil ctx must yield nil")
	}
}

func TestObserveRetentionDirectiveNoOp(t *testing.T) {
	// Must not panic on nil ctx / nil retention.
	observeRetentionDirective(nil)
	observeRetentionDirective(&RequestContext{})
}
