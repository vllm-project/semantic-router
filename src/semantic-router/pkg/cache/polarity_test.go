package cache

import "testing"

func TestPolarityMismatch(t *testing.T) {
	tests := []struct {
		name     string
		incoming string
		cached   string
		want     bool
	}{
		// A negation cue on one side must reject.
		{"not-inserted", "Should I commit this change?", "Should I not commit this change?", true},
		{"not-inserted-reverse", "Should I not commit this change?", "Should I commit this change?", true},
		{"require-not-require", "Does this feature require a license?", "Does this feature not require a license?", true},
		{"contraction-nt", "Is the cache enabled?", "Isn't the cache enabled?", true},
		{"contraction-cant", "Can I commit this change?", "Can't I commit this change?", true},
		{"contraction-curly-cant", "Can I commit this change?", "Can’t I commit this change?", true},
		{"contraction-wont", "Will it retry the request?", "Won't it retry the request?", true},
		{"contraction-doesnt", "Does it require a license?", "Doesn't it require a license?", true},
		{"contraction-aint", "Is the cache enabled?", "Ain't the cache enabled?", true},
		{"contraction-aint-are", "Are the workers ready?", "Ain't the workers ready?", true},
		{"without-cue", "Deploy with a sidecar", "Deploy without a sidecar", true},
		{"never-cue", "Should I retry the request?", "Should I never retry the request?", true},

		// Antonym swaps must reject.
		{"on-off", "How to turn on dark mode?", "How to turn off dark mode?", true},
		{"enable-disable", "How do I enable two-factor authentication?", "How do I disable two-factor authentication?", true},
		{"enabled-disabled", "Is caching enabled in production?", "Is caching disabled in production?", true},
		{"open-closed", "Is port 6379 open by default?", "Is port 6379 closed by default?", true},
		{"active-inactive", "Is the API rate limit currently active?", "Is the API rate limit currently inactive?", true},
		{"increase-decrease", "Can I increase my storage quota?", "Can I decrease my storage quota?", true},
		{"start-stop", "How do I start the service?", "How do I stop the service?", true},
		{"add-remove-direct", "How do I add a tag?", "How do I remove a tag?", true},
		{"grant-revoke-direct", "How do I grant admin access to the dashboard?", "How do I revoke admin access to the dashboard?", true},

		// Genuine paraphrases must remain eligible.
		{"paraphrase-password", "How do I reset my password?", "What's the way to reset my password?", false},
		{"paraphrase-2fa", "How to enable two-factor auth?", "How do I turn on 2FA?", false},
		{"paraphrase-logs", "Where are the logs stored?", "What location holds the log files?", false},

		// Unrelated or merely different queries must remain eligible.
		{"identical", "How do I enable 2FA?", "How do I enable 2FA?", false},
		{"different-topic", "How do I rotate my API key?", "How do I delete my account?", false},

		// An unpaired antonym token must not reject.
		{"unpaired-on", "How to turn on dark mode?", "How to turn on light mode?", false},

		// The surface gate excludes distant antonym pairs.
		{"antonym-but-far", "How do I enable the new dashboard widget?", "How do I disable the old sidebar menu?", false},

		// Ambiguous cues must be read in context, not matched bare. The tier is
		// unconditional, so a false positive costs recall with no opt-out.
		{"no-abbreviates-number", "Show invoice no. 123", "Show invoice 123", false},
		{"no-abbreviates-number-reverse", "Show invoice 123", "Show invoice no. 123", false},
		{"no-abbreviates-number-bare", "Show invoice no 123", "Show invoice 123", false},
		{"no-negates", "Is there a way to disable it?", "Is there no way to disable it?", true},
		{"no-compound-code", "Is there a no-code option?", "Is there a code option?", false},
		{"no-compound-cache", "How do I set no-cache headers?", "How do I set cache headers?", false},
		{"no-abbreviates-alnum-id", "Show invoice no. A123", "Show invoice A123", false},
		{"on-off-as-preposition", "routing based on embeddings", "routing based off embeddings", false},
		{"on-off-switches-state", "Is dark mode on?", "Is dark mode off?", true},
		{"on-off-switches-state-detached", "How do I turn it on?", "How do I turn it off?", true},

		// Do-support and elided auxiliaries must not push a negation past the
		// token gate before its cue is reached.
		{"do-support-negation", "This change works", "This change doesn't work", true},
		{"be-contraction-negation", "It's safe to deploy", "It isn't safe to deploy", true},
		{"do-support-negation-plain", "This change works", "This change does not work", true},
		{"have-contraction-negation", "It has worked since the upgrade", "It hasn't worked since the upgrade", true},
		{"modal-cannot", "I can deploy to staging", "I cannot deploy to staging", true},

		// Preposition changes can push a pair beyond the token gate.
		{"add-remove-unguarded", "How do I add a member to the team?", "How do I remove a member from the team?", false},
		{"grant-revoke-unguarded", "How do I grant access to the bucket?", "How do I revoke access from the bucket?", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := polarityMismatch(tt.incoming, tt.cached); got != tt.want {
				t.Errorf("polarityMismatch(%q, %q) = %v, want %v", tt.incoming, tt.cached, got, tt.want)
			}
		})
	}
}
