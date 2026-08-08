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
