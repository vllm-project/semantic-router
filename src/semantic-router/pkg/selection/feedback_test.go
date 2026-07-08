/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"context"
	"errors"
	"math"
	"testing"
)

func TestNormalizeFeedbackCanonicalizesStateKeys(t *testing.T) {
	feedback := &Feedback{
		WinnerModel:    " model-a ",
		LoserModel:     " model-b ",
		DecisionName:   " coding ",
		UserID:         " user-1 ",
		SessionID:      " session-1 ",
		ConversationID: " conversation-1 ",
		FeedbackType:   " satisfied ",
		Confidence:     0.75,
	}

	if err := NormalizeFeedback(feedback); err != nil {
		t.Fatalf("NormalizeFeedback() error = %v", err)
	}

	if feedback.WinnerModel != "model-a" || feedback.LoserModel != "model-b" {
		t.Fatalf("expected model names to be trimmed, got winner=%q loser=%q", feedback.WinnerModel, feedback.LoserModel)
	}
	if feedback.DecisionName != "coding" || feedback.UserID != "user-1" ||
		feedback.SessionID != "session-1" || feedback.ConversationID != "conversation-1" ||
		feedback.FeedbackType != "satisfied" {
		t.Fatalf("expected metadata keys to be trimmed, got %#v", feedback)
	}
}

func TestValidateFeedbackRejectsInvalidContracts(t *testing.T) {
	for _, tc := range []struct {
		name     string
		feedback *Feedback
		wantErr  error
	}{
		{
			name:     "nil feedback",
			feedback: nil,
			wantErr:  ErrFeedbackRequired,
		},
		{
			name:     "missing models",
			feedback: &Feedback{},
			wantErr:  ErrFeedbackModelRequired,
		},
		{
			name: "self comparison",
			feedback: &Feedback{
				WinnerModel: "model-a",
				LoserModel:  "model-a",
			},
			wantErr: ErrFeedbackSelfComparison,
		},
		{
			name: "self comparison after trimming",
			feedback: &Feedback{
				WinnerModel: " model-a ",
				LoserModel:  "model-a",
			},
			wantErr: ErrFeedbackSelfComparison,
		},
		{
			name: "negative confidence",
			feedback: &Feedback{
				WinnerModel: "model-a",
				Confidence:  -0.01,
			},
			wantErr: ErrFeedbackConfidenceRange,
		},
		{
			name: "nan confidence",
			feedback: &Feedback{
				WinnerModel: "model-a",
				Confidence:  math.NaN(),
			},
			wantErr: ErrFeedbackConfidenceRange,
		},
		{
			name: "infinite confidence",
			feedback: &Feedback{
				WinnerModel: "model-a",
				Confidence:  math.Inf(1),
			},
			wantErr: ErrFeedbackConfidenceRange,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateFeedback(tc.feedback)
			if !errors.Is(err, tc.wantErr) {
				t.Fatalf("expected %v, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestValidateFeedbackAllowsSingleModelFeedback(t *testing.T) {
	for _, feedback := range []*Feedback{
		{WinnerModel: "model-a"},
		{LoserModel: "model-b"},
		{WinnerModel: "model-a", Confidence: 1},
	} {
		if err := ValidateFeedback(feedback); err != nil {
			t.Fatalf("ValidateFeedback(%#v) error = %v", feedback, err)
		}
	}
}

func TestEloSelectorRejectsInvalidFeedbackBeforeMutation(t *testing.T) {
	selector := NewEloSelector(DefaultEloConfig())
	selector.setGlobalRating("model-a", &ModelRating{Model: "model-a", Rating: 1500})

	err := selector.UpdateFeedback(context.Background(), &Feedback{
		WinnerModel: "model-a",
		LoserModel:  " model-a ",
	})
	if !errors.Is(err, ErrFeedbackSelfComparison) {
		t.Fatalf("expected self-comparison error, got %v", err)
	}

	rating := selector.getGlobalRating("model-a")
	if rating == nil {
		t.Fatal("expected existing rating to remain")
	}
	if rating.Rating != 1500 || rating.Wins != 0 || rating.Losses != 0 || rating.Comparisons != 0 {
		t.Fatalf("expected invalid feedback not to mutate rating, got %#v", rating)
	}
}

func TestRLDrivenSelectorRejectsInvalidFeedbackBeforeMutation(t *testing.T) {
	selector := NewRLDrivenSelector(nil)

	err := selector.UpdateFeedback(context.Background(), &Feedback{
		WinnerModel: "model-a",
		LoserModel:  "model-a",
		Confidence:  0.8,
	})
	if !errors.Is(err, ErrFeedbackSelfComparison) {
		t.Fatalf("expected self-comparison error, got %v", err)
	}
	if pref := selector.getGlobalPreference("model-a"); pref != nil {
		t.Fatalf("expected invalid feedback not to create preference, got %#v", pref)
	}
}

func TestHybridSelectorRejectsFeedbackBeforeFanoutWhenWinnerRequired(t *testing.T) {
	selector := NewHybridSelector(DefaultHybridConfig())
	selector.eloSelector.setGlobalRating("model-a", &ModelRating{Model: "model-a", Rating: 1500})

	err := selector.UpdateFeedback(context.Background(), &Feedback{
		LoserModel: " model-a ",
		Confidence: 0.7,
	})
	if !errors.Is(err, ErrFeedbackWinnerRequired) {
		t.Fatalf("expected winner-required error, got %v", err)
	}

	rating := selector.eloSelector.getGlobalRating("model-a")
	if rating == nil {
		t.Fatal("expected existing rating to remain")
	}
	if rating.Rating != 1500 || rating.Wins != 0 || rating.Losses != 0 || rating.Comparisons != 0 {
		t.Fatalf("expected hybrid preflight to avoid partial Elo mutation, got %#v", rating)
	}
}
