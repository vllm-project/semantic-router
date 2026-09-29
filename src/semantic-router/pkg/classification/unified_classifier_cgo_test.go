package classification

import "testing"

func TestUnifiedClassifierRejectsAggregateBatchResult(t *testing.T) {
	if err := validateUnifiedBatchResults([]string{"one", "two"}, &UnifiedBatchResults{BatchSize: 2, IntentResults: make([]IntentResult, 1), PIIResults: make([]PIIResult, 1), SecurityResults: make([]SecurityResult, 1)}); err == nil {
		t.Fatal("accepted one aggregate prediction for multiple independent inputs")
	}
}
