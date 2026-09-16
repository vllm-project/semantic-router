package classification

import "testing"

func TestSafetyIndependentHazards(t *testing.T) {
	mapping := newDeclaredLabelMapping([]string{"privacy", "fraud", "other"})
	scores := []httpClassifyLabelScore{{Label: "fraud", Score: 0.8}, {Label: "other", Score: 0.05}, {Label: "privacy", Score: 0.9}}
	if _, err := alignScoresToMapping(mapping, scores); err == nil {
		t.Fatal("categorical adapter accepted independent sigmoid scores")
	}
	result, err := alignIndependentLabelScores(mapping, scores)
	if err != nil {
		t.Fatal(err)
	}
	if result[0] != 0.9 || result[1] != 0.8 {
		t.Fatalf("label scores changed: %v", result)
	}
	if _, err := alignIndependentLabelScores(mapping, scores[:2]); err == nil {
		t.Fatal("missing labels treated as zero")
	}
	if got := selectedHazardScore(map[string]float64{"privacy": 0.4, "fraud": 0.4}, []string{"privacy", "fraud"}); got != 0.4 {
		t.Fatalf("independent probabilities were added: %v", got)
	}
}
