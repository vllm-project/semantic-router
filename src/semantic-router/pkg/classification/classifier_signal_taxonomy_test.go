package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestKBSignalMatchConfidenceGroupBestUsesBestGroup(t *testing.T) {
	result := &KBClassifyResult{
		BestLabel:             "generic_coding",
		BestSimilarity:        0.99,
		BestMatchedLabel:      "generic_coding",
		BestMatchedSimilarity: 0.99,
		BestGroup:             "local_standard",
		BestMatchedGroup:      "local_standard",
		MatchedLabels:         []string{"prompt_injection", "generic_coding"},
		MatchedGroups:         []string{"security_containment", "local_standard"},
		LabelConfidences: map[string]float64{
			"prompt_injection": 0.80,
			"generic_coding":   0.99,
		},
		GroupScores: map[string]float64{
			"security_containment": 0.80,
			"local_standard":       0.99,
		},
	}

	securityRule := config.KBSignalRule{
		Name: "security_containment",
		KB:   "privacy_kb",
		Target: config.KBSignalTarget{
			Kind:  config.KBTargetKindGroup,
			Value: "security_containment",
		},
		Match: config.KBMatchBest,
	}
	if confidence, matched := kbSignalMatchConfidence(securityRule, result); matched || confidence != 0 {
		t.Fatalf("security group should not match when best group is local_standard, got matched=%v confidence=%.2f", matched, confidence)
	}

	localRule := config.KBSignalRule{
		Name: "local_standard",
		KB:   "privacy_kb",
		Target: config.KBSignalTarget{
			Kind:  config.KBTargetKindGroup,
			Value: "local_standard",
		},
		Match: config.KBMatchBest,
	}
	confidence, matched := kbSignalMatchConfidence(localRule, result)
	if !matched {
		t.Fatal("local_standard group should match when it is the best group")
	}
	if confidence != 0.99 {
		t.Fatalf("local_standard confidence = %.2f, want 0.99", confidence)
	}
}

func TestKBSignalMatchConfidenceThresholdRequiresMatchedTarget(t *testing.T) {
	result := &KBClassifyResult{
		BestLabel:             "generic_coding",
		BestSimilarity:        0.42,
		BestMatchedLabel:      "",
		BestMatchedSimilarity: 0,
		BestGroup:             "local_standard",
		MatchedLabels:         nil,
		MatchedGroups:         nil,
		LabelConfidences:      map[string]float64{"generic_coding": 0.42},
		GroupScores:           map[string]float64{"local_standard": 0.42},
	}

	labelRule := config.KBSignalRule{
		Name: "generic_coding",
		KB:   "privacy_kb",
		Target: config.KBSignalTarget{
			Kind:  config.KBTargetKindLabel,
			Value: "generic_coding",
		},
		Match: config.KBMatchThreshold,
	}
	if confidence, matched := kbSignalMatchConfidence(labelRule, result); matched || confidence != 0 {
		t.Fatalf("threshold label match should require a matched label, got matched=%v confidence=%.2f", matched, confidence)
	}
}
