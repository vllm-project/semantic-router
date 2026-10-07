package classification

import "testing"

func TestServedLabelsBuildMappingsInOutputOrder(t *testing.T) {
	category, err := categoryMappingFromLabels([]string{"math", "law", "other"})
	if err != nil || category.CategoryToIdx["law"] != 1 || category.IdxToCategory["2"] != "other" {
		t.Fatalf("category mapping = %+v, %v", category, err)
	}
	pii, err := piiMappingFromLabels([]string{"O", "B-PERSON", "I-PERSON"})
	if err != nil || pii.LabelToIdx["B-PERSON"] != 1 || pii.IdxToLabel["2"] != "I-PERSON" {
		t.Fatalf("PII mapping = %+v, %v", pii, err)
	}
	jailbreak, err := jailbreakMappingFromLabels([]string{"benign", "jailbreak"})
	if err != nil || jailbreak.LabelToIdx["jailbreak"] != 1 || jailbreak.IdxToLabel["0"] != "benign" {
		t.Fatalf("jailbreak mapping = %+v, %v", jailbreak, err)
	}
}

func TestServedLabelsRejectVocabulariesTheConsumersCannotUse(t *testing.T) {
	for name, build := range map[string]func() error{
		"repeated category": func() error { _, err := categoryMappingFromLabels([]string{"math", "math"}); return err },
		"repeated PII":      func() error { _, err := piiMappingFromLabels([]string{"O", "O"}); return err },
		"PII sentinel":      func() error { _, err := piiMappingFromLabels([]string{"O", PIIClassificationErrorType}); return err },
		"jailbreak sentinel": func() error {
			_, err := jailbreakMappingFromLabels([]string{"benign", JailbreakClassificationErrorType})
			return err
		},
	} {
		if build() == nil {
			t.Fatalf("%s: accepted", name)
		}
	}
}
