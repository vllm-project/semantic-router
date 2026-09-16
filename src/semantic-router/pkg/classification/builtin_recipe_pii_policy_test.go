package classification

import (
	"context"
	"slices"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestBuiltinVaultPIICategoriesUsePersonalContext(t *testing.T) {
	for _, tt := range []struct {
		name, text, entity, want string
	}{
		{"unassociated attribute", "Explain nationalities in general.", "NRP", "private"},
		{"owned attribute", "My nationality is listed here.", "NRP", "sensitive"},
		{"third person attribute", "Her religious affiliation is listed here.", "NRP", "sensitive"},
		{"quoted owned attribute", `Translate the phrase "My nationality is listed here".`, "NRP", "sensitive"},
		{"record attribute", "Applicant profile: affiliation is listed below.", "NRP", "sensitive"},
		{"context without attribute", "My nationality is not provided.", "", "private"},
		{"quoted identifier", `Translate the phrase "contact address".`, "EMAIL_ADDRESS", "sensitive"},
		{"identifier without context", "Reformat this record.", "US_SSN", "sensitive"},
		{"bare first person is not ownership", "I am learning a new language.", "NRP", "private"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := builtinPolicyClassifier(t, "vault")
			attachBuiltinPolicyHeuristics(t, c)
			in := evaluateBuiltinPolicyHeuristics(c, tt.text)
			c.PIIMapping = &PIIMapping{IdxToLabel: map[string]string{"0": "O", "1": "B-NRP", "2": "B-EMAIL_ADDRESS", "3": "B-US_SSN"}}
			scored := true
			result := tasks.TokenClassificationResult{ScoresAvailable: &scored}
			if tt.entity != "" {
				result.Entities = []tasks.TokenEntity{{EntityType: tt.entity, Confidence: .99}}
			}
			c.piiInference = &rawPIITask{result: result}
			var mu sync.Mutex
			c.evaluatePIISignal(context.Background(), in, &mu, tt.text, nil)
			if tt.entity == "NRP" {
				if !slices.Contains(in.MatchedPIIRules, "personal_attribute") || slices.Contains(in.MatchedPIIRules, "personal_data") {
					t.Fatalf("NRP category partition = %v", in.MatchedPIIRules)
				}
			}
			assertBuiltinPolicy(t, c, in, tt.want)
		})
	}
}

func TestBuiltinVaultRetainsOtherPIICategories(t *testing.T) {
	// These are the non-NRP types in the published 35-label BIO mapping.
	for _, entity := range []string{"AGE", "CREDIT_CARD", "DATE_TIME", "DOMAIN_NAME", "EMAIL_ADDRESS", "GPE", "IBAN_CODE", "IP_ADDRESS", "ORGANIZATION", "PERSON", "PHONE_NUMBER", "STREET_ADDRESS", "TITLE", "US_DRIVER_LICENSE", "US_SSN", "ZIP_CODE"} {
		t.Run(entity, func(t *testing.T) {
			c := builtinPolicyClassifier(t, "vault")
			for _, rule := range c.Config.PIIRules {
				denied := findDeniedEntities(map[string]bool{entity: true}, rule.PIITypesAllowed)
				want := rule.Name == "personal_data"
				if (len(denied) != 0) != want || rule.Threshold != .7 || !rule.IncludeHistory {
					t.Fatalf("category %s, rule %+v: denied=%v", entity, rule, denied)
				}
			}
		})
	}
}

func TestBuiltinVaultPersonalContextWording(t *testing.T) {
	c := builtinPolicyClassifier(t, "vault")
	attachBuiltinPolicyHeuristics(t, c)
	for _, tt := range []struct {
		text string
		want bool
	}{
		{"His political views are recorded here.", true},
		{"The applicant's citizenship is listed below.", true},
		{"她的宗教信仰记录在这里。", true},
		{"客户档案：国籍和宗教。", true},
		{"Mi afiliación política aparece en este campo.", true},
		{"Die Religion der Kundin ist hier angegeben.", true},
		{"ديانة العميل مذكورة هنا.", true},
		{"申請者の国籍をこの欄に記載しています。", true},
		{"My question concerns the history of grammar.", false},
		{"I am learning about religious architecture.", false},
		{"Explain what an applicant profile means without using any individual records.", false},
		{"介绍国籍和宗教的含义，不涉及任何个人。", false},
	} {
		t.Run(tt.text, func(t *testing.T) {
			in := evaluateBuiltinPolicyHeuristics(c, tt.text)
			if got := slices.Contains(in.MatchedKeywordRules, "personal_context"); got != tt.want {
				t.Fatalf("personal_context = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBuiltinVaultAttributeHistoryUsesCurrentContext(t *testing.T) {
	for _, tt := range []struct {
		name, entity, current, want string
	}{
		{"identifier history remains sensitive", "EMAIL_ADDRESS", "Continue.", "sensitive"},
		{"attribute history alone keeps private pool", "NRP", "Continue.", "private"},
		{"attribute history with current ownership", "NRP", "Reformat my nationality field.", "sensitive"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			c := builtinPolicyClassifier(t, "vault")
			attachBuiltinPolicyHeuristics(t, c)
			in := evaluateBuiltinPolicyHeuristics(c, tt.current)
			c.PIIMapping = &PIIMapping{IdxToLabel: map[string]string{"0": "O", "1": "B-NRP", "2": "B-EMAIL_ADDRESS"}}
			model := &MockPIIInference{responseMap: make(map[string]MockPIIInferenceResponse)}
			model.setMockResponse("earlier personal field", []tasks.TokenEntity{{EntityType: tt.entity, Confidence: .99}}, nil)
			c.piiInference = model
			var mu sync.Mutex
			c.evaluatePIISignal(context.Background(), in, &mu, tt.current, []string{"earlier personal field"})
			assertBuiltinPolicy(t, c, in, tt.want)
		})
	}
}
