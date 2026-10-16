package classification

import (
	"slices"
	"testing"
)

// Run the authored wording through the actual keyword, structure, projection,
// and decision path. No learned classifier result is supplied by these cases.
func TestBuiltinCorrectionGrammarAndScope(t *testing.T) {
	for _, profile := range []struct{ name, fallback string }{
		{"balance", "medium"}, {"cost", "economy"}, {"accuracy", "simple"},
	} {
		t.Run(profile.name, func(t *testing.T) {
			c := builtinPolicyClassifier(t, profile.name)
			attachBuiltinPolicyHeuristics(t, c)
			for _, tt := range []struct {
				name, text string
				repair     bool
				history    bool
			}{
				{name: "recommendation", text: "The recommendation contains an error. Revise it.", repair: true},
				{name: "reversed assessment", text: "There is an incorrect assessment here. Fix this.", repair: true},
				{name: "explanation", text: "Your explanation is inconsistent. Recheck it.", repair: true},
				{name: "repair after answer", text: "Correct your recommendation.", history: true, repair: true},
				{name: "missing error and history", text: "Correct your recommendation."},
				{name: "negated English repair", text: "The assessment is wrong. Do not revise it."},
				{name: "English quotation", text: "Translate the phrase: The explanation is wrong. Fix it."},
				{name: "English style only", text: "Revise the explanation into bullet points."},
				{name: "Korean contents object", text: "답이 잘못됐습니다. 내용을 수정하세요.", repair: true},
				{name: "Korean explanation object", text: "결과가 틀렸습니다. 설명을 정정하세요.", repair: true},
				{name: "Korean negation", text: "결과가 틀렸습니다. 내용을 수정하지 마세요."},
				{name: "Korean quotation", text: "번역할 표현: 답변이 틀렸습니다. 내용을 수정하세요."},
				{name: "Korean style only", text: "설명을 수정해서 더 짧게 쓰세요."},
				{name: "Arabic semicolon", text: "النتيجة خاطئة؛ صحح النتيجة.", repair: true},
				{name: "Arabic comma", text: "الحساب غير صحيح، أعد الحساب.", repair: true},
				{name: "Arabic question mark", text: "هل الإجابة خاطئة؟ راجع الإجابة.", repair: true},
				{name: "Arabic repair after answer", text: "من فضلك؛ صحح الإجابة.", history: true, repair: true},
				{name: "Arabic negation", text: "النتيجة خاطئة؛ لا تصحح الإجابة."},
				{name: "Arabic quotation", text: "ترجم العبارة: النتيجة خاطئة؛ صحح النتيجة."},
				{name: "Arabic style only", text: "اجعل الإجابة أقصر، عدّل الإجابة إلى نقاط."},
			} {
				t.Run(tt.name, func(t *testing.T) {
					in := evaluateBuiltinPolicyHeuristics(c, tt.text)
					if tt.history {
						in.MatchedConversationRules = []string{"has_answer"}
					}
					want := profile.fallback
					if tt.repair {
						want = "reasoning"
						if tt.history {
							if !slices.Contains(in.MatchedKeywordRules, "answer_repair") {
								t.Fatal("direct correction did not match answer_repair")
							}
						} else if !slices.Contains(in.MatchedKeywordRules, "answer_error") || !slices.Contains(in.MatchedKeywordRules, "answer_revision") {
							t.Fatalf("self-contained correction lacks independent error/revision evidence: %v", in.MatchedKeywordRules)
						}
					}
					assertBuiltinPolicy(t, c, in, want)
				})
			}
		})
	}
}
