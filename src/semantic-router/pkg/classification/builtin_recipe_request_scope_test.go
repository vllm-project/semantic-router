package classification

import "testing"

func TestBuiltinWorkflowRequiresSeveralExecutors(t *testing.T) {
	c := builtinPolicyClassifier(t, "accuracy")
	attachBuiltinPolicyHeuristics(t, c)
	for _, row := range []struct{ name, text, want string }{
		{"zh distributed units", "请编排工作流：先让独立工作单元分别建立索引和检查重复条目。", "agent"},
		{"zh one person same tasks", "请编排工作流：单人分阶段建立索引和检查重复条目。", "simple"},
		{"zh distribution limited to one", "请编排工作流：先让独立工作单元分别建立索引和检查重复条目。但只用一个执行者分阶段做完。", "simple"},
		{"zh prohibited distribution", "请编排工作流，但不要让独立工作单元分别建立索引和检查重复条目。", "simple"},
		{"ja prohibited distribution", "ワークフローを構成してください。ただし独立した作業担当に索引作成と重複点検を分担させないでください。", "simple"},
		{"zh quotation", "翻译：请编排工作流：先让独立工作单元分别建立索引和检查重复条目。", "simple"},
		{"zh strong ambiguous", "请协调独立的执行者检查目录。", "simple"},
		{"zh delegation ambiguous", "请委派独立的执行者检查目录。", "simple"},
		{"ja distributed units", "ワークフローを構成してください。独立した作業担当に索引作成と重複点検を分担させてください。", "agent"},
		{"ja one person same tasks", "ワークフローを構成してください。一人で索引作成と重複点検を順番に処理してください。", "simple"},
		{"ja distribution limited to one", "ワークフローを構成してください。独立した作業担当に索引作成と重複点検を分担させてください。ただし一人の担当者だけを使ってください。", "simple"},
		{"ja quotation", "翻訳してください：ワークフローを構成してください。独立した作業担当に索引作成と重複点検を分担させてください。", "simple"},
		{"ja strong ambiguous", "独立した担当者を連携させてください。", "simple"},
		{"ja delegation ambiguous", "独立した担当者に点検を任せてください。", "simple"},
		{"en no added workers", "Run a workflow with separate workers. Use one worker only; do not add other workers.", "simple"},
		{"en no added agents", "Organize a workflow with two independent agents. Use one agent only; do not create additional agents.", "simple"},
		{"zh no added units", "请编排工作流，让独立工作单元分别建立索引和检查重复条目。不要增加其他工作单元。", "simple"},
	} {
		t.Run(row.name, func(t *testing.T) {
			in := evaluateBuiltinPolicyHeuristics(c, row.text)
			in.SignalValues["embedding:workflow_intent"] = .8
			in.SignalValues["embedding:informational"] = .4
			assertBuiltinPolicy(t, c, in, row.want)
		})
	}
}

func TestBuiltinDirectRepairRequiresRequestScope(t *testing.T) {
	for _, recipe := range []struct{ name, fallback string }{{"balance", "medium"}, {"cost", "economy"}, {"accuracy", "simple"}} {
		c := builtinPolicyClassifier(t, recipe.name)
		attachBuiltinPolicyHeuristics(t, c)
		for _, row := range []struct {
			name, text      string
			history, repair bool
		}{
			{"prior answer", "Please correct the result.", true, true},
			{"no previous answer", "Please correct the result.", false, false},
			{"self contained error", "The result is incorrect. Please correct the result.", false, true},
			{"quoted", "Translate: Please correct the result.", true, false},
			{"negative", "Do not correct the result.", true, false},
			{"spelling only", "Please fix the spelling.", true, false},
		} {
			t.Run(recipe.name+"/"+row.name, func(t *testing.T) {
				in := evaluateBuiltinPolicyHeuristics(c, row.text)
				if row.history {
					in.MatchedConversationRules = []string{"has_answer"}
				}
				want := recipe.fallback
				if row.repair {
					want = "reasoning"
				}
				assertBuiltinPolicy(t, c, in, want)
			})
		}
	}
}

func TestBuiltinPersonalActionNeedsRelevantDomain(t *testing.T) {
	for _, recipe := range []struct{ name, fallback string }{{"balance", "medium"}, {"accuracy", "simple"}} {
		c := builtinPolicyClassifier(t, recipe.name)
		attachBuiltinPolicyHeuristics(t, c)
		for _, row := range []struct {
			name, text, domain string
			care               bool
		}{
			{"personal action", "Should I stop taking the prescribed medicine?", "health", true},
			{"ordinary definition", "Define the word medicine.", "health", false},
			{"unrelated subject", "What should I do to align the heading?", "computer science", false},
			{"no corroborating domain", "Should I stop taking the prescribed medicine?", "", false},
			{"quoted action", "Translate: Should I stop taking the prescribed medicine?", "health", false},
		} {
			t.Run(recipe.name+"/"+row.name, func(t *testing.T) {
				in := evaluateBuiltinPolicyHeuristics(c, row.text)
				if row.domain != "" {
					in.MatchedDomainRules = []string{row.domain}
				}
				want := recipe.fallback
				if row.care {
					want = "reasoning"
				}
				assertBuiltinPolicy(t, c, in, want)
			})
		}
	}
}
