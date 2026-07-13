package mlmodelselection

import (
	"testing"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func TestProfileRegistersSelectedModelHeaderSecurityContract(t *testing.T) {
	const testcaseName = "selected-model-header-security"
	registered, err := pkgtestcases.ListByNames(testcaseName)
	if err != nil || len(registered) != 1 || registered[0].Fn == nil {
		t.Fatalf("registered selected-model testcase = %#v, err = %v", registered, err)
	}

	for _, name := range NewProfile().GetTestCases() {
		if name == testcaseName {
			return
		}
	}
	t.Fatalf("ML profile does not include registered testcase %q", testcaseName)
}
