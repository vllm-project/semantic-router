package framework

import (
	"context"
	"strings"
	"testing"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

type scopedProfile struct{ stubProfile }

func (p *scopedProfile) GetTestCases() []string {
	return []string{"scope-test-p1", "scope-test-p2"}
}

func init() {
	noop := func(context.Context, *kubernetes.Clientset, testcases.TestCaseOptions) error { return nil }
	testcases.Register("scope-test-p1", testcases.TestCase{Fn: noop})
	testcases.Register("scope-test-p2", testcases.TestCase{Priority: testcases.P2, Fn: noop})
}

func TestSelectTestCasesByScope(t *testing.T) {
	for scope, want := range map[string]string{"standard": "scope-test-p1", "full": "scope-test-p1,scope-test-p2"} {
		runner := &Runner{opts: &TestOptions{BaselineSuite: scope}, profile: &scopedProfile{}}
		cases, err := runner.selectTestCases()
		if err != nil {
			t.Fatal(err)
		}
		var names []string
		for _, tc := range cases {
			names = append(names, tc.Name)
		}
		if got := strings.Join(names, ","); got != want {
			t.Fatalf("scope %s selected %s, want %s", scope, got, want)
		}
	}
}

// A scope that leaves the profile nothing to run fails from selection, which
// Run calls before the cluster is built.
func TestSelectTestCasesRefusesAnEmptyScope(t *testing.T) {
	runner := &Runner{opts: &TestOptions{BaselineSuite: "p0"}, profile: &scopedProfile{}}
	if _, err := runner.selectTestCases(); err == nil || !strings.Contains(err.Error(), `no case in scope "p0"`) {
		t.Fatalf("expected an empty p0 scope to be refused, got %v", err)
	}
}
