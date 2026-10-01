package framework

import (
	"context"
	"errors"
	"strings"
	"testing"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func countingCase(calls *int, failUntil int) testcases.TestCase {
	return testcases.TestCase{
		Name: "flaky",
		Fn: func(context.Context, *kubernetes.Clientset, testcases.TestCaseOptions) error {
			*calls++
			if *calls < failUntil {
				return errors.New("transient failure")
			}
			return nil
		},
	}
}

func TestRunSingleTestRetriesUntilTheCasePasses(t *testing.T) {
	runner := &Runner{opts: &TestOptions{FlakeAttempts: 3}, profile: &stubProfile{}}

	calls := 0
	result := runner.runSingleTest(context.Background(), nil, countingCase(&calls, 3))

	if calls != 3 {
		t.Fatalf("test ran %d times, want 3", calls)
	}
	if !result.Passed {
		t.Fatalf("result should pass once a retry succeeds, got error %v", result.Error)
	}
	if !result.Flaked {
		t.Error("a case that passed only on a retry must be reported as flaky")
	}
	if result.Attempts != 3 {
		t.Errorf("Attempts = %d, want 3", result.Attempts)
	}
}

func TestRunSingleTestDoesNotMarkAFirstAttemptPassAsFlaky(t *testing.T) {
	runner := &Runner{opts: &TestOptions{FlakeAttempts: 3}, profile: &stubProfile{}}

	calls := 0
	result := runner.runSingleTest(context.Background(), nil, countingCase(&calls, 1))

	if calls != 1 {
		t.Fatalf("test ran %d times, want 1", calls)
	}
	if !result.Passed || result.Flaked {
		t.Errorf("a first-attempt pass must not be flaky: passed=%v flaked=%v", result.Passed, result.Flaked)
	}
	if result.Attempts != 1 {
		t.Errorf("Attempts = %d, want 1", result.Attempts)
	}
}

func TestRunSingleTestReportsFailureAfterEveryAttempt(t *testing.T) {
	runner := &Runner{opts: &TestOptions{FlakeAttempts: 2}, profile: &stubProfile{}}

	calls := 0
	result := runner.runSingleTest(context.Background(), nil, countingCase(&calls, 99))

	if calls != 2 {
		t.Fatalf("test ran %d times, want 2", calls)
	}
	if result.Passed {
		t.Error("a case that never passed must be reported as failed")
	}
	if result.Flaked {
		t.Error("a case that never passed is not flaky")
	}
	if result.Error == nil {
		t.Error("the failure must be retained")
	}
}

func TestRetriesStayDisabledWhenFlakeAttemptsIsUnset(t *testing.T) {
	for _, attempts := range []int{0, -1} {
		runner := &Runner{opts: &TestOptions{FlakeAttempts: attempts}, profile: &stubProfile{}}

		calls := 0
		result := runner.runSingleTest(context.Background(), nil, countingCase(&calls, 99))

		if calls != 1 {
			t.Errorf("FlakeAttempts=%d ran the test %d times, want 1", attempts, calls)
		}
		if result.Attempts != 1 {
			t.Errorf("FlakeAttempts=%d gave Attempts=%d, want 1", attempts, result.Attempts)
		}
	}
}

func TestCasesThatChangeClusterStateOptOutOfRetries(t *testing.T) {
	runner := &Runner{opts: &TestOptions{FlakeAttempts: 3}, profile: &stubProfile{}}

	calls := 0
	testCase := countingCase(&calls, 99)
	testCase.MutatesClusterState = true
	result := runner.runSingleTest(context.Background(), nil, testCase)

	if calls != 1 {
		t.Fatalf("a cluster-mutating case ran %d times, want 1", calls)
	}
	if result.Attempts != 1 {
		t.Errorf("Attempts = %d, want 1", result.Attempts)
	}
	if result.Passed || result.Flaked {
		t.Errorf("the failure must be reported as-is: passed=%v flaked=%v", result.Passed, result.Flaked)
	}
}

func TestMutatingCasesAreMarkedInTheRegistry(t *testing.T) {
	want := []string{
		"dashboard-restart-recovery",
		"failover-during-traffic",
		"response-api-restart-recovery",
		"router-replay-restart-recovery",
		"vectorstore-registry-restart-recovery",
		"workflow-resume-restart-recovery",
	}

	for _, name := range want {
		testCase, ok := testcases.Get(name)
		if !ok {
			t.Errorf("test case %q is not registered", name)
			continue
		}
		if !testCase.MutatesClusterState {
			t.Errorf("test case %q deletes a pod but is not marked as mutating cluster state", name)
		}
	}
}

func TestFlakyTestsAreCountedAndNamed(t *testing.T) {
	rg := &ReportGenerator{report: &TestReport{}}

	rg.AddTestResults([]TestResult{
		{Name: "steady", Passed: true, Attempts: 1},
		{Name: "wobbly", Passed: true, Attempts: 2, Flaked: true},
		{Name: "broken", Passed: false, Attempts: 2},
	})

	if rg.report.FlakyTests != 1 {
		t.Errorf("FlakyTests = %d, want 1", rg.report.FlakyTests)
	}
	if rg.report.PassedTests != 2 || rg.report.FailedTests != 1 {
		t.Errorf("passed=%d failed=%d, want 2 and 1", rg.report.PassedTests, rg.report.FailedTests)
	}

	names := flakyTestNames(rg.report.TestResults)
	if len(names) != 1 || names[0] != "wobbly" {
		t.Errorf("flakyTestNames = %v, want [wobbly]", names)
	}
}

func TestMarkdownReportListsTheRetriedCaseAsFlaky(t *testing.T) {
	rg := &ReportGenerator{report: &TestReport{Profile: "envoy-ai-gateway"}}
	rg.AddTestResults([]TestResult{
		{Name: "steady", Passed: true, Attempts: 1, Duration: "1s"},
		{Name: "wobbly", Passed: true, Attempts: 2, Flaked: true, Duration: "3s"},
	})

	markdown := rg.generateMarkdown()

	for _, want := range []string{
		"| Flaky Tests | 1 |",
		"| Failed Tests | 0 |",
		"> ⚠️ **1 test(s) passed only after a retry and are flaky:** wobbly",
		"- ⚠️ **wobbly** (3s) - passed on attempt 2",
		"- ✅ **steady** (1s)",
	} {
		if !strings.Contains(markdown, want) {
			t.Errorf("markdown report is missing %q:\n%s", want, markdown)
		}
	}
	if strings.Contains(markdown, "- ⚠️ **steady**") {
		t.Errorf("a first-attempt pass must not be marked flaky:\n%s", markdown)
	}
}
