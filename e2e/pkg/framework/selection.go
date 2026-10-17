package framework

import "github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

// BaselineSuiteProfile is the one registered profile whose runtime
// GetTestCases selection the runner narrows by baseline suite. Every other
// profile executes its GetTestCases selection unchanged.
const BaselineSuiteProfile = "envoy-ai-gateway"

// BaselineSuiteEnv is the environment variable CI uses to dispatch the
// baseline suite into the e2e binary (integration-test-k8s.yml sets it from
// the baseline-suite workflow input; cmd/e2e reads it as the flag default).
const BaselineSuiteEnv = "E2E_BASELINE_SUITE"

// DefaultBaselineSuite is the suite the runner uses when neither the flag
// nor BaselineSuiteEnv names one.
const DefaultBaselineSuite = "standard"

// SelectionSource names the layer that decided the effective testcase
// selection for one runner invocation.
type SelectionSource string

const (
	// SelectionExplicit: the caller passed -tests, which bypasses both the
	// profile selection and the baseline suite layer.
	SelectionExplicit SelectionSource = "explicit-tests"

	// SelectionBaselineSuite: the profile is BaselineSuiteProfile and the
	// baseline suite narrowed its selection via testmatrix.BaselineCases.
	SelectionBaselineSuite SelectionSource = "baseline-suite"

	// SelectionProfile: the profile's GetTestCases selection ran as is.
	SelectionProfile SelectionSource = "profile"
)

// EffectiveTestCases resolves the testcase names the runner executes for
// profileName, given the profile's runtime GetTestCases selection and the
// run options. It is the single production selection path: runTests calls
// it before resolving names against the registry, and the verification
// package derives its audit view of effective selection from the same
// function so no second selector can drift from what CI actually runs.
//
// The returned slice is always a fresh copy; the returned SelectionSource
// records which layer decided the result. An unknown baseline suite is an
// error only for BaselineSuiteProfile, mirroring the runner behavior that
// other profiles ignore the suite entirely.
func EffectiveTestCases(profileName string, profileSelection []string, opts *TestOptions) ([]string, SelectionSource, error) {
	if len(opts.TestCases) > 0 {
		return append([]string(nil), opts.TestCases...), SelectionExplicit, nil
	}
	if profileName == BaselineSuiteProfile {
		cases, err := testmatrix.BaselineCases(opts.BaselineSuite)
		if err != nil {
			return nil, "", err
		}
		return cases, SelectionBaselineSuite, nil
	}
	return append([]string(nil), profileSelection...), SelectionProfile, nil
}
