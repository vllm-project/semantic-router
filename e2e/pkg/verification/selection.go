package verification

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"
)

// SuiteExclusion records one testcase a profile selects at runtime that a
// baseline suite omits from execution, with the production reason for the
// omission. Reason is empty when no production table explains it: the
// unrecorded-exclusion scenario Gate E fails on.
type SuiteExclusion struct {
	Name   string `json:"name"`
	Reason string `json:"reason"`
}

// SuiteSelection is the effective selection of one profile under one
// baseline suite, computed by the production selector
// (framework.EffectiveTestCases) rather than by a second implementation.
type SuiteSelection struct {
	Suite     string           `json:"suite"`
	Source    string           `json:"source"`
	TestCases []string         `json:"test_cases"`
	Excluded  []SuiteExclusion `json:"excluded"`
}

// ProfileSelection records every suite's effective selection for one
// registered profile.
type ProfileSelection struct {
	Profile string           `json:"profile"`
	Suites  []SuiteSelection `json:"suites"`
}

// EffectiveSelection is the runtime-derived view of the baseline-suite
// selection layer: which profile the layer narrows, how CI dispatches the
// suite, and what each profile actually executes under each suite.
type EffectiveSelection struct {
	BaselineProfile string             `json:"baseline_profile"`
	SuiteEnv        string             `json:"suite_env"`
	DefaultSuite    string             `json:"default_suite"`
	Suites          []string           `json:"suites"`
	Profiles        []ProfileSelection `json:"profiles"`
}

// StressExclusionReason is the recorded reason for a standard-suite
// omission that testmatrix.BaselineStress explains.
const StressExclusionReason = "stress: testmatrix.BaselineStress, executed only by the full baseline suite"

// DeriveEffectiveSelection runs the production selector over every
// registered profile and every accepted baseline suite. Each profile's
// GetTestCases selection comes from the inventory; the effective selection
// and its source come from framework.EffectiveTestCases, so this view can
// only drift from CI execution if the runner stops calling that function.
func DeriveEffectiveSelection(inventory Inventory) (EffectiveSelection, error) {
	selection := EffectiveSelection{
		BaselineProfile: framework.BaselineSuiteProfile,
		SuiteEnv:        framework.BaselineSuiteEnv,
		DefaultSuite:    framework.DefaultBaselineSuite,
		Suites:          append([]string(nil), testmatrix.BaselineSuites...),
		Profiles:        make([]ProfileSelection, 0, len(inventory.Profiles)),
	}
	stress := toSet(testmatrix.BaselineStress)

	for _, profile := range inventory.Profiles {
		entry := ProfileSelection{Profile: profile.Name, Suites: make([]SuiteSelection, 0, len(selection.Suites))}
		for _, suite := range selection.Suites {
			opts := framework.TestOptions{Profile: profile.Name, BaselineSuite: suite}
			effective, source, err := framework.EffectiveTestCases(profile.Name, profile.TestCases, &opts)
			if err != nil {
				return EffectiveSelection{}, fmt.Errorf("profile %q suite %q: %w", profile.Name, suite, err)
			}
			effective = dedupSorted(effective)
			effectiveSet := toSet(effective)
			excluded := make([]SuiteExclusion, 0)
			for _, name := range profile.TestCases {
				if effectiveSet[name] {
					continue
				}
				reason := ""
				if stress[name] {
					reason = StressExclusionReason
				}
				excluded = append(excluded, SuiteExclusion{Name: name, Reason: reason})
			}
			entry.Suites = append(entry.Suites, SuiteSelection{
				Suite:     suite,
				Source:    string(source),
				TestCases: effective,
				Excluded:  excluded,
			})
		}
		selection.Profiles = append(selection.Profiles, entry)
	}
	return selection, nil
}

// SuiteFinding locates one drift finding at a suite.
type SuiteFinding struct {
	Suite string `json:"suite"`
	Name  string `json:"name"`
}

// SuiteDrift is the diagnosis of one profile's suite-layer selection against
// its runtime GetTestCases selection and the production exclusion table.
type SuiteDrift struct {
	// OmittedFromAllSuites lists testcases the profile selects that no
	// baseline suite executes: declared coverage that CI never runs. This is
	// the suite-layer form of the silent-omission scenario and is not a
	// legitimate exclusion under any suite.
	OmittedFromAllSuites []string `json:"omitted_from_all_suites"`

	// UnrecordedExclusions lists per-suite omissions that no production
	// table explains (the only recorded reason today is BaselineStress).
	UnrecordedExclusions []SuiteFinding `json:"unrecorded_exclusions"`

	// Phantom lists per-suite effective cases the profile does not select
	// at runtime: the suite layer executing outside the profile contract.
	Phantom []SuiteFinding `json:"phantom"`

	// StaleStress lists BaselineStress entries the profile does not select,
	// so the stress exclusion they claim never applies to a real case.
	StaleStress []string `json:"stale_stress"`
}

// DiagnoseSuiteDrift is a pure function over its inputs so the omission,
// unrecorded, phantom, and stale scenarios can be pinned by synthetic tests
// independently of the live registries. effectiveBySuite maps each suite to
// the cases the production selector returns for it; stress is the exclusion
// table the standard suite is allowed to apply.
func DiagnoseSuiteDrift(selected []string, effectiveBySuite map[string][]string, stress []string) SuiteDrift {
	selectedSet := toSet(selected)
	stressSet := toSet(stress)
	suites := make([]string, 0, len(effectiveBySuite))
	for suite := range effectiveBySuite {
		suites = append(suites, suite)
	}
	sort.Strings(suites)

	drift := SuiteDrift{
		OmittedFromAllSuites: []string{},
		UnrecordedExclusions: []SuiteFinding{},
		Phantom:              []SuiteFinding{},
		StaleStress:          []string{},
	}

	executedSomewhere := make(map[string]bool, len(selected))
	for _, suite := range suites {
		effectiveSet := toSet(effectiveBySuite[suite])
		for name := range effectiveSet {
			executedSomewhere[name] = true
			if !selectedSet[name] {
				drift.Phantom = append(drift.Phantom, SuiteFinding{Suite: suite, Name: name})
			}
		}
		for name := range selectedSet {
			if !effectiveSet[name] && !stressSet[name] {
				drift.UnrecordedExclusions = append(drift.UnrecordedExclusions, SuiteFinding{Suite: suite, Name: name})
			}
		}
	}
	for name := range selectedSet {
		if !executedSomewhere[name] {
			drift.OmittedFromAllSuites = append(drift.OmittedFromAllSuites, name)
		}
	}
	for name := range stressSet {
		if !selectedSet[name] {
			drift.StaleStress = append(drift.StaleStress, name)
		}
	}

	sort.Strings(drift.OmittedFromAllSuites)
	sort.Strings(drift.StaleStress)
	sortFindings(drift.UnrecordedExclusions)
	sortFindings(drift.Phantom)
	return drift
}

func sortFindings(findings []SuiteFinding) {
	sort.Slice(findings, func(i, j int) bool {
		if findings[i].Suite != findings[j].Suite {
			return findings[i].Suite < findings[j].Suite
		}
		return findings[i].Name < findings[j].Name
	})
}

// CISuiteDispatch is the baseline-suite contract the integration workflow
// declares: the workflow_call input default and the job environment
// variable that carries the input into the e2e binary.
type CISuiteDispatch struct {
	InputDefault string `json:"input_default"`
	EnvName      string `json:"env_name"`
	EnvValue     string `json:"env_value"`
}

// LoadCISuiteDispatch parses the integration workflow at path and returns
// its baseline-suite dispatch. A workflow without the input, or a job
// without an environment variable bound to it, is a hard error: the suite
// layer would then have no CI path at all.
func LoadCISuiteDispatch(path string) (CISuiteDispatch, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return CISuiteDispatch{}, fmt.Errorf("reading CI workflow %s: %w", path, err)
	}
	var doc struct {
		On struct {
			WorkflowCall struct {
				Inputs map[string]struct {
					Default string `yaml:"default"`
				} `yaml:"inputs"`
			} `yaml:"workflow_call"`
		} `yaml:"on"`
		Jobs map[string]struct {
			Env map[string]string `yaml:"env"`
		} `yaml:"jobs"`
	}
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		return CISuiteDispatch{}, fmt.Errorf("parsing CI workflow %s: %w", path, err)
	}

	input, ok := doc.On.WorkflowCall.Inputs["baseline-suite"]
	if !ok {
		return CISuiteDispatch{}, fmt.Errorf("CI workflow %s declares no baseline-suite workflow_call input", path)
	}
	dispatch := CISuiteDispatch{InputDefault: input.Default}

	jobNames := make([]string, 0, len(doc.Jobs))
	for name := range doc.Jobs {
		jobNames = append(jobNames, name)
	}
	sort.Strings(jobNames)
	for _, name := range jobNames {
		for envName, envValue := range doc.Jobs[name].Env {
			if !strings.Contains(envValue, "inputs.baseline-suite") {
				continue
			}
			if dispatch.EnvName != "" && dispatch.EnvName != envName {
				return CISuiteDispatch{}, fmt.Errorf("CI workflow %s binds inputs.baseline-suite to both %s and %s", path, dispatch.EnvName, envName)
			}
			dispatch.EnvName = envName
			dispatch.EnvValue = envValue
		}
	}
	if dispatch.EnvName == "" {
		return CISuiteDispatch{}, fmt.Errorf("CI workflow %s binds inputs.baseline-suite to no job environment variable", path)
	}
	return dispatch, nil
}
