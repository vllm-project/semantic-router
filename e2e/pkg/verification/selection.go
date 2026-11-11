package verification

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"

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

// ciSuiteDispatchProbe drives the production CI path that dispatches a
// baseline suite: tools/ci/ci_plan.py plans every e2e record twice (the
// standard plan and the full plan), and tools/ci/run_e2e_batch.py builds the
// environment and command it would hand bin/e2e for each record. A record
// without baseline_suite exercises the runner's own fallback. The env and
// flag names come from argv so Go stays their single source.
const ciSuiteDispatchProbe = `
import json, sys, tempfile
from pathlib import Path

root, env_name, flag = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
sys.path.insert(0, str(root / "tools" / "ci"))
import ci_plan
import run_e2e_batch as runner

sha = "0" * 40


def e2e(plan):
    return [r for r in plan["verifications"] if r["executor"] == "e2e"]


full = e2e(ci_plan.make_plan([], source_sha=sha, full=True))
standard = e2e(ci_plan.make_plan([], source_sha=sha, requested=tuple(sorted(r["id"] for r in full))))
fallback = [{k: v for k, v in r.items() if k != "baseline_suite"} for r in full]


def observe(plan, record):
    with tempfile.TemporaryDirectory() as tmp:
        env = runner.profile_environment(record, Path(tmp) / "raw", Path(tmp) / "state", {})
    command = runner.profile_command(record, "e2e-audit")
    return {
        "plan": plan,
        "profile": record["profile"],
        "planned_suite": record.get("baseline_suite", ""),
        "env_suite": env.get(env_name, ""),
        "flag_suites": [a[len(flag):] for a in command if a.startswith(flag)],
    }


rows = [observe(name, r) for name, records in (("standard", standard), ("full", full), ("runner-fallback", fallback)) for r in records]
rows.sort(key=lambda row: (row["plan"], row["profile"]))
json.dump(rows, sys.stdout)
`

// BaselineSuiteFlag is the cmd/e2e flag the batch runner passes the
// planned suite through, alongside framework.BaselineSuiteEnv.
const BaselineSuiteFlag = "-baseline-suite="

// CISuiteDispatch is one baseline-suite dispatch on the CI path: the suite
// tools/ci/ci_plan.py planned for an e2e record (empty for the runner
// fallback rows, where the record carries none) and what
// tools/ci/run_e2e_batch.py hands bin/e2e for it through
// framework.BaselineSuiteEnv and BaselineSuiteFlag.
type CISuiteDispatch struct {
	Plan         string   `json:"plan"`
	Profile      string   `json:"profile"`
	PlannedSuite string   `json:"planned_suite"`
	EnvSuite     string   `json:"env_suite"`
	FlagSuites   []string `json:"flag_suites"`
}

// LoadCISuiteDispatch runs the planner and batch runner from the repository
// at repoRoot and returns every dispatch they produce. It prefers the agent
// harness interpreter (.venv-agent) and falls back to python3. Failing to
// run the probe, or a probe that plans no e2e record, is a hard error: the
// suite layer would then have no observed CI path at all.
func LoadCISuiteDispatch(repoRoot string) ([]CISuiteDispatch, error) {
	python := filepath.Join(repoRoot, ".venv-agent", "bin", "python")
	if _, err := os.Stat(python); err != nil {
		python = "python3"
	}
	cmd := exec.Command(python, "-c", ciSuiteDispatchProbe, repoRoot, framework.BaselineSuiteEnv, BaselineSuiteFlag)
	cmd.Dir = repoRoot
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("probing CI planner and batch runner with %s: %w\n%s", python, err, stderr.String())
	}
	var dispatches []CISuiteDispatch
	if err := json.Unmarshal(out, &dispatches); err != nil {
		return nil, fmt.Errorf("decoding CI suite dispatch probe output: %w", err)
	}
	if len(dispatches) == 0 {
		return nil, fmt.Errorf("CI planner dispatches no e2e record; the baseline suite has no CI path")
	}
	return dispatches, nil
}
