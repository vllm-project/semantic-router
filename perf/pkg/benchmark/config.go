package benchmark

import (
	"bytes"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

const ManifestSchemaVersion = 1

var manifestName = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

// Manifest is the versioned execution contract shared by local runs and CI.
// Environments describe runner capabilities, profiles choose a bounded set of
// suites, and suites own the command needed to produce measurements.
type Manifest struct {
	SchemaVersion int                          `yaml:"schema_version"`
	Environments  map[string]EnvironmentConfig `yaml:"environments"`
	Profiles      map[string]ProfileConfig     `yaml:"profiles"`
	Suites        map[string]SuiteConfig       `yaml:"suites"`
}

type EnvironmentConfig struct {
	Kind         string            `yaml:"kind"`
	Accelerator  string            `yaml:"accelerator,omitempty"`
	Description  string            `yaml:"description,omitempty"`
	Capabilities []string          `yaml:"capabilities"`
	Env          map[string]string `yaml:"env,omitempty"`
}

type ProfileConfig struct {
	Description string   `yaml:"description,omitempty"`
	Suites      []string `yaml:"suites"`
	Count       int      `yaml:"count,omitempty"`
	BenchTime   string   `yaml:"benchtime,omitempty"`
	Timeout     string   `yaml:"timeout,omitempty"`
}

// SuiteConfig describes either a native Go benchmark suite or an external
// producer. External producers receive VSR_PERF_RESULT_FILE and must write the
// same Baseline JSON schema used by the built-in runner.
type SuiteConfig struct {
	Description  string              `yaml:"description,omitempty"`
	Runner       string              `yaml:"runner"`
	Module       string              `yaml:"module"`
	Packages     []string            `yaml:"packages,omitempty"`
	Benchmark    string              `yaml:"benchmark,omitempty"`
	Command      []string            `yaml:"command,omitempty"`
	Environments []string            `yaml:"environments"`
	Requires     []string            `yaml:"requires,omitempty"`
	Dimensions   map[string][]string `yaml:"dimensions,omitempty"`
	SourcePaths  []string            `yaml:"source_paths,omitempty"`
}

type ResolvedRun struct {
	EnvironmentName string
	Environment     EnvironmentConfig
	ProfileName     string
	Profile         ProfileConfig
	Suites          []ResolvedSuite
}

type ResolvedSuite struct {
	Name   string
	Config SuiteConfig
}

func LoadManifest(path string) (*Manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read performance manifest: %w", err)
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	var manifest Manifest
	if err := decoder.Decode(&manifest); err != nil {
		return nil, fmt.Errorf("parse performance manifest: %w", err)
	}
	if err := manifest.Validate(); err != nil {
		return nil, err
	}
	return &manifest, nil
}

func (m *Manifest) Validate() error {
	if m.SchemaVersion != ManifestSchemaVersion {
		return fmt.Errorf("unsupported performance manifest schema_version %d (want %d)", m.SchemaVersion, ManifestSchemaVersion)
	}
	if len(m.Environments) == 0 || len(m.Profiles) == 0 || len(m.Suites) == 0 {
		return fmt.Errorf("performance manifest requires environments, profiles, and suites")
	}
	if err := m.validateEnvironments(); err != nil {
		return err
	}
	if err := m.validateSuites(); err != nil {
		return err
	}
	return m.validateProfiles()
}

func (m *Manifest) validateEnvironments() error {
	for name, environment := range m.Environments {
		if !manifestName.MatchString(name) || strings.TrimSpace(environment.Kind) == "" {
			return fmt.Errorf("environment names and kinds must be non-empty")
		}
		if environment.Kind != "cpu" && environment.Kind != "gpu" {
			return fmt.Errorf("environment %q has unsupported kind %q", name, environment.Kind)
		}
		if environment.Kind == "gpu" && environment.Accelerator == "" {
			return fmt.Errorf("GPU environment %q must name an accelerator", name)
		}
		if duplicate := firstDuplicate(environment.Capabilities); duplicate != "" {
			return fmt.Errorf("environment %q repeats capability %q", name, duplicate)
		}
	}
	return nil
}

func (m *Manifest) validateSuites() error {
	for name, suite := range m.Suites {
		if err := m.validateSuite(name, suite); err != nil {
			return err
		}
	}
	return nil
}

func (m *Manifest) validateProfiles() error {
	for name, profile := range m.Profiles {
		if err := m.validateProfile(name, profile); err != nil {
			return err
		}
	}
	return nil
}

func (m *Manifest) validateProfile(name string, profile ProfileConfig) error {
	if !manifestName.MatchString(name) {
		return fmt.Errorf("invalid profile name %q", name)
	}
	if len(profile.Suites) == 0 {
		return fmt.Errorf("profile %q has no suites", name)
	}
	if profile.Count < 0 {
		return fmt.Errorf("profile %q count must be non-negative", name)
	}
	if err := validateProfileDuration(name, "benchtime", profile.BenchTime); err != nil {
		return err
	}
	if err := validateProfileDuration(name, "timeout", profile.Timeout); err != nil {
		return err
	}
	if duplicate := firstDuplicate(profile.Suites); duplicate != "" {
		return fmt.Errorf("profile %q repeats suite %q", name, duplicate)
	}
	for _, suite := range profile.Suites {
		if _, ok := m.Suites[suite]; !ok {
			return fmt.Errorf("profile %q references unknown suite %q", name, suite)
		}
	}
	return nil
}

func validateProfileDuration(profileName, field, value string) error {
	if value == "" {
		return nil
	}
	if _, err := time.ParseDuration(value); err != nil {
		return fmt.Errorf("profile %q has invalid %s: %w", profileName, field, err)
	}
	return nil
}

func (m *Manifest) validateSuite(name string, suite SuiteConfig) error {
	if !manifestName.MatchString(name) {
		return fmt.Errorf("invalid suite name %q", name)
	}
	if suite.Module == "" {
		return fmt.Errorf("suite %q must set module", name)
	}
	if len(suite.Environments) == 0 {
		return fmt.Errorf("suite %q must name at least one environment", name)
	}
	for _, environment := range suite.Environments {
		if _, ok := m.Environments[environment]; !ok {
			return fmt.Errorf("suite %q references unknown environment %q", name, environment)
		}
	}
	if duplicate := firstDuplicate(suite.Environments); duplicate != "" {
		return fmt.Errorf("suite %q repeats environment %q", name, duplicate)
	}
	return validateSuiteRunner(name, suite)
}

func validateSuiteRunner(name string, suite SuiteConfig) error {
	switch suite.Runner {
	case "go_benchmark":
		return validateGoBenchmarkSuite(name, suite)
	case "external":
		return validateExternalSuite(name, suite)
	default:
		return fmt.Errorf("suite %q has unsupported runner %q", name, suite.Runner)
	}
}

func validateGoBenchmarkSuite(name string, suite SuiteConfig) error {
	if len(suite.Packages) == 0 || suite.Benchmark == "" {
		return fmt.Errorf("go_benchmark suite %q requires packages and benchmark", name)
	}
	if _, err := regexp.Compile(suite.Benchmark); err != nil {
		return fmt.Errorf("suite %q has invalid benchmark regexp: %w", name, err)
	}
	if len(suite.Command) > 0 {
		return fmt.Errorf("go_benchmark suite %q cannot set command", name)
	}
	return nil
}

func validateExternalSuite(name string, suite SuiteConfig) error {
	if len(suite.Command) == 0 || strings.TrimSpace(suite.Command[0]) == "" {
		return fmt.Errorf("external suite %q requires command", name)
	}
	if len(suite.Packages) > 0 || suite.Benchmark != "" {
		return fmt.Errorf("external suite %q cannot set packages or benchmark", name)
	}
	return nil
}

func (m *Manifest) Resolve(environmentName, profileName string) (*ResolvedRun, error) {
	environment, ok := m.Environments[environmentName]
	if !ok {
		return nil, fmt.Errorf("unknown performance environment %q (available: %s)", environmentName, strings.Join(sortedMapKeys(m.Environments), ", "))
	}
	profile, ok := m.Profiles[profileName]
	if !ok {
		return nil, fmt.Errorf("unknown performance profile %q (available: %s)", profileName, strings.Join(sortedMapKeys(m.Profiles), ", "))
	}
	if profile.Count == 0 {
		profile.Count = 1
	}
	if profile.BenchTime == "" {
		profile.BenchTime = "1s"
	}
	if profile.Timeout == "" {
		profile.Timeout = "15m"
	}

	capabilities := make(map[string]struct{}, len(environment.Capabilities))
	for _, capability := range environment.Capabilities {
		capabilities[capability] = struct{}{}
	}
	resolved := &ResolvedRun{
		EnvironmentName: environmentName,
		Environment:     environment,
		ProfileName:     profileName,
		Profile:         profile,
	}
	for _, suiteName := range profile.Suites {
		suite := m.Suites[suiteName]
		if !contains(suite.Environments, environmentName) {
			return nil, fmt.Errorf("profile %q selects suite %q, which does not support environment %q", profileName, suiteName, environmentName)
		}
		for _, requirement := range suite.Requires {
			if _, ok := capabilities[requirement]; !ok {
				return nil, fmt.Errorf("suite %q requires capability %q, unavailable in environment %q", suiteName, requirement, environmentName)
			}
		}
		resolved.Suites = append(resolved.Suites, ResolvedSuite{Name: suiteName, Config: suite})
	}
	return resolved, nil
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func firstDuplicate(values []string) string {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			return value
		}
		seen[value] = struct{}{}
	}
	return ""
}

func sortedMapKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
