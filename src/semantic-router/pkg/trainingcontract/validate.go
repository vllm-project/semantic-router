package trainingcontract

import (
	"fmt"
	"regexp"
	"slices"
)

var (
	revisionPattern   = regexp.MustCompile(`^[0-9a-f]{40}$`)
	repositoryPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`)
	handlePattern     = regexp.MustCompile(`^[a-z]+_[a-zA-Z0-9-]+$`)
)

func ValidateHandle(handle string) error {
	if !handlePattern.MatchString(handle) {
		return fmt.Errorf("expected an owned handle, got %q", handle)
	}
	return nil
}

func ValidateTarget(target Target) error {
	switch target {
	case Selector, LabelScores, Spans:
		return nil
	}
	return fmt.Errorf("unknown target_contract %q", target)
}

func ValidateComponent(c Component) error {
	if c.Name == "" || c.Version == "" {
		return fmt.Errorf("component name and version are required")
	}
	return nil
}

func ValidateModel(m ModelRef) error {
	if !repositoryPattern.MatchString(m.Repository) || !revisionPattern.MatchString(m.Revision) {
		return fmt.Errorf("source requires a repository and pinned 40-character revision")
	}
	return nil
}

func ValidateProfile(p Profile) error {
	switch p.TargetContract {
	case Selector:
		if p.Selector == nil || p.Classifier != nil || p.Spans != nil {
			return fmt.Errorf("selector target requires only selector profile")
		}
		if !nonemptyUnique(p.Selector.CandidateModels) || !nonemptyUnique(p.Selector.ObservationFields) {
			return fmt.Errorf("selector requires nonempty unique candidates and observation fields")
		}
	case LabelScores:
		if p.Classifier == nil || p.Selector != nil || p.Spans != nil {
			return fmt.Errorf("label-scores target requires only classifier profile")
		}
		labels := p.Classifier.LabelMapping
		if len(labels) == 0 {
			return fmt.Errorf("label_mapping is required")
		}
		seen := make(map[int]bool, len(labels))
		for label, index := range labels {
			if label == "" || index < 0 || index >= len(labels) || seen[index] {
				return fmt.Errorf("label_mapping must have unique contiguous indices")
			}
			seen[index] = true
		}
	case Spans:
		if p.Spans == nil || p.Selector != nil || p.Classifier != nil || !nonemptyUnique(p.Spans.Labels) || p.Spans.OffsetUnit != "unicode-codepoint" {
			return fmt.Errorf("spans target requires labels and unicode-codepoint offsets")
		}
	default:
		return ValidateTarget(p.TargetContract)
	}
	return nil
}

func ValidateRun(r SubmitRunRequest) error {
	if r.SchemaVersion != Version || r.IdempotencyKey == "" {
		return fmt.Errorf("schema_version and idempotency_key are required")
	}
	return ValidateRunSpec(r.Spec)
}

// ValidateRunSpec checks relationships that JSON Schema cannot express.
// Resource existence, ownership and trainer capabilities belong to the management service.
func ValidateRunSpec(spec RunSpec) error {
	if spec.BaseModel != nil {
		if err := ValidateModel(*spec.BaseModel); err != nil {
			return err
		}
	}
	if err := ValidateHandle(spec.ExperimentID); err != nil {
		return err
	}
	if err := ValidateHandle(spec.SnapshotID); err != nil {
		return err
	}
	if err := ValidateTarget(spec.TargetContract); err != nil {
		return err
	}
	if err := ValidateComponent(spec.Trainer); err != nil {
		return err
	}
	if len(spec.Tasks) == 0 {
		return fmt.Errorf("at least one task is required")
	}
	tasks := map[string]TaskSpec{}
	for _, t := range spec.Tasks {
		if t.Key == "" {
			return fmt.Errorf("task key is required")
		}
		if _, ok := tasks[t.Key]; ok {
			return fmt.Errorf("duplicate task key %q", t.Key)
		}
		if err := ValidateComponent(t.Executor); err != nil {
			return err
		}
		if !unique(t.DependsOn) {
			return fmt.Errorf("duplicate dependency for %q", t.Key)
		}
		tasks[t.Key] = t
	}
	visiting, visited := map[string]bool{}, map[string]bool{}
	var visit func(string) error
	visit = func(key string) error {
		t, ok := tasks[key]
		if !ok {
			return fmt.Errorf("unknown dependency %q", key)
		}
		if visiting[key] {
			return fmt.Errorf("dependency cycle at %q", key)
		}
		if visited[key] {
			return nil
		}
		visiting[key] = true
		for _, dep := range t.DependsOn {
			if err := visit(dep); err != nil {
				return err
			}
		}
		visiting[key] = false
		visited[key] = true
		return nil
	}
	for key := range tasks {
		if err := visit(key); err != nil {
			return err
		}
	}
	return nil
}

func Terminal(s Status) bool { return s == Succeeded || s == Failed || s == Cancelled || s == Skipped }

// ValidateTransition describes the public run state machine. Retry is explicit.
func ValidateTransition(from, to Status) error {
	allowed := false
	switch from {
	case Pending:
		allowed = to == Running || to == Cancelling
	case Running:
		allowed = to == Succeeded || to == Failed || to == Cancelling
	case Cancelling:
		allowed = to == Cancelled
	case Failed, Cancelled:
		allowed = to == Pending
	}
	if !allowed {
		return fmt.Errorf("invalid state transition %s -> %s", from, to)
	}
	return nil
}

func nonemptyUnique(values []string) bool {
	return len(values) > 0 && unique(values)
}

func unique(values []string) bool {
	for i, value := range values {
		if slices.Contains(values[:i], value) {
			return false
		}
	}
	return true
}
