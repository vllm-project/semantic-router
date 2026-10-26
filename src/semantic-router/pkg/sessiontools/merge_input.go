package sessiontools

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const configMaxRetainedTools = config.StickyToolSelectionMaxToolsUpperBound

// preparedSelection owns its slices/maps. Preparation happens once per
// manager operation, so caller mutation during a store callback cannot change
// a retry's candidate set or grow its per-turn budget.
type preparedSelection struct {
	input    SelectionInput
	eligible map[string]ToolIdentity
	called   map[string]bool
	required map[string]bool
}

func prepareSelection(input SelectionInput) (preparedSelection, error) {
	p := preparedSelection{input: input}
	if err := validateSelectionBounds(input); err != nil {
		return p, err
	}
	if err := p.prepareEligible(input.Eligible); err != nil {
		return p, err
	}
	if err := p.prepareRanking(input.Ranked); err != nil {
		return p, err
	}
	var err error
	p.input.Called, err = normalizeObservations(input.Called)
	if err != nil {
		return p, err
	}
	p.input.Required, err = normalizeObservations(input.Required)
	if err != nil {
		return p, err
	}
	p.called, p.required = make(map[string]bool), make(map[string]bool)
	for _, name := range p.input.Called {
		if _, ok := p.eligible[name]; ok && input.Bounds.PinCalledTools {
			p.called[name] = true
		}
	}
	for _, name := range p.input.Required {
		if _, ok := p.eligible[name]; !ok {
			return p, &RequiredToolsError{Unavailable: true}
		}
		p.required[name] = true
	}
	return p, nil
}

func validateSelectionBounds(input SelectionInput) error {
	b := input.Bounds
	if b.MaxTools < 1 || b.MaxTools > config.StickyToolSelectionMaxToolsUpperBound ||
		b.MaxNewToolsPerTurn < 0 || b.MaxNewToolsPerTurn > b.MaxTools ||
		b.MaxStateBytes < 1 || b.MaxStateBytes > config.ToolSessionStoreMaxMaxStateBytes {
		return fmt.Errorf("%w: invalid count or byte bounds", ErrInvalidSelection)
	}
	if len(input.Eligible) > MaxSelectionInputs || len(input.Ranked) > MaxSelectionInputs ||
		len(input.Called) > MaxSelectionInputs || len(input.Required) > MaxSelectionInputs {
		return fmt.Errorf("%w: input list exceeds %d", ErrInvalidSelection, MaxSelectionInputs)
	}
	for _, value := range []string{input.Fingerprints.Policy, input.Fingerprints.Catalog, input.Fingerprints.Capability} {
		if !validFingerprint(value) {
			return fmt.Errorf("%w: missing or malformed fingerprint", ErrInvalidSelection)
		}
	}
	if len(input.StrategyID) > 128 {
		return fmt.Errorf("%w: strategy identifier exceeds 128 bytes", ErrInvalidSelection)
	}
	return nil
}

func validFingerprint(value string) bool {
	return value != "" && len(value) <= 128 && strings.TrimSpace(value) == value
}

func normalizeToolName(name string) (string, error) {
	name = strings.TrimSpace(name)
	if name == "" || len(name) > 256 {
		return "", fmt.Errorf("%w: tool name must contain 1..256 bytes", ErrInvalidSelection)
	}
	return name, nil
}

func normalizeIdentity(tool ToolIdentity) (ToolIdentity, error) {
	var err error
	tool.Name, err = normalizeToolName(tool.Name)
	if err != nil {
		return tool, err
	}
	if !validFingerprint(tool.DefinitionFingerprint) {
		return tool, fmt.Errorf("%w: malformed definition fingerprint", ErrInvalidSelection)
	}
	return tool, nil
}

func (p *preparedSelection) prepareEligible(input []ToolIdentity) error {
	p.eligible = make(map[string]ToolIdentity, len(input))
	p.input.Eligible = make([]ToolIdentity, 0, len(input))
	for _, raw := range input {
		tool, err := normalizeIdentity(raw)
		if err != nil {
			return err
		}
		if _, exists := p.eligible[tool.Name]; exists {
			return fmt.Errorf("%w: duplicate eligible tool name", ErrInvalidSelection)
		}
		p.eligible[tool.Name] = tool
		p.input.Eligible = append(p.input.Eligible, tool)
	}
	return nil
}

func (p *preparedSelection) prepareRanking(input []RankedTool) error {
	p.input.Ranked = make([]RankedTool, 0, len(input))
	seen := make(map[string]bool, len(input))
	for _, candidate := range input {
		tool, err := normalizeIdentity(candidate.ToolIdentity)
		if err != nil {
			return err
		}
		if math.IsNaN(candidate.Score) || math.IsInf(candidate.Score, 0) || seen[tool.Name] {
			return fmt.Errorf("%w: non-finite score or duplicate candidate", ErrInvalidSelection)
		}
		seen[tool.Name] = true
		if current, ok := p.eligible[tool.Name]; !ok || current != tool {
			continue // Ranking and observations can only narrow eligibility.
		}
		candidate.ToolIdentity = tool
		p.input.Ranked = append(p.input.Ranked, candidate)
	}
	sort.Slice(p.input.Ranked, func(i, j int) bool {
		a, b := p.input.Ranked[i], p.input.Ranked[j]
		if a.Score == b.Score {
			return a.Name < b.Name
		}
		return a.Score > b.Score
	})
	return nil
}

func normalizeObservations(input []string) ([]string, error) {
	seen := make(map[string]bool, len(input))
	result := make([]string, 0, len(input))
	for _, value := range input {
		name, err := normalizeToolName(value)
		if err != nil {
			return nil, err
		}
		if !seen[name] {
			result = append(result, name)
			seen[name] = true
		}
	}
	sort.Strings(result)
	return result, nil
}
