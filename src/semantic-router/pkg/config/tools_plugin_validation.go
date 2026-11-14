package config

import (
	"errors"
	"fmt"
	"io/fs"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

// ErrToolSelectionStickyUnsupported is returned when sticky.enabled is true.
// The config, state and trust contract of issue #3347 ship without a request
// path that reads them, so an enabled block would construct and then never
// run; it is refused until Phase 3 (#4519) integrates reuse, authorization,
// invalidation, recovery and fallback together.
var ErrToolSelectionStickyUnsupported = errors.New(
	"sticky.enabled is not supported in this release (runtime integration is scheduled for Phase 3, issue #4519)",
)

// validateToolsPlugin checks a decision's tools plugin; path is the document
// path of its configuration.
func validateToolsPlugin(cfg *RouterConfig, path string, plugin *ToolsPluginConfig) error {
	if err := plugin.Policy.validate(path + ".policy"); err != nil {
		return err
	}
	if plugin.Select == nil {
		return nil
	}
	return validateToolsSelect(cfg, path+".select", plugin.Select, plugin.Policy)
}

func (p *ToolsPolicyConfig) validate(path string) error {
	if p == nil {
		return nil
	}
	mode := p.EffectiveMode()
	switch mode {
	case ToolsPolicyModePassthrough, ToolsPolicyModeFiltered, ToolsPolicyModeNone:
	default:
		return fmt.Errorf("%s.mode must be %s, %s or %s", path, ToolsPolicyModePassthrough, ToolsPolicyModeFiltered, ToolsPolicyModeNone)
	}
	lists := len(p.AllowTools) > 0 || len(p.BlockTools) > 0
	if mode == ToolsPolicyModeFiltered && !lists {
		return fmt.Errorf("%s: mode %s requires allow_tools or block_tools", path, ToolsPolicyModeFiltered)
	}
	if mode != ToolsPolicyModeFiltered && lists {
		return fmt.Errorf("%s: allow_tools and block_tools require mode %s", path, ToolsPolicyModeFiltered)
	}
	if p.StripToolHistory && mode != ToolsPolicyModeNone {
		return fmt.Errorf("%s.strip_tool_history requires mode %s", path, ToolsPolicyModeNone)
	}
	if err := p.TrustedFacts.Validate(); err != nil {
		return fmt.Errorf("%s.trusted_facts: %w", path, err)
	}
	return nil
}

// validateToolsSelect checks a selection: its mode's fields, its gate and
// keep rules, the deployment that answers tool_rank and tool_need, and add
// mode's catalog.
func validateToolsSelect(cfg *RouterConfig, path string, spec *ToolsSelectConfig, policy *ToolsPolicyConfig) error {
	if spec.Mode != ToolsSelectModeFilter && spec.Mode != ToolsSelectModeAdd {
		return fmt.Errorf("%s.mode must be %s or %s", path, ToolsSelectModeFilter, ToolsSelectModeAdd)
	}
	if policy.EffectiveMode() == ToolsPolicyModeNone {
		return fmt.Errorf("%s: policy mode %s removes every tool, so there is none to select", path, ToolsPolicyModeNone)
	}
	if err := validateToolsSelectModeFields(path, spec); err != nil {
		return err
	}
	if bound := spec.NeedThreshold; bound != nil && !within(*bound, 0, 1) {
		return fmt.Errorf("%s.need_threshold must be within [0, 1]", path)
	}
	if spec.TimeoutMs < 0 || spec.TimeoutMs > MaxSystemOneTimeoutMs {
		return fmt.Errorf("%s.timeout_ms must be within [1, %d] when set", path, MaxSystemOneTimeoutMs)
	}
	if err := validateToolsSelectMatch(cfg, path+".match", spec.Match); err != nil {
		return err
	}
	if err := spec.Sticky.validate(path + ".sticky"); err != nil {
		return err
	}
	if spec.Mode == ToolsSelectModeAdd {
		return validateToolCatalog(cfg, path+".tools_db_path", spec.ToolsDBPath, policy)
	}
	return nil
}

func validateToolsSelectModeFields(path string, spec *ToolsSelectConfig) error {
	if spec.Mode == ToolsSelectModeAdd {
		switch {
		case strings.TrimSpace(spec.ToolsDBPath) == "":
			return fmt.Errorf("%s.tools_db_path is required: add mode chooses from its catalog", path)
		case spec.PreserveCount != 0:
			return fmt.Errorf("%s.preserve_count keeps request tools in filter mode; add mode never removes them", path)
		case spec.OnError != "" && spec.OnError != ToolsSelectOnErrorSkip && spec.OnError != ToolsSelectOnErrorFail:
			return fmt.Errorf("%s.on_error must be %s or %s in add mode", path, ToolsSelectOnErrorSkip, ToolsSelectOnErrorFail)
		}
		return nil
	}
	switch {
	case spec.ToolsDBPath != "":
		return fmt.Errorf("%s.tools_db_path is read in add mode; filter mode chooses among the request's tools", path)
	case spec.PreserveCount < 0 || spec.PreserveCount > presets.MaxChoices:
		return fmt.Errorf("%s.preserve_count must be within [0, %d]", path, presets.MaxChoices)
	case spec.OnError != "" && spec.OnError != ToolsSelectOnErrorKeep && spec.OnError != ToolsSelectOnErrorStrip && spec.OnError != ToolsSelectOnErrorFail:
		return fmt.Errorf("%s.on_error must be %s, %s or %s in filter mode", path, ToolsSelectOnErrorKeep, ToolsSelectOnErrorStrip, ToolsSelectOnErrorFail)
	}
	return nil
}

// validateToolsSelectMatch checks the judge that ranks the candidates, which
// a written block must have: recall only narrows them. The selection's own
// on_error and timeout_ms govern its one call, which also asks tool_need on
// the judge's deployment.
func validateToolsSelectMatch(cfg *RouterConfig, path string, block *MatchConfig) error {
	if err := validateMatch(cfg, path, block, MatchToolsSelect); err != nil {
		return err
	}
	if block != nil && block.Judge == nil {
		return fmt.Errorf("%s: tool selection keeps the tools its judge ranks; add a judge, such as judge: {preset: %s}", path, presets.ToolRank)
	}
	judge := *MatchToolsSelect.Effective(block).Judge
	switch {
	case judge.OnError != "":
		return fmt.Errorf("%s.judge.on_error: the selection's on_error decides what a failed call does; remove it", path)
	case judge.TimeoutMs != 0:
		return fmt.Errorf("%s.judge.timeout_ms: the selection's timeout_ms bounds its call; remove it", path)
	case judge.TopK > presets.MaxChoices:
		return fmt.Errorf("%s.judge.top_k must be within [1, %d]", path, presets.MaxChoices)
	}
	deployment := cfg.MatchJudgeDeployment(judge)
	for _, questionType := range []string{judge.QuestionType(), SystemOneQuestionNoul} {
		if err := validateDeclaredQuestionType(cfg, deployment, questionType); err != nil {
			return fmt.Errorf("%s.judge: %w", path, err)
		}
	}
	return nil
}

// validateToolCatalog checks add mode's catalog where the configuration
// loads, when the file is there to read; the Router reads it again to serve
// it. A catalog of any size loads: the judge reads at most its
// max_candidates of the tools the policy allows.
func validateToolCatalog(cfg *RouterConfig, path, file string, policy *ToolsPolicyConfig) error {
	if cfg.SkipExternalAssetValidation {
		return nil
	}
	catalog, err := LoadToolCatalog(file)
	var duplicate *DuplicateToolError
	switch {
	case errors.Is(err, fs.ErrNotExist):
		return nil
	case errors.As(err, &duplicate):
		return CheckError(path, CodeDuplicateToolName, fmt.Sprintf("%s lists tool %q twice; a decision model tells the tools apart by name", file, duplicate.Name))
	case err != nil:
		return fmt.Errorf("%s: %w", path, err)
	}
	if len(policy.AllowedTools(catalog)) == 0 {
		return fmt.Errorf("%s: %s has no tool the policy allows", path, file)
	}
	return nil
}

func (s *StickyToolSelectionConfig) validate(path string) error {
	if s == nil {
		return nil
	}
	if s.MaxTools != nil && (*s.MaxTools <= 0 || *s.MaxTools > StickyToolSelectionMaxToolsUpperBound) {
		return fmt.Errorf("%s.max_tools must be within [1, %d]", path, StickyToolSelectionMaxToolsUpperBound)
	}
	if limit := s.EffectiveMaxTools(); s.MaxNewToolsPerTurn != nil && (*s.MaxNewToolsPerTurn < 0 || *s.MaxNewToolsPerTurn > limit) {
		return fmt.Errorf("%s.max_new_tools_per_turn must be within [0, %d], max_tools", path, limit)
	}
	if s.Enabled {
		return fmt.Errorf("%s: %w", path, ErrToolSelectionStickyUnsupported)
	}
	return nil
}

// Validate checks the trusted_facts block for legal enforcement modes,
// trust sources, freshness bounds, and stage roles (issue #3476). When the
// receiver is nil or Enabled is false, validation is a no-op so that adding
// the block never breaks existing deployments. Only operator-owned recipe
// policy, gateway-attested authorization, and bounded fresh runtime
// availability are authoritative; client metadata, prompt text, and model
// output are never allowed as trust sources. The check can only narrow
// availability, never widen privileges.
func (t *TrustedFactsConfig) Validate() error {
	if t == nil || !t.Enabled {
		return nil
	}

	switch t.EffectiveEnforcement() {
	case TrustedEnforcementDisabled, TrustedEnforcementAdvisory, TrustedEnforcementAuthoritative:
	default:
		return fmt.Errorf("enforcement must be one of %q, %q, or %q",
			TrustedEnforcementDisabled, TrustedEnforcementAdvisory, TrustedEnforcementAuthoritative)
	}

	if len(t.TrustSources) == 0 {
		return fmt.Errorf("trust_sources must declare at least one authoritative source")
	}
	allowedSources := map[string]bool{
		TrustedSourceOperatorPolicy:  true,
		TrustedSourceGatewayAttested: true,
		TrustedSourceRuntimeFresh:    true,
	}
	for _, s := range t.TrustSources {
		if !allowedSources[s] {
			return fmt.Errorf("trust_sources %q is not authoritative (must be one of %q, %q, %q)",
				s, TrustedSourceOperatorPolicy, TrustedSourceGatewayAttested, TrustedSourceRuntimeFresh)
		}
	}

	if t.FreshnessSeconds < 0 || t.FreshnessSeconds > 86400 {
		return fmt.Errorf("freshness_seconds must be in [0, 86400]")
	}

	// A runtime-fresh claim without a freshness bound disables the very check
	// that makes the claim meaningful. Require an explicit bound when the
	// source is declared.
	for _, s := range t.TrustSources {
		if s == TrustedSourceRuntimeFresh && t.FreshnessSeconds == 0 {
			return fmt.Errorf("freshness_seconds must be >0 when trust_sources includes %q", TrustedSourceRuntimeFresh)
		}
	}

	if len(t.StageRoles) == 0 {
		return fmt.Errorf("stage_roles must declare at least one Looper stage role")
	}
	allowedRoles := map[string]bool{
		TrustedStageCandidate: true,
		TrustedStageVerifier:  true,
		TrustedStageAdvisor:   true,
		TrustedStageFinal:     true,
	}
	for _, r := range t.StageRoles {
		if !allowedRoles[r] {
			return fmt.Errorf("stage_roles %q must be one of %q, %q, %q, %q",
				r, TrustedStageCandidate, TrustedStageVerifier, TrustedStageAdvisor, TrustedStageFinal)
		}
	}

	return nil
}
