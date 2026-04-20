package dsl

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ---------- Level 4: Conflict Detection ----------

func (v *Validator) checkConflicts() {
	v.checkDomainSignalOverlap()
	v.checkSameSignalTypeGuard()
	v.checkProjectionPartitions()
	v.checkProjections()
	v.checkTestBlocks()
	v.checkTierConstraints()
	v.checkSessionStates()
}

// checkSessionStates validates SESSION_STATE declarations for duplicate names,
// invalid field types, duplicate field names, and empty names.
func (v *Validator) checkSessionStates() {
	seen := make(map[string]bool)
	validTypes := map[string]bool{"int": true, "string": true, "float": true}

	for _, ss := range v.prog.SessionStates {
		if ss.Name == "" {
			v.addDiag(DiagConstraint, ss.Pos, "SESSION_STATE: name cannot be empty", nil)
			continue
		}
		if seen[ss.Name] {
			v.addDiag(DiagConstraint, ss.Pos,
				fmt.Sprintf("SESSION_STATE %q: duplicate declaration name", ss.Name), nil)
			continue
		}
		seen[ss.Name] = true

		fieldsSeen := make(map[string]bool)
		for _, f := range ss.Fields {
			if f.Name == "" {
				v.addDiag(DiagConstraint, ss.Pos,
					fmt.Sprintf("SESSION_STATE %q: field name cannot be empty", ss.Name), nil)
				continue
			}
			if fieldsSeen[f.Name] {
				v.addDiag(DiagConstraint, ss.Pos,
					fmt.Sprintf("SESSION_STATE %q: duplicate field name %q", ss.Name, f.Name), nil)
				continue
			}
			fieldsSeen[f.Name] = true
			if !validTypes[f.TypeName] {
				v.addDiag(DiagConstraint, ss.Pos,
					fmt.Sprintf("SESSION_STATE %q: field %q has invalid type %q (supported: int, string, float)",
						ss.Name, f.Name, f.TypeName), nil)
			}
		}
	}
}

// checkDomainSignalOverlap detects MMLU category strings shared by two or more
// domain signals. When two domain signals list the same category, both will
// fire on queries in that category, causing silent routing conflicts where
// priority alone determines the winner regardless of classifier confidence.
func (v *Validator) checkDomainSignalOverlap() {
	type catSource struct {
		signalName string
		pos        Position
	}
	categoryToSignal := make(map[string]catSource)

	for _, s := range v.prog.Signals {
		if s.SignalType != "domain" {
			continue
		}
		cats := getMMLUCategories(s)
		for _, cat := range cats {
			if existing, clash := categoryToSignal[cat]; clash {
				v.addDiag(DiagWarning, s.Pos,
					fmt.Sprintf(
						"domain(%q) and domain(%q) share MMLU category %q — "+
							"both signals will fire on queries in this category, "+
							"causing priority to resolve ambiguously",
						s.Name, existing.signalName, cat),
					&QuickFix{
						Description: fmt.Sprintf("Remove %q from domain(%q) or domain(%q)", cat, s.Name, existing.signalName),
						NewText:     "",
					},
				)
			}
			categoryToSignal[cat] = catSource{signalName: s.Name, pos: s.Pos}
		}
	}
}

// getMMLUCategories extracts the mmlu_categories string array from a signal's fields.
func getMMLUCategories(s *SignalDecl) []string {
	raw, ok := s.Fields["mmlu_categories"]
	if !ok {
		return nil
	}
	av, ok := raw.(ArrayValue)
	if !ok {
		return nil
	}
	var cats []string
	for _, item := range av.Items {
		if sv, ok := item.(StringValue); ok {
			cats = append(cats, sv.V)
		}
	}
	return cats
}

// collectSignalRefs walks a boolean expression tree and classifies signal
// references as positive (directly referenced) or negated (under a NOT).
func collectSignalRefs(expr BoolExpr, negated bool, info *routeSignalInfo) {
	switch e := expr.(type) {
	case *BoolAnd:
		collectSignalRefs(e.Left, negated, info)
		collectSignalRefs(e.Right, negated, info)
	case *BoolOr:
		collectSignalRefs(e.Left, negated, info)
		collectSignalRefs(e.Right, negated, info)
	case *BoolNot:
		collectSignalRefs(e.Expr, !negated, info)
	case *SignalRefExpr:
		if negated {
			info.negatedRefs[e.SignalType] = append(info.negatedRefs[e.SignalType], e.SignalName)
		} else {
			info.positiveRefs[e.SignalType] = append(info.positiveRefs[e.SignalType], e.SignalName)
		}
	}
}

func containsString(ss []string, target string) bool {
	for _, s := range ss {
		if s == target {
			return true
		}
	}
	return false
}

// checkProjectionPartitions validates PROJECTION partition declarations: member existence,
// MMLU category disjointness within the group, default member existence,
// valid semantics value, and temperature range.
func (v *Validator) checkProjectionPartitions() {
	for _, partition := range v.prog.ProjectionPartitions {
		v.checkProjectionPartition(partition)
	}
	v.checkProjectionPartitionImpossibleANDs()
}

func (v *Validator) checkProjectionPartition(partition *ProjectionPartitionDecl) {
	context := fmt.Sprintf("PROJECTION partition %s", partition.Name)

	v.checkProjectionPartitionSemantics(partition, context)
	v.checkProjectionPartitionMembers(partition, context)
	v.checkProjectionPartitionMemberTypes(partition, context)
	v.checkProjectionPartitionDefault(partition, context)
	v.checkProjectionPartitionCategoryDisjointness(partition, context)
	v.checkProjectionPartitionSupportedDomainValues(partition, context)
}

func (v *Validator) checkProjectionPartitionSemantics(
	partition *ProjectionPartitionDecl,
	context string,
) {
	validSemantics := []string{"exclusive", "softmax_exclusive"}
	if partition.Semantics != "" && !containsString(validSemantics, partition.Semantics) {
		v.addDiag(DiagConstraint, partition.Pos,
			fmt.Sprintf("%s: unknown semantics %q (supported: exclusive, softmax_exclusive)", context, partition.Semantics),
			nil,
		)
	}
	if partition.Semantics == "softmax_exclusive" && partition.Temperature <= 0 {
		v.addDiag(DiagConstraint, partition.Pos,
			fmt.Sprintf("%s: softmax_exclusive requires temperature > 0", context),
			&QuickFix{Description: "Set temperature to 0.1", NewText: "0.1"},
		)
	}
}

func (v *Validator) checkProjectionPartitionMembers(
	partition *ProjectionPartitionDecl,
	context string,
) {
	if len(partition.Members) == 0 {
		v.addDiag(DiagConstraint, partition.Pos,
			fmt.Sprintf("%s: members list is empty", context),
			nil,
		)
	}
	for _, member := range partition.Members {
		if !v.isSignalDeclaredByName(member) {
			v.addDiag(DiagWarning, partition.Pos,
				fmt.Sprintf("%s: member %q is not defined as a signal", context, member),
				v.suggestSignalByName(member),
			)
		}
	}
}

func (v *Validator) checkProjectionPartitionMemberTypes(
	partition *ProjectionPartitionDecl,
	context string,
) {
	membersByType := make(map[string][]string)
	for _, member := range partition.Members {
		sig := v.findSignalByName(member)
		if sig == nil {
			continue
		}
		membersByType[sig.SignalType] = append(membersByType[sig.SignalType], member)
	}
	if len(membersByType) == 0 {
		return
	}

	if len(membersByType) == 1 {
		for signalType, members := range membersByType {
			if isSupportedProjectionPartitionType(signalType) {
				return
			}
			v.addDiag(DiagConstraint, partition.Pos,
				fmt.Sprintf(
					"%s: members must use a supported runtime signal type (domain or embedding), found %s=%v",
					context,
					signalType,
					members,
				),
				nil,
			)
			return
		}
	}

	v.addDiag(DiagConstraint, partition.Pos,
		fmt.Sprintf(
			"%s: members must all share one supported runtime signal type (domain or embedding), found %s",
			context,
			describeProjectionPartitionMemberTypes(membersByType),
		),
		nil,
	)
}

func (v *Validator) checkProjectionPartitionDefault(
	partition *ProjectionPartitionDecl,
	context string,
) {
	if partition.Default == "" {
		v.addDiag(DiagConstraint, partition.Pos,
			fmt.Sprintf("%s: a default member is required for coverage — every query must route somewhere", context),
			nil,
		)
		return
	}
	if !containsString(partition.Members, partition.Default) {
		v.addDiag(DiagWarning, partition.Pos,
			fmt.Sprintf("%s: default %q is not listed in members", context, partition.Default),
			nil,
		)
	}
}

func (v *Validator) checkProjectionPartitionCategoryDisjointness(
	partition *ProjectionPartitionDecl,
	context string,
) {
	catOwner := make(map[string]string)
	for _, member := range partition.Members {
		sig := v.findSignalByName(member)
		if sig == nil {
			continue
		}
		for _, cat := range getMMLUCategories(sig) {
			if existing, clash := catOwner[cat]; clash {
				v.addDiag(DiagWarning, partition.Pos,
					fmt.Sprintf(
						"%s: members %q and %q share MMLU category %q — "+
							"violates partition disjointness",
						context, member, existing, cat),
					nil,
				)
			}
			catOwner[cat] = member
		}
	}
}

func (v *Validator) checkProjectionPartitionSupportedDomainValues(
	partition *ProjectionPartitionDecl,
	context string,
) {
	if partition.Semantics != "softmax_exclusive" {
		return
	}
	for _, member := range partition.Members {
		sig := v.findSignalByName(member)
		if sig == nil || sig.SignalType != "domain" {
			continue
		}
		mmluCategories := getMMLUCategories(sig)
		if len(mmluCategories) == 0 {
			if config.IsSupportedRoutingDomainName(sig.Name) {
				continue
			}
			v.addDiag(
				DiagConstraint,
				partition.Pos,
				fmt.Sprintf(
					"%s: domain member %q must use a supported routing domain name (%s) or declare mmlu_categories explicitly%s",
					context,
					member,
					strings.Join(config.SupportedRoutingDomainNames(), ", "),
					formatDomainQuickFixSuffix(member),
				),
				domainQuickFix(member),
			)
			continue
		}
		for _, mmluCategory := range mmluCategories {
			if config.IsSupportedRoutingDomainName(mmluCategory) {
				continue
			}
			v.addDiag(
				DiagConstraint,
				partition.Pos,
				fmt.Sprintf(
					"%s: domain member %q has unsupported mmlu_categories value %q; supported values: %s%s",
					context,
					member,
					mmluCategory,
					strings.Join(config.SupportedRoutingDomainNames(), ", "),
					formatDomainQuickFixSuffix(mmluCategory),
				),
				domainQuickFix(mmluCategory),
			)
		}
	}
}

func isSupportedProjectionPartitionType(signalType string) bool {
	return signalType == "domain" || signalType == "embedding"
}

func describeProjectionPartitionMemberTypes(
	membersByType map[string][]string,
) string {
	keys := make([]string, 0, len(membersByType))
	for signalType := range membersByType {
		keys = append(keys, signalType)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, signalType := range keys {
		members := append([]string(nil), membersByType[signalType]...)
		sort.Strings(members)
		parts = append(parts, fmt.Sprintf("%s=%v", signalType, members))
	}
	return strings.Join(parts, ", ")
}

func (v *Validator) isSignalDeclaredByName(name string) bool {
	for _, s := range v.prog.Signals {
		if s.Name == name {
			return true
		}
	}
	return false
}

func (v *Validator) findSignalByName(name string) *SignalDecl {
	for _, s := range v.prog.Signals {
		if s.Name == name {
			return s
		}
	}
	return nil
}

func (v *Validator) checkProjectionPartitionImpossibleANDs() {
	memberToPartition := v.projectionPartitionMembers()
	if len(memberToPartition) == 0 {
		return
	}

	for _, route := range v.prog.Routes {
		if route.When == nil {
			continue
		}
		v.checkProjectionPartitionImpossibleANDsInRoute(route, memberToPartition)
	}
}

func (v *Validator) projectionPartitionMembers() map[string]string {
	memberToPartition := make(map[string]string)
	for _, partition := range v.prog.ProjectionPartitions {
		for _, member := range partition.Members {
			sig := v.findSignalByName(member)
			if sig == nil || !isSupportedProjectionPartitionType(sig.SignalType) {
				continue
			}
			memberToPartition[projectionPartitionMemberKey(sig.SignalType, member)] = partition.Name
		}
	}
	return memberToPartition
}

func projectionPartitionMemberKey(signalType string, signalName string) string {
	return signalType + ":" + signalName
}

func (v *Validator) checkProjectionPartitionImpossibleANDsInRoute(
	route *RouteDecl,
	memberToPartition map[string]string,
) {
	clauses := positiveConjunctionClauses(route.When)
	seen := make(map[string]struct{})

	for _, clause := range clauses {
		partitionMembers := make(map[string]SignalRefExpr)
		for _, ref := range clause {
			partitionName, ok := memberToPartition[projectionPartitionMemberKey(ref.SignalType, ref.SignalName)]
			if !ok {
				continue
			}

			if existing, clash := partitionMembers[partitionName]; clash && existing.SignalName != ref.SignalName {
				pair := []string{
					projectionPartitionMemberKey(existing.SignalType, existing.SignalName),
					projectionPartitionMemberKey(ref.SignalType, ref.SignalName),
				}
				sort.Strings(pair)
				diagKey := route.Name + "|" + partitionName + "|" + strings.Join(pair, "|")
				if _, alreadyReported := seen[diagKey]; alreadyReported {
					continue
				}
				seen[diagKey] = struct{}{}

				v.addDiag(DiagConstraint, route.Pos,
					fmt.Sprintf(
						"ROUTE %q: WHEN clause ANDs PROJECTION partition %q members %s(%q) and %s(%q), but that partition declares them mutually exclusive",
						route.Name,
						partitionName,
						existing.SignalType,
						existing.SignalName,
						ref.SignalType,
						ref.SignalName,
					),
					nil,
				)
				continue
			}

			partitionMembers[partitionName] = ref
		}
	}
}

func positiveConjunctionClauses(expr BoolExpr) [][]SignalRefExpr {
	switch e := expr.(type) {
	case *SignalRefExpr:
		return [][]SignalRefExpr{{*e}}
	case *BoolNot:
		return [][]SignalRefExpr{{}}
	case *BoolOr:
		left := positiveConjunctionClauses(e.Left)
		right := positiveConjunctionClauses(e.Right)
		return append(left, right...)
	case *BoolAnd:
		left := positiveConjunctionClauses(e.Left)
		right := positiveConjunctionClauses(e.Right)
		clauses := make([][]SignalRefExpr, 0, len(left)*len(right))
		for _, leftClause := range left {
			for _, rightClause := range right {
				clause := make([]SignalRefExpr, 0, len(leftClause)+len(rightClause))
				clause = append(clause, leftClause...)
				clause = append(clause, rightClause...)
				clauses = append(clauses, clause)
			}
		}
		return clauses
	default:
		return nil
	}
}

func (v *Validator) suggestSignalByName(name string) *QuickFix {
	var candidates []string
	for _, s := range v.prog.Signals {
		candidates = append(candidates, s.Name)
	}
	closest := suggestSimilar(name, candidates)
	if closest == "" {
		return nil
	}
	return &QuickFix{
		Description: fmt.Sprintf("Change to %q", closest),
		NewText:     closest,
	}
}

func (v *Validator) checkProjections() {
	scoreNames := make(map[string]bool, len(v.prog.ProjectionScores))
	for _, score := range v.prog.ProjectionScores {
		if score.Name == "" {
			v.addDiag(DiagConstraint, score.Pos, "PROJECTION score: name cannot be empty", nil)
			continue
		}
		if scoreNames[score.Name] {
			v.addDiag(DiagConstraint, score.Pos, fmt.Sprintf("PROJECTION score %q: duplicate score name", score.Name), nil)
			continue
		}
		scoreNames[score.Name] = true
		v.checkProjectionScore(score)
	}

	outputNames := make(map[string]bool)
	for _, mapping := range v.prog.ProjectionMappings {
		if mapping.Name == "" {
			v.addDiag(DiagConstraint, mapping.Pos, "PROJECTION mapping: name cannot be empty", nil)
			continue
		}
		v.checkProjectionMapping(mapping, scoreNames, outputNames)
	}
}

func (v *Validator) checkProjectionScore(score *ProjectionScoreDecl) {
	context := fmt.Sprintf("PROJECTION score %s", score.Name)
	if score.Method == "" {
		score.Method = "weighted_sum"
	}
	if score.Method != "weighted_sum" {
		v.addDiag(
			DiagConstraint,
			score.Pos,
			fmt.Sprintf("%s: unknown method %q (supported: weighted_sum)", context, score.Method),
			nil,
		)
	}
	if len(score.Inputs) == 0 {
		v.addDiag(DiagConstraint, score.Pos, fmt.Sprintf("%s: inputs cannot be empty", context), nil)
		return
	}

	for _, input := range score.Inputs {
		v.checkProjectionScoreInput(context, score.Pos, input)
	}
}

func (v *Validator) checkProjectionScoreInput(context string, pos Position, input *ProjectionScoreInputDecl) {
	if input == nil {
		return
	}
	if !isProjectionInputTypeSupported(input.SignalType) {
		v.addDiag(
			DiagConstraint,
			pos,
			fmt.Sprintf(
				"%s: input %s(%q) uses unsupported type %q",
				context,
				input.SignalType,
				input.SignalName,
				input.SignalType,
			),
			nil,
		)
		return
	}
	if !v.isSignalDefined(input.SignalType, input.SignalName) {
		v.addDiag(
			DiagWarning,
			pos,
			fmt.Sprintf("%s: input %s(%q) is not defined", context, input.SignalType, input.SignalName),
			v.suggestSignalByName(input.SignalName),
		)
	}
	switch input.ValueSource {
	case "", "binary", "confidence", "raw":
	default:
		v.addDiag(
			DiagConstraint,
			pos,
			fmt.Sprintf(
				"%s: input %s(%q) has unsupported value_source %q (supported: binary, confidence, raw)",
				context,
				input.SignalType,
				input.SignalName,
				input.ValueSource,
			),
			nil,
		)
	}
}

func isProjectionInputTypeSupported(signalType string) bool {
	switch signalType {
	case config.SignalTypeKeyword,
		config.SignalTypeEmbedding,
		config.SignalTypeDomain,
		config.SignalTypeFactCheck,
		config.SignalTypeUserFeedback,
		config.SignalTypeReask,
		config.SignalTypePreference,
		config.SignalTypeLanguage,
		config.SignalTypeContext,
		config.SignalTypeStructure,
		config.SignalTypeComplexity,
		config.SignalTypeModality,
		config.SignalTypeAuthz,
		config.SignalTypeJailbreak,
		config.SignalTypePII:
		return true
	default:
		return false
	}
}

func (v *Validator) checkProjectionMapping(mapping *ProjectionMappingDecl, scoreNames map[string]bool, outputNames map[string]bool) {
	context := fmt.Sprintf("PROJECTION mapping %s", mapping.Name)
	if mapping.Method == "" {
		mapping.Method = "threshold_bands"
	}
	if !scoreNames[mapping.Source] {
		v.addDiag(
			DiagConstraint,
			mapping.Pos,
			fmt.Sprintf("%s: source %q is not a declared projection score", context, mapping.Source),
			nil,
		)
	}
	if mapping.Method != "threshold_bands" {
		v.addDiag(
			DiagConstraint,
			mapping.Pos,
			fmt.Sprintf("%s: unknown method %q (supported: threshold_bands)", context, mapping.Method),
			nil,
		)
	}
	if mapping.Calibration != nil {
		switch mapping.Calibration.Method {
		case "", "sigmoid_distance":
		default:
			v.addDiag(
				DiagConstraint,
				mapping.Pos,
				fmt.Sprintf(
					"%s: unsupported calibration method %q (supported: sigmoid_distance)",
					context,
					mapping.Calibration.Method,
				),
				nil,
			)
		}
	}
	if len(mapping.Outputs) == 0 {
		v.addDiag(DiagConstraint, mapping.Pos, fmt.Sprintf("%s: outputs cannot be empty", context), nil)
		return
	}
	for _, output := range mapping.Outputs {
		v.checkProjectionMappingOutput(context, mapping.Pos, output, outputNames)
	}
}

func (v *Validator) checkProjectionMappingOutput(
	context string,
	pos Position,
	output *ProjectionMappingOutputDecl,
	outputNames map[string]bool,
) {
	if output == nil {
		return
	}
	if output.Name == "" {
		v.addDiag(DiagConstraint, pos, fmt.Sprintf("%s: output name cannot be empty", context), nil)
		return
	}
	if outputNames[output.Name] {
		v.addDiag(DiagConstraint, pos, fmt.Sprintf("%s: duplicate output name %q", context, output.Name), nil)
		return
	}
	outputNames[output.Name] = true

	if output.GT == nil && output.GTE == nil && output.LT == nil && output.LTE == nil {
		v.addDiag(
			DiagConstraint,
			pos,
			fmt.Sprintf("%s output %q: at least one threshold bound is required", context, output.Name),
			nil,
		)
	}
	if output.GT != nil && output.GTE != nil {
		v.addDiag(DiagConstraint, pos, fmt.Sprintf("%s output %q: cannot set both gt and gte", context, output.Name), nil)
	}
	if output.LT != nil && output.LTE != nil {
		v.addDiag(DiagConstraint, pos, fmt.Sprintf("%s output %q: cannot set both lt and lte", context, output.Name), nil)
	}
}

// checkTestBlocks validates TEST block entries: referenced routes must exist.
func (v *Validator) checkTestBlocks() {
	routeNames := make(map[string]bool)
	for _, r := range v.prog.Routes {
		routeNames[r.Name] = true
	}

	for _, tb := range v.prog.TestBlocks {
		for _, entry := range tb.Entries {
			if !routeNames[entry.RouteName] {
				v.addDiag(DiagWarning, entry.Pos,
					fmt.Sprintf("TEST %s: route %q is not defined", tb.Name, entry.RouteName),
					nil,
				)
			}
			if entry.Query == "" {
				v.addDiag(DiagConstraint, entry.Pos,
					fmt.Sprintf("TEST %s: query string is empty", tb.Name),
					nil,
				)
			}
		}
	}
}

// checkTierConstraints validates TIER values on routes.
func (v *Validator) checkTierConstraints() {
	for _, r := range v.prog.Routes {
		if r.Tier < 0 {
			v.addDiag(DiagConstraint, r.Pos,
				fmt.Sprintf("ROUTE %s: tier must be >= 0, got %d", r.Name, r.Tier),
				&QuickFix{Description: "Set tier to 0", NewText: "0"},
			)
		}
	}
}

func keysOfBool(m map[string]bool) []string {
	result := make([]string, 0, len(m))
	for k := range m {
		result = append(result, k)
	}
	sort.Strings(result)
	return result
}
