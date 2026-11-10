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
	v.checkContextSignalBands()
	v.checkSameSignalTypeGuard()
	v.checkProjectionPartitions()
	v.checkProjections()
	v.checkTestBlocks()
	v.checkTierConstraints()
}

// checkDomainSignalOverlap detects a domain label two or more domain signals
// match. Both fire on a request in that domain, so priority alone decides
// between them.
func (v *Validator) checkDomainSignalOverlap() {
	type labelSource struct {
		signalName string
		pos        Position
	}
	labelToSignal := make(map[string]labelSource)

	for _, s := range v.prog.Signals {
		if s.SignalType != "domain" {
			continue
		}
		for _, label := range domainLabels(s) {
			if existing, clash := labelToSignal[label]; clash {
				v.addDiag(DiagWarning, s.Pos,
					fmt.Sprintf(
						"domain(%q) and domain(%q) both match domain %q — "+
							"both signals will fire on queries in this domain, "+
							"causing priority to resolve ambiguously",
						s.Name, existing.signalName, label),
					&QuickFix{
						Description: fmt.Sprintf("Remove %q from domain(%q) or domain(%q)", label, s.Name, existing.signalName),
						NewText:     "",
					},
				)
			}
			labelToSignal[label] = labelSource{signalName: s.Name, pos: s.Pos}
		}
	}
}

// domainLabels are the domains a domain signal matches: its labels, else its
// name.
func domainLabels(s *SignalDecl) []string {
	raw, ok := s.Fields["labels"]
	if !ok {
		return []string{s.Name}
	}
	av, ok := raw.(ArrayValue)
	if !ok {
		return nil
	}
	var labels []string
	for _, item := range av.Items {
		if sv, ok := item.(StringValue); ok {
			labels = append(labels, sv.V)
		}
	}
	return labels
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
			info.negatedRefs[e.SignalType] = append(info.negatedRefs[e.SignalType], guardSignalName(e))
		} else {
			info.positiveRefs[e.SignalType] = append(info.positiveRefs[e.SignalType], guardSignalName(e))
		}
	}
}

// guardSignalName is the exact reference that route guards compare. A
// labelled reference names its label, so conditions on different options of
// one decision question or classifier are different references.
func guardSignalName(ref *SignalRefExpr) string {
	if label, ok := ref.Fields["label"].(StringValue); ok && label.V != "" {
		return ref.SignalName + ":" + label.V
	}
	return ref.SignalName
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
		if sig == nil || isSupportedProjectionPartitionType(sig.SignalType) {
			continue
		}
		membersByType[sig.SignalType] = append(membersByType[sig.SignalType], member)
	}
	if len(membersByType) == 0 {
		return
	}
	v.addDiag(DiagConstraint, partition.Pos,
		fmt.Sprintf(
			"%s: members must be domain signals, found %s",
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
	labelOwner := make(map[string]string)
	for _, member := range partition.Members {
		sig := v.findSignalByName(member)
		if sig == nil || sig.SignalType != "domain" {
			continue
		}
		for _, label := range domainLabels(sig) {
			if existing, clash := labelOwner[label]; clash {
				v.addDiag(DiagWarning, partition.Pos,
					fmt.Sprintf(
						"%s: members %q and %q both match domain %q — "+
							"violates partition disjointness",
						context, member, existing, label),
					nil,
				)
			}
			labelOwner[label] = member
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
		for _, label := range domainLabels(sig) {
			if config.IsSupportedRoutingDomainName(label) {
				continue
			}
			v.addDiag(
				DiagConstraint,
				partition.Pos,
				fmt.Sprintf(
					"%s: domain member %q matches %q, which is not a domain; supported values: %s%s",
					context,
					member,
					label,
					strings.Join(config.SupportedRoutingDomainNames(), ", "),
					formatDomainQuickFixSuffix(label),
				),
				domainQuickFix(label),
			)
		}
	}
}

func isSupportedProjectionPartitionType(signalType string) bool {
	return signalType == "domain"
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
	// Distributing AND over OR can produce contradictory cross-products even
	// when the original expression is satisfiable: (A OR B) AND (A OR B) still
	// accepts A or B. An impossible-route constraint requires every alternative
	// to contradict a partition; one viable alternative disproves that claim.
	clauses := positiveConjunctionClauses(route.When)
	seen := make(map[string]struct{})
	var conflicts []string
	for _, clause := range clauses {
		partitionMembers := make(map[string]SignalRefExpr)
		impossible := false
		for _, ref := range clause {
			partitionName, ok := memberToPartition[projectionPartitionMemberKey(ref.SignalType, ref.SignalName)]
			if !ok {
				continue
			}
			if existing, clash := partitionMembers[partitionName]; clash && existing.SignalName != ref.SignalName {
				impossible = true
				pair := []string{projectionPartitionMemberKey(existing.SignalType, existing.SignalName), projectionPartitionMemberKey(ref.SignalType, ref.SignalName)}
				sort.Strings(pair)
				diagKey := partitionName + "|" + strings.Join(pair, "|")
				if _, alreadyReported := seen[diagKey]; alreadyReported {
					continue
				}
				seen[diagKey] = struct{}{}
				conflicts = append(conflicts, fmt.Sprintf(
					"ROUTE %q: WHEN clause ANDs PROJECTION partition %q members %s(%q) and %s(%q), but that partition declares them mutually exclusive",
					route.Name, partitionName, existing.SignalType, existing.SignalName, ref.SignalType, ref.SignalName,
				))
				continue
			}
			partitionMembers[partitionName] = ref
		}
		if !impossible {
			return
		}
	}
	for _, message := range conflicts {
		v.addDiag(DiagConstraint, route.Pos, message, nil)
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
		if ref := signalReferenceName(s); ref != "" {
			candidates = append(candidates, ref)
		}
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
	}
	validated := make(map[string]bool, len(scoreNames))
	for _, score := range v.prog.ProjectionScores {
		if score.Name == "" || !scoreNames[score.Name] || validated[score.Name] {
			continue
		}
		validated[score.Name] = true
		v.checkProjectionScore(score, scoreNames)
	}

	v.checkProjectionScoreCycles()

	outputNames := make(map[string]bool)
	for _, mapping := range v.prog.ProjectionMappings {
		if mapping.Name == "" {
			v.addDiag(DiagConstraint, mapping.Pos, "PROJECTION mapping: name cannot be empty", nil)
			continue
		}
		v.checkProjectionMapping(mapping, scoreNames, outputNames)
	}
}

func (v *Validator) checkProjectionScore(score *ProjectionScoreDecl, scoreNames map[string]bool) {
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
		v.checkProjectionScoreInput(context, score.Pos, input, scoreNames)
	}
}

func (v *Validator) checkProjectionScoreInputProjectionRef(context string, pos Position, input *ProjectionScoreInputDecl, scoreNames map[string]bool) {
	if input.SignalName == "" {
		v.addDiag(DiagConstraint, pos,
			fmt.Sprintf("%s: projection input requires a name referencing a declared score or mapping output", context), nil)
		return
	}
	switch input.ValueSource {
	case "", config.ProjectionValueSourceScore:
		if !scoreNames[input.SignalName] {
			v.addDiag(DiagWarning, pos,
				fmt.Sprintf("%s: projection input %q references undeclared score", context, input.SignalName), nil)
		}
	case config.ProjectionValueSourceConfidence:
		if !v.isProjectionOutputDefined(input.SignalName) {
			v.addDiag(DiagWarning, pos,
				fmt.Sprintf("%s: projection input %q with value_source \"confidence\" references undeclared mapping output", context, input.SignalName), nil)
		}
	default:
		v.addDiag(DiagConstraint, pos,
			fmt.Sprintf("%s: projection input %q has unsupported value_source %q (supported: score, confidence)",
				context, input.SignalName, input.ValueSource), nil)
	}
}

func (v *Validator) checkProjectionScoreInput(context string, pos Position, input *ProjectionScoreInputDecl, scoreNames map[string]bool) {
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
	if strings.EqualFold(input.SignalType, config.SignalTypeProjection) {
		v.checkProjectionScoreInputProjectionRef(context, pos, input, scoreNames)
		return
	}
	name := input.SignalName
	if strings.EqualFold(input.SignalType, config.SignalTypeSystemOne) || config.IsApplicationSignalType(strings.ToLower(input.SignalType)) {
		// A systemone or application input may read one option of its
		// question: "<rule>:<option>".
		name, _, _ = strings.Cut(name, ":")
	}
	if !v.isSignalDefined(input.SignalType, name) {
		v.addDiag(
			DiagWarning,
			pos,
			fmt.Sprintf("%s: input %s(%q) is not defined", context, input.SignalType, input.SignalName),
			v.suggestSignalByName(input.SignalName),
		)
	}
	switch input.ValueSource {
	case "", config.ProjectionValueSourceBinary, config.ProjectionValueSourceConfidence, config.ProjectionValueSourceRaw:
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
	return config.IsSupportedSignalType(signalType) ||
		signalType == config.SignalTypeProjection
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
