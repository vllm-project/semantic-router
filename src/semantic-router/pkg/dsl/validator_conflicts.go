package dsl

import (
	"fmt"
	"sort"
	"strings"
)

// ---------- Level 4: Conflict Detection ----------

func (v *Validator) checkConflicts() {
	v.checkDomainSignalOverlap()
	v.checkSameSignalTypeGuard()
	v.checkSignalGroups()
	v.checkTestBlocks()
	v.checkTierConstraints()
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

// checkSameSignalTypeGuard warns when two routes reference the same signal type
// in their WHEN clauses without a NOT guard for mutual exclusion. Without a guard,
// both routes can match the same query, and priority alone picks the winner —
// even if the lower-priority route's signal has higher confidence.
// routeSignalInfo collects positive and negated signal references from a route's WHEN clause.
type routeSignalInfo struct {
	route        *RouteDecl
	positiveRefs map[string][]string // signalType → [signalName, ...]
	negatedRefs  map[string][]string // signalType → [signalName, ...] (under NOT)
}

func (v *Validator) checkSameSignalTypeGuard() {
	infos := v.collectRouteSignalInfos()
	for i := 0; i < len(infos); i++ {
		for j := i + 1; j < len(infos); j++ {
			v.checkSignalTypeGuardPair(infos[i], infos[j])
		}
	}
}

func (v *Validator) collectRouteSignalInfos() []routeSignalInfo {
	var infos []routeSignalInfo
	for _, r := range v.prog.Routes {
		if r.When == nil {
			continue
		}
		info := routeSignalInfo{
			route:        r,
			positiveRefs: make(map[string][]string),
			negatedRefs:  make(map[string][]string),
		}
		collectSignalRefs(r.When, false, &info)
		infos = append(infos, info)
	}
	return infos
}

func (v *Validator) checkSignalTypeGuardPair(a, b routeSignalInfo) {
	for sigType, aNamesPos := range a.positiveRefs {
		bNamesPos, ok := b.positiveRefs[sigType]
		if !ok {
			continue
		}
		for _, aName := range aNamesPos {
			for _, bName := range bNamesPos {
				if aName == bName {
					continue
				}
				v.emitGuardDiagIfNeeded(sigType, a, b, aName, bName)
			}
		}
	}
}

func (v *Validator) emitGuardDiagIfNeeded(sigType string, a, b routeSignalInfo, aName, bName string) {
	hiRoute, loRoute, hiName, loName := a.route, b.route, aName, bName
	if b.route.Priority > a.route.Priority ||
		(b.route.Priority == a.route.Priority && b.route.Tier > a.route.Tier) {
		hiRoute, loRoute, hiName, loName = b.route, a.route, bName, aName
	}
	loInfo := b
	if hiRoute == b.route {
		loInfo = a
	}
	if containsString(loInfo.negatedRefs[sigType], hiName) {
		return
	}

	v.addDiag(DiagWarning, loRoute.Pos,
		fmt.Sprintf(
			"ROUTE %q and ROUTE %q both reference %s signals "+
				"(%s(%q) and %s(%q)) with no mutual exclusion guard — "+
				"both can fire on the same query",
			hiRoute.Name, loRoute.Name,
			sigType, sigType, hiName, sigType, loName),
		&QuickFix{
			Description: fmt.Sprintf(
				"Add guard: WHEN %s(%q) AND NOT %s(%q)",
				sigType, loName, sigType, hiName),
			NewText: fmt.Sprintf(
				"%s(\"%s\") AND NOT %s(\"%s\")",
				sigType, loName, sigType, hiName),
		},
	)
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

// checkSignalGroups validates SIGNAL_GROUP declarations: member existence,
// MMLU category disjointness within the group, default member existence,
// valid semantics value, and temperature range.
func (v *Validator) checkSignalGroups() {
	for _, sg := range v.prog.SignalGroups {
		v.checkSignalGroup(sg)
	}
	v.checkSignalGroupImpossibleANDs()
}

func (v *Validator) checkSignalGroup(sg *SignalGroupDecl) {
	context := fmt.Sprintf("SIGNAL_GROUP %s", sg.Name)

	v.checkSignalGroupSemantics(sg, context)
	v.checkSignalGroupMembers(sg, context)
	v.checkSignalGroupMemberTypes(sg, context)
	v.checkSignalGroupDefault(sg, context)
	v.checkSignalGroupCategoryDisjointness(sg, context)
}

func (v *Validator) checkSignalGroupSemantics(sg *SignalGroupDecl, context string) {
	validSemantics := []string{"exclusive", "softmax_exclusive"}
	if sg.Semantics != "" && !containsString(validSemantics, sg.Semantics) {
		v.addDiag(DiagConstraint, sg.Pos,
			fmt.Sprintf("%s: unknown semantics %q (supported: exclusive, softmax_exclusive)", context, sg.Semantics),
			nil,
		)
	}
	if sg.Semantics == "softmax_exclusive" && sg.Temperature <= 0 {
		v.addDiag(DiagConstraint, sg.Pos,
			fmt.Sprintf("%s: softmax_exclusive requires temperature > 0", context),
			&QuickFix{Description: "Set temperature to 0.1", NewText: "0.1"},
		)
	}
}

func (v *Validator) checkSignalGroupMembers(sg *SignalGroupDecl, context string) {
	if len(sg.Members) == 0 {
		v.addDiag(DiagConstraint, sg.Pos,
			fmt.Sprintf("%s: members list is empty", context),
			nil,
		)
	}
	for _, member := range sg.Members {
		if !v.isSignalDeclaredByName(member) {
			v.addDiag(DiagWarning, sg.Pos,
				fmt.Sprintf("%s: member %q is not defined as a signal", context, member),
				v.suggestSignalByName(member),
			)
		}
	}
}

func (v *Validator) checkSignalGroupMemberTypes(sg *SignalGroupDecl, context string) {
	membersByType := make(map[string][]string)
	for _, member := range sg.Members {
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
			if isSupportedSignalGroupType(signalType) {
				return
			}
			v.addDiag(DiagConstraint, sg.Pos,
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

	v.addDiag(DiagConstraint, sg.Pos,
		fmt.Sprintf(
			"%s: members must all share one supported runtime signal type (domain or embedding), found %s",
			context,
			describeSignalGroupMemberTypes(membersByType),
		),
		nil,
	)
}

func (v *Validator) checkSignalGroupDefault(sg *SignalGroupDecl, context string) {
	if sg.Default == "" {
		v.addDiag(DiagConstraint, sg.Pos,
			fmt.Sprintf("%s: a default member is required for coverage — every query must route somewhere", context),
			nil,
		)
		return
	}
	if !containsString(sg.Members, sg.Default) {
		v.addDiag(DiagWarning, sg.Pos,
			fmt.Sprintf("%s: default %q is not listed in members", context, sg.Default),
			nil,
		)
	}
}

func (v *Validator) checkSignalGroupCategoryDisjointness(sg *SignalGroupDecl, context string) {
	catOwner := make(map[string]string)
	for _, member := range sg.Members {
		sig := v.findSignalByName(member)
		if sig == nil {
			continue
		}
		for _, cat := range getMMLUCategories(sig) {
			if existing, clash := catOwner[cat]; clash {
				v.addDiag(DiagWarning, sg.Pos,
					fmt.Sprintf(
						"%s: members %q and %q share MMLU category %q — "+
							"violates group disjointness",
						context, member, existing, cat),
					nil,
				)
			}
			catOwner[cat] = member
		}
	}
}

func isSupportedSignalGroupType(signalType string) bool {
	return signalType == "domain" || signalType == "embedding"
}

func describeSignalGroupMemberTypes(membersByType map[string][]string) string {
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

func (v *Validator) checkSignalGroupImpossibleANDs() {
	memberToGroup := v.signalGroupMembers()
	if len(memberToGroup) == 0 {
		return
	}

	for _, route := range v.prog.Routes {
		if route.When == nil {
			continue
		}
		v.checkSignalGroupImpossibleANDsInRoute(route, memberToGroup)
	}
}

func (v *Validator) signalGroupMembers() map[string]string {
	memberToGroup := make(map[string]string)
	for _, group := range v.prog.SignalGroups {
		for _, member := range group.Members {
			sig := v.findSignalByName(member)
			if sig == nil || !isSupportedSignalGroupType(sig.SignalType) {
				continue
			}
			memberToGroup[signalGroupMemberKey(sig.SignalType, member)] = group.Name
		}
	}
	return memberToGroup
}

func signalGroupMemberKey(signalType string, signalName string) string {
	return signalType + ":" + signalName
}

func (v *Validator) checkSignalGroupImpossibleANDsInRoute(route *RouteDecl, memberToGroup map[string]string) {
	clauses := positiveConjunctionClauses(route.When)
	seen := make(map[string]struct{})

	for _, clause := range clauses {
		groupMembers := make(map[string]SignalRefExpr)
		for _, ref := range clause {
			groupName, ok := memberToGroup[signalGroupMemberKey(ref.SignalType, ref.SignalName)]
			if !ok {
				continue
			}

			if existing, clash := groupMembers[groupName]; clash && existing.SignalName != ref.SignalName {
				pair := []string{
					signalGroupMemberKey(existing.SignalType, existing.SignalName),
					signalGroupMemberKey(ref.SignalType, ref.SignalName),
				}
				sort.Strings(pair)
				diagKey := route.Name + "|" + groupName + "|" + strings.Join(pair, "|")
				if _, alreadyReported := seen[diagKey]; alreadyReported {
					continue
				}
				seen[diagKey] = struct{}{}

				v.addDiag(DiagConstraint, route.Pos,
					fmt.Sprintf(
						"ROUTE %q: WHEN clause ANDs SIGNAL_GROUP %q members %s(%q) and %s(%q), but that group declares them mutually exclusive",
						route.Name,
						groupName,
						existing.SignalType,
						existing.SignalName,
						ref.SignalType,
						ref.SignalName,
					),
					nil,
				)
				continue
			}

			groupMembers[groupName] = ref
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
