package dsl

import (
	"fmt"
	"sort"
	"strings"
)

// routeSignalInfo collects positive and negated signal references from a route's WHEN clause.
type routeSignalInfo struct {
	route        *RouteDecl
	positiveRefs map[string][]string // signalType -> [signalName, ...]
	negatedRefs  map[string][]string // signalType -> [signalName, ...] (under NOT)
	clauses      []routeSignalClause
	isOverlay    bool
}

type signalTypeGuardOverlap struct {
	signalType string
	hiName     string
	loName     string
}

// checkSameSignalTypeGuard warns when two routes reference the same signal type
// in their WHEN clauses without a NOT guard for mutual exclusion. Without a guard,
// both routes can match the same query, and priority alone picks the winner —
// even if the lower-priority route's signal has higher confidence.
func (v *Validator) checkSameSignalTypeGuard() {
	semantics := v.guardConflictSemantics()
	infos := v.collectRouteSignalInfos(semantics)
	for i := 0; i < len(infos); i++ {
		for j := i + 1; j < len(infos); j++ {
			v.checkSignalTypeGuardPair(infos[i], infos[j], semantics)
		}
	}
}

func (v *Validator) collectRouteSignalInfos(semantics guardConflictSemantics) []routeSignalInfo {
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
		normalizeRouteSignalInfo(&info)
		info.clauses = normalizeRouteSignalClauses(collectRouteSignalClauses(r.When, false), semantics)
		info.isOverlay = routeSignalClausesRequireSignalType(info.clauses, overlaySignalType())
		infos = append(infos, info)
	}
	return infos
}

func normalizeRouteSignalInfo(info *routeSignalInfo) {
	normalizeRouteSignalRefMap(info.positiveRefs)
	normalizeRouteSignalRefMap(info.negatedRefs)
}

func normalizeRouteSignalRefMap(refs map[string][]string) {
	for signalType, names := range refs {
		refs[signalType] = sortedUniqueStrings(names)
	}
}

func sortedUniqueStrings(names []string) []string {
	if len(names) == 0 {
		return nil
	}
	unique := make(map[string]struct{}, len(names))
	for _, name := range names {
		unique[name] = struct{}{}
	}
	result := make([]string, 0, len(unique))
	for name := range unique {
		result = append(result, name)
	}
	sort.Strings(result)
	return result
}

func (v *Validator) checkSignalTypeGuardPair(
	a routeSignalInfo,
	b routeSignalInfo,
	semantics guardConflictSemantics,
) {
	hiInfo, loInfo := orderedRouteSignalInfos(a, b)
	if guardWarningPairSuppressed(hiInfo, loInfo) {
		return
	}
	overlaps := collectSignalTypeGuardOverlaps(hiInfo, loInfo, semantics)
	if len(overlaps) == 0 {
		return
	}
	v.emitAggregatedGuardDiag(hiInfo, loInfo, overlaps)
}

func orderedRouteSignalInfos(a, b routeSignalInfo) (routeSignalInfo, routeSignalInfo) {
	if b.route.Priority > a.route.Priority ||
		(b.route.Priority == a.route.Priority && b.route.Tier > a.route.Tier) {
		return b, a
	}
	return a, b
}

func collectSignalTypeGuardOverlaps(
	hiInfo routeSignalInfo,
	loInfo routeSignalInfo,
	semantics guardConflictSemantics,
) []signalTypeGuardOverlap {
	overlapSet := make(map[string]signalTypeGuardOverlap)
	for _, hiClause := range hiInfo.clauses {
		collectCompatibleClauseGuardOverlaps(overlapSet, hiClause, loInfo.clauses, semantics)
	}
	return sortSignalTypeGuardOverlaps(overlapSet)
}

func collectCompatibleClauseGuardOverlaps(
	overlapSet map[string]signalTypeGuardOverlap,
	hiClause routeSignalClause,
	loClauses []routeSignalClause,
	semantics guardConflictSemantics,
) {
	for _, loClause := range loClauses {
		if !routeSignalClausesCompatible(hiClause, loClause, semantics) {
			continue
		}
		collectClauseSignalTypeGuardOverlaps(overlapSet, hiClause, loClause)
	}
}

func collectClauseSignalTypeGuardOverlaps(
	overlapSet map[string]signalTypeGuardOverlap,
	hiClause routeSignalClause,
	loClause routeSignalClause,
) {
	for signalType, hiNames := range hiClause.positiveRefs {
		if !signalTypeParticipatesInGuardWarning(signalType) {
			continue
		}
		loNames, ok := loClause.positiveRefs[signalType]
		if !ok {
			continue
		}
		addSignalTypeGuardOverlaps(overlapSet, signalType, hiNames, loNames)
	}
}

func addSignalTypeGuardOverlaps(
	overlapSet map[string]signalTypeGuardOverlap,
	signalType string,
	hiNames []string,
	loNames []string,
) {
	for _, hiName := range hiNames {
		for _, loName := range loNames {
			if hiName == loName {
				continue
			}
			key := signalType + "|" + hiName + "|" + loName
			overlapSet[key] = signalTypeGuardOverlap{
				signalType: signalType,
				hiName:     hiName,
				loName:     loName,
			}
		}
	}
}

func sortSignalTypeGuardOverlaps(overlapSet map[string]signalTypeGuardOverlap) []signalTypeGuardOverlap {
	overlaps := make([]signalTypeGuardOverlap, 0, len(overlapSet))
	for _, overlap := range overlapSet {
		overlaps = append(overlaps, overlap)
	}
	sort.Slice(overlaps, func(i, j int) bool {
		if overlaps[i].signalType != overlaps[j].signalType {
			return overlaps[i].signalType < overlaps[j].signalType
		}
		if overlaps[i].hiName != overlaps[j].hiName {
			return overlaps[i].hiName < overlaps[j].hiName
		}
		return overlaps[i].loName < overlaps[j].loName
	})
	return overlaps
}

func (v *Validator) projectionOutputOwners() map[string]string {
	owners := make(map[string]string)
	for _, mapping := range v.prog.ProjectionMappings {
		for _, output := range mapping.Outputs {
			if output == nil || output.Name == "" {
				continue
			}
			owners[output.Name] = mapping.Name
		}
	}
	return owners
}

func projectionOutputsShareMapping(owners map[string]string, firstOutput string, secondOutput string) bool {
	firstMapping, firstOK := owners[firstOutput]
	secondMapping, secondOK := owners[secondOutput]
	return firstOK && secondOK && firstMapping == secondMapping
}

func (v *Validator) emitAggregatedGuardDiag(
	hiInfo routeSignalInfo,
	loInfo routeSignalInfo,
	overlaps []signalTypeGuardOverlap,
) {
	countsByType := make(map[string]int)
	examples := make([]string, 0, 3)
	for _, overlap := range overlaps {
		countsByType[overlap.signalType]++
		if len(examples) < 3 {
			examples = append(examples, formatSignalTypeGuardExample(overlap))
		}
	}

	countParts := make([]string, 0, len(countsByType))
	for _, signalType := range sortedCountKeys(countsByType) {
		countParts = append(countParts, fmt.Sprintf("%s=%d", signalType, countsByType[signalType]))
	}

	totalOverlaps := len(overlaps)
	overlapLabel := "overlap"
	if totalOverlaps != 1 {
		overlapLabel = "overlaps"
	}
	exampleLabel := "example"
	if len(examples) != 1 {
		exampleLabel = "examples"
	}

	firstOverlap := overlaps[0]
	v.addDiag(DiagWarning, loInfo.route.Pos,
		fmt.Sprintf(
			"ROUTE %q and ROUTE %q have %d same-signal-type %s with no mutual exclusion guard (%s); %s: %s; both can fire on the same query",
			hiInfo.route.Name,
			loInfo.route.Name,
			totalOverlaps,
			overlapLabel,
			strings.Join(countParts, ", "),
			exampleLabel,
			strings.Join(examples, ", "),
		),
		&QuickFix{
			Description: fmt.Sprintf(
				"Add guard: WHEN %s(%q) AND NOT %s(%q)",
				firstOverlap.signalType,
				firstOverlap.loName,
				firstOverlap.signalType,
				firstOverlap.hiName,
			),
			NewText: fmt.Sprintf(
				"%s(\"%s\") AND NOT %s(\"%s\")",
				firstOverlap.signalType,
				firstOverlap.loName,
				firstOverlap.signalType,
				firstOverlap.hiName,
			),
		},
	)
}

func formatSignalTypeGuardExample(overlap signalTypeGuardOverlap) string {
	return fmt.Sprintf(
		"%s(%q) vs %s(%q)",
		overlap.signalType,
		overlap.hiName,
		overlap.signalType,
		overlap.loName,
	)
}

func sortedCountKeys(counts map[string]int) []string {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
