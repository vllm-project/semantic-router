package dsl

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type routeSignalClause struct {
	positiveRefs map[string][]string // signalType -> [signalName, ...]
	negatedRefs  map[string][]string // signalType -> [signalName, ...]
}

type guardConflictSemantics struct {
	exclusivePartitionMembers map[string]string
	projectionOutputOwners    map[string]string
	contextRanges             map[string]contextSignalRange
}

type contextSignalRange struct {
	minTokens int
	maxTokens int
}

func (v *Validator) guardConflictSemantics() guardConflictSemantics {
	return guardConflictSemantics{
		exclusivePartitionMembers: v.projectionPartitionMembers(),
		projectionOutputOwners:    v.projectionOutputOwners(),
		contextRanges:             v.contextSignalRanges(),
	}
}

func (v *Validator) contextSignalRanges() map[string]contextSignalRange {
	ranges := make(map[string]contextSignalRange)
	for _, signal := range v.prog.Signals {
		if signal.SignalType != config.SignalTypeContext {
			continue
		}

		minTokens, ok := getStringField(signal.Fields, "min_tokens")
		if !ok {
			continue
		}
		maxTokens, ok := getStringField(signal.Fields, "max_tokens")
		if !ok {
			continue
		}

		minValue, err := config.TokenCount(minTokens).Value()
		if err != nil {
			continue
		}
		maxValue, err := config.TokenCount(maxTokens).Value()
		if err != nil {
			continue
		}
		ranges[signal.Name] = contextSignalRange{
			minTokens: minValue,
			maxTokens: maxValue,
		}
	}
	return ranges
}

func collectRouteSignalClauses(expr BoolExpr, negated bool) []routeSignalClause {
	switch e := expr.(type) {
	case *SignalRefExpr:
		clause := routeSignalClause{
			positiveRefs: make(map[string][]string),
			negatedRefs:  make(map[string][]string),
		}
		target := clause.positiveRefs
		if negated {
			target = clause.negatedRefs
		}
		target[e.SignalType] = append(target[e.SignalType], e.SignalName)
		return []routeSignalClause{clause}
	case *BoolNot:
		return collectRouteSignalClauses(e.Expr, !negated)
	case *BoolAnd:
		if negated {
			left := collectRouteSignalClauses(e.Left, true)
			right := collectRouteSignalClauses(e.Right, true)
			return append(left, right...)
		}
		return crossRouteSignalClauses(
			collectRouteSignalClauses(e.Left, false),
			collectRouteSignalClauses(e.Right, false),
		)
	case *BoolOr:
		if negated {
			return crossRouteSignalClauses(
				collectRouteSignalClauses(e.Left, true),
				collectRouteSignalClauses(e.Right, true),
			)
		}
		left := collectRouteSignalClauses(e.Left, false)
		right := collectRouteSignalClauses(e.Right, false)
		return append(left, right...)
	default:
		return nil
	}
}

func crossRouteSignalClauses(left []routeSignalClause, right []routeSignalClause) []routeSignalClause {
	if len(left) == 0 {
		return right
	}
	if len(right) == 0 {
		return left
	}

	clauses := make([]routeSignalClause, 0, len(left)*len(right))
	for _, leftClause := range left {
		for _, rightClause := range right {
			mergedClause, ok := mergeRouteSignalClauses(leftClause, rightClause)
			if ok {
				clauses = append(clauses, mergedClause)
			}
		}
	}
	return clauses
}

func mergeRouteSignalClauses(left routeSignalClause, right routeSignalClause) (routeSignalClause, bool) {
	merged := routeSignalClause{
		positiveRefs: cloneRouteSignalRefMap(left.positiveRefs),
		negatedRefs:  cloneRouteSignalRefMap(left.negatedRefs),
	}
	mergeRouteSignalRefMap(merged.positiveRefs, right.positiveRefs)
	mergeRouteSignalRefMap(merged.negatedRefs, right.negatedRefs)
	if routeSignalRefMapsConflict(merged.positiveRefs, merged.negatedRefs) {
		return routeSignalClause{}, false
	}
	return merged, true
}

func cloneRouteSignalRefMap(refs map[string][]string) map[string][]string {
	cloned := make(map[string][]string, len(refs))
	for signalType, names := range refs {
		cloned[signalType] = append([]string(nil), names...)
	}
	return cloned
}

func mergeRouteSignalRefMap(dst map[string][]string, src map[string][]string) {
	for signalType, names := range src {
		dst[signalType] = append(dst[signalType], names...)
	}
}

func routeSignalRefMapsConflict(positiveRefs map[string][]string, negatedRefs map[string][]string) bool {
	for signalType, positiveNames := range positiveRefs {
		if routeSignalNamesOverlap(positiveNames, negatedRefs[signalType]) {
			return true
		}
	}
	return false
}

func routeSignalNamesOverlap(first []string, second []string) bool {
	if len(first) == 0 || len(second) == 0 {
		return false
	}
	secondSet := make(map[string]struct{}, len(second))
	for _, name := range second {
		secondSet[name] = struct{}{}
	}
	for _, name := range first {
		if _, ok := secondSet[name]; ok {
			return true
		}
	}
	return false
}

func normalizeRouteSignalClauses(
	clauses []routeSignalClause,
	semantics guardConflictSemantics,
) []routeSignalClause {
	if len(clauses) == 0 {
		return nil
	}

	var normalized []routeSignalClause
	seen := make(map[string]struct{}, len(clauses))
	for _, clause := range clauses {
		normalizeRouteSignalRefMap(clause.positiveRefs)
		normalizeRouteSignalRefMap(clause.negatedRefs)
		if !routeSignalClauseIsSatisfiable(clause, semantics) {
			continue
		}

		signature := routeSignalClauseSignature(clause)
		if _, alreadySeen := seen[signature]; alreadySeen {
			continue
		}
		seen[signature] = struct{}{}
		normalized = append(normalized, clause)
	}
	return normalized
}

func routeSignalClauseSignature(clause routeSignalClause) string {
	return fmt.Sprintf(
		"pos=%s|neg=%s",
		routeSignalRefMapSignature(clause.positiveRefs),
		routeSignalRefMapSignature(clause.negatedRefs),
	)
}

func routeSignalRefMapSignature(refs map[string][]string) string {
	if len(refs) == 0 {
		return ""
	}
	signalTypes := make([]string, 0, len(refs))
	for signalType := range refs {
		signalTypes = append(signalTypes, signalType)
	}
	sort.Strings(signalTypes)

	parts := make([]string, 0, len(signalTypes))
	for _, signalType := range signalTypes {
		parts = append(parts, signalType+":"+strings.Join(refs[signalType], ","))
	}
	return strings.Join(parts, ";")
}

func routeSignalClauseIsSatisfiable(
	clause routeSignalClause,
	semantics guardConflictSemantics,
) bool {
	if routeSignalRefMapsConflict(clause.positiveRefs, clause.negatedRefs) {
		return false
	}

	for signalType, positiveNames := range clause.positiveRefs {
		for i := 0; i < len(positiveNames); i++ {
			for j := i + 1; j < len(positiveNames); j++ {
				if signalsMutuallyExclusive(signalType, positiveNames[i], positiveNames[j], semantics) {
					return false
				}
			}
		}
	}
	return true
}

func routeSignalClausesCompatible(
	hiClause routeSignalClause,
	loClause routeSignalClause,
	semantics guardConflictSemantics,
) bool {
	if routeSignalRefMapsConflict(hiClause.positiveRefs, loClause.negatedRefs) ||
		routeSignalRefMapsConflict(loClause.positiveRefs, hiClause.negatedRefs) {
		return false
	}

	for signalType, hiNames := range hiClause.positiveRefs {
		loNames := loClause.positiveRefs[signalType]
		for _, hiName := range hiNames {
			for _, loName := range loNames {
				if signalsMutuallyExclusive(signalType, hiName, loName, semantics) {
					return false
				}
			}
		}
	}
	return true
}

func signalsMutuallyExclusive(
	signalType string,
	firstName string,
	secondName string,
	semantics guardConflictSemantics,
) bool {
	if firstName == secondName {
		return false
	}

	if signalType == config.SignalTypeProjection &&
		projectionOutputsShareMapping(semantics.projectionOutputOwners, firstName, secondName) {
		return true
	}
	if signalType == config.SignalTypeContext &&
		contextSignalsDisjoint(semantics.contextRanges, firstName, secondName) {
		return true
	}
	return signalPartitionMembersShareExclusiveGroup(semantics.exclusivePartitionMembers, signalType, firstName, secondName)
}

func signalPartitionMembersShareExclusiveGroup(
	partitionMembers map[string]string,
	signalType string,
	firstName string,
	secondName string,
) bool {
	firstPartition, firstOK := partitionMembers[projectionPartitionMemberKey(signalType, firstName)]
	secondPartition, secondOK := partitionMembers[projectionPartitionMemberKey(signalType, secondName)]
	return firstOK && secondOK && firstPartition == secondPartition
}

func contextSignalsDisjoint(
	contextRanges map[string]contextSignalRange,
	firstName string,
	secondName string,
) bool {
	firstRange, firstOK := contextRanges[firstName]
	secondRange, secondOK := contextRanges[secondName]
	if !firstOK || !secondOK {
		return false
	}
	return firstRange.maxTokens < secondRange.minTokens || secondRange.maxTokens < firstRange.minTokens
}

func signalTypeParticipatesInGuardWarning(signalType string) bool {
	return signalType != config.SignalTypeProjection
}

func overlaySignalType() string {
	return config.SignalTypeUserFeedback
}

func routeSignalClausesRequireSignalType(clauses []routeSignalClause, signalType string) bool {
	if len(clauses) == 0 {
		return false
	}
	for _, clause := range clauses {
		if len(clause.positiveRefs[signalType]) == 0 {
			return false
		}
	}
	return true
}

func guardWarningPairSuppressed(hiInfo routeSignalInfo, loInfo routeSignalInfo) bool {
	return hiInfo.isOverlay || loInfo.isOverlay
}
