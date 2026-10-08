package classification

import (
	"context"
	"fmt"
	"strconv"
)

// A local consumer without a mapping file takes its label vocabulary, in
// output order, from the card of the model the runtime serves. A mapping file
// stays an explicit override that renames those labels.

const servedLabelsSource = "served labels"

func (m *classifierModelRuntime) servedLabels(consumer, artifact, contract string, useCPU bool) ([]string, error) {
	spec, err := m.localSpec(consumer, artifact, "auto", contract, useCPU)
	if err != nil {
		return nil, err
	}
	labels, err := m.runtime.Labels(context.Background(), spec)
	if err != nil {
		return nil, fmt.Errorf("%s labels: %w", consumer, err)
	}
	return labels, nil
}

func categoryMappingFromLabels(labels []string) (*CategoryMapping, error) {
	mapping := &CategoryMapping{CategoryToIdx: make(map[string]int, len(labels)), IdxToCategory: make(map[string]string, len(labels))}
	for index, label := range labels {
		mapping.CategoryToIdx[label] = index
		mapping.IdxToCategory[strconv.Itoa(index)] = label
	}
	if err := validateCategoryMapping(servedLabelsSource, mapping); err != nil {
		return nil, err
	}
	return mapping, nil
}

func piiMappingFromLabels(labels []string) (*PIIMapping, error) {
	mapping := &PIIMapping{LabelToIdx: make(map[string]int, len(labels)), IdxToLabel: make(map[string]string, len(labels)), spanNamed: labels == nil}
	for index, label := range labels {
		mapping.LabelToIdx[label] = index
		mapping.IdxToLabel[strconv.Itoa(index)] = label
	}
	if len(mapping.LabelToIdx) != len(labels) {
		return nil, fmt.Errorf("PII %s repeat a label", servedLabelsSource)
	}
	if mapping.hasReservedLabel() {
		return nil, fmt.Errorf("PII %s: labels %q and %q are reserved for the on_error and on_unscanned sentinels", servedLabelsSource, PIIClassificationErrorType, PIIUnscannedType)
	}
	return mapping, nil
}

func jailbreakMappingFromLabels(labels []string) (*JailbreakMapping, error) {
	mapping := &JailbreakMapping{LabelToIdx: make(map[string]int, len(labels)), IdxToLabel: make(map[string]string, len(labels))}
	for index, label := range labels {
		mapping.LabelToIdx[label] = index
		mapping.IdxToLabel[strconv.Itoa(index)] = label
	}
	if err := canonicalizeJailbreakMapping(servedLabelsSource, mapping); err != nil {
		return nil, err
	}
	if _, collides := mapping.GetIndexForJailbreakType(JailbreakClassificationErrorType); collides {
		return nil, fmt.Errorf("jailbreak %s: label %q is reserved for the on_error: block sentinel", servedLabelsSource, JailbreakClassificationErrorType)
	}
	if _, collides := mapping.GetIndexForJailbreakType(JailbreakUnscannedType); collides {
		return nil, fmt.Errorf("jailbreak %s: label %q is reserved for content the model did not read", servedLabelsSource, JailbreakUnscannedType)
	}
	return mapping, nil
}
