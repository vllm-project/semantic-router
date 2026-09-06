package config

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
)

// DiffCanonicalDocuments walks two redacted YAML documents against the
// canonical schema. Named slices are keyed by name when present.
func DiffCanonicalDocuments(active, candidate map[string]interface{}) *ConfigDiff {
	diff := &ConfigDiff{
		Added:   []DiffEntry{},
		Removed: []DiffEntry{},
		Changed: []DiffEntry{},
	}
	left := normalizeYAMLValue(active)
	right := normalizeYAMLValue(candidate)
	walkDiff(diff, "", left, right)
	return diff
}

func walkDiff(diff *ConfigDiff, path string, left, right any) {
	if !diff.accept() {
		diff.Truncated = true
		return
	}
	if reflect.DeepEqual(comparableYAMLValue(left), comparableYAMLValue(right)) {
		return
	}
	leftMap, leftIsMap := asStringMap(left)
	rightMap, rightIsMap := asStringMap(right)
	if leftIsMap && rightIsMap {
		walkMapDiff(diff, path, leftMap, rightMap)
		return
	}
	leftSlice, leftIsSlice := asSlice(left)
	rightSlice, rightIsSlice := asSlice(right)
	if leftIsSlice && rightIsSlice {
		walkSliceDiff(diff, path, leftSlice, rightSlice)
		return
	}
	if left == nil {
		diff.add(DiffEntry{Field: displayDiffPath(path), New: redactDiffValue(path, right)})
		return
	}
	if right == nil {
		diff.remove(DiffEntry{Field: displayDiffPath(path), Old: redactDiffValue(path, left)})
		return
	}
	diff.change(DiffEntry{
		Field: displayDiffPath(path),
		Old:   redactDiffValue(path, left),
		New:   redactDiffValue(path, right),
	})
}

func walkMapDiff(diff *ConfigDiff, path string, left, right map[string]interface{}) {
	keys := make([]string, 0, len(left)+len(right))
	seen := map[string]struct{}{}
	for key := range left {
		keys = append(keys, key)
		seen[key] = struct{}{}
	}
	for key := range right {
		if _, ok := seen[key]; !ok {
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	for _, key := range keys {
		child := joinPath(path, key)
		leftValue, leftOK := left[key]
		rightValue, rightOK := right[key]
		switch {
		case leftOK && rightOK:
			walkDiff(diff, child, leftValue, rightValue)
		case rightOK:
			diff.add(DiffEntry{Field: displayDiffPath(child), New: redactDiffValue(child, rightValue)})
		default:
			diff.remove(DiffEntry{Field: displayDiffPath(child), Old: redactDiffValue(child, leftValue)})
		}
	}
}

func walkSliceDiff(diff *ConfigDiff, path string, left, right []interface{}) {
	if keyedSlice(left) && keyedSlice(right) {
		walkNamedSliceDiff(diff, path, left, right)
		return
	}
	limit := len(left)
	if len(right) > limit {
		limit = len(right)
	}
	for i := 0; i < limit; i++ {
		child := fmt.Sprintf("%s[%d]", path, i)
		var leftValue, rightValue any
		if i < len(left) {
			leftValue = left[i]
		}
		if i < len(right) {
			rightValue = right[i]
		}
		if i >= len(left) {
			diff.add(DiffEntry{Field: displayDiffPath(child), New: redactDiffValue(child, rightValue)})
			continue
		}
		if i >= len(right) {
			diff.remove(DiffEntry{Field: displayDiffPath(child), Old: redactDiffValue(child, leftValue)})
			continue
		}
		walkDiff(diff, child, leftValue, rightValue)
	}
}

func walkNamedSliceDiff(diff *ConfigDiff, path string, left, right []interface{}) {
	leftByName := namedSliceMap(left)
	rightByName := namedSliceMap(right)
	names := make([]string, 0, len(leftByName)+len(rightByName))
	seen := map[string]struct{}{}
	for name := range leftByName {
		names = append(names, name)
		seen[name] = struct{}{}
	}
	for name := range rightByName {
		if _, ok := seen[name]; !ok {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	for _, name := range names {
		child := fmt.Sprintf("%s[%q]", path, name)
		leftValue, leftOK := leftByName[name]
		rightValue, rightOK := rightByName[name]
		switch {
		case leftOK && rightOK:
			walkDiff(diff, child, leftValue, rightValue)
		case rightOK:
			diff.add(DiffEntry{Field: displayDiffPath(child), New: redactDiffValue(child, rightValue)})
		default:
			diff.remove(DiffEntry{Field: displayDiffPath(child), Old: redactDiffValue(child, leftValue)})
		}
	}
}

func keyedSlice(items []interface{}) bool {
	if len(items) == 0 {
		return false
	}
	for _, item := range items {
		itemMap, ok := asStringMap(item)
		if !ok {
			return false
		}
		name, _ := itemMap["name"].(string)
		if strings.TrimSpace(name) == "" {
			return false
		}
	}
	return true
}

func namedSliceMap(items []interface{}) map[string]any {
	out := make(map[string]any, len(items))
	for _, item := range items {
		itemMap, ok := asStringMap(item)
		if !ok {
			continue
		}
		name, _ := itemMap["name"].(string)
		if name != "" {
			out[name] = item
		}
	}
	return out
}

func (d *ConfigDiff) accept() bool {
	return d != nil && d.size() < DiffEntryLimit
}

func (d *ConfigDiff) size() int {
	return len(d.Added) + len(d.Removed) + len(d.Changed)
}

func (d *ConfigDiff) add(entry DiffEntry) {
	if !d.accept() {
		d.Truncated = true
		return
	}
	d.Added = append(d.Added, entry)
}

func (d *ConfigDiff) remove(entry DiffEntry) {
	if !d.accept() {
		d.Truncated = true
		return
	}
	d.Removed = append(d.Removed, entry)
}

func (d *ConfigDiff) change(entry DiffEntry) {
	if !d.accept() {
		d.Truncated = true
		return
	}
	d.Changed = append(d.Changed, entry)
}

func redactDiffValue(path string, value any) any {
	if isSensitiveConfigKey(lastPathSegment(path)) {
		return RedactedConfigValue
	}
	return RedactSensitiveConfigValue(normalizeYAMLValue(value))
}

func lastPathSegment(path string) string {
	segment := path
	if index := strings.LastIndex(segment, "."); index >= 0 {
		segment = segment[index+1:]
	}
	if index := strings.Index(segment, "["); index >= 0 {
		segment = segment[:index]
	}
	return segment
}

func displayDiffPath(path string) string {
	if path == "" {
		return "."
	}
	return path
}

func asStringMap(value any) (map[string]interface{}, bool) {
	normalized := normalizeYAMLValue(value)
	typed, ok := normalized.(map[string]interface{})
	return typed, ok
}

func asSlice(value any) ([]interface{}, bool) {
	normalized := normalizeYAMLValue(value)
	typed, ok := normalized.([]interface{})
	return typed, ok
}

func normalizeYAMLValue(value any) any {
	switch typed := value.(type) {
	case map[string]interface{}:
		out := make(map[string]interface{}, len(typed))
		for key, nested := range typed {
			out[key] = normalizeYAMLValue(nested)
		}
		return out
	case map[interface{}]interface{}:
		return normalizeYAMLValue(nestedStringMap(typed))
	case []interface{}:
		out := make([]interface{}, len(typed))
		for i, nested := range typed {
			out[i] = normalizeYAMLValue(nested)
		}
		return out
	default:
		return typed
	}
}

func comparableYAMLValue(value any) any {
	switch typed := value.(type) {
	case int:
		return int64(typed)
	case int32:
		return int64(typed)
	case int64:
		return typed
	case uint:
		return int64(typed)
	case uint32:
		return int64(typed)
	case uint64:
		return int64(typed)
	case float32:
		return float64(typed)
	case map[string]interface{}, []interface{}:
		return normalizeYAMLValue(typed)
	default:
		return typed
	}
}
