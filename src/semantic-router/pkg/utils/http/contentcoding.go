package http

import "strings"

// ContentCoding reads a Content-Encoding field's values, every field line in
// order, as the one content coding a reader would undo: "" for none or
// identity, "gzip" for gzip or its x-gzip alias, and any other coding
// lowercased. single is false when the values name more than one coding, a
// stack that a reader undoing one layer cannot decode.
func ContentCoding(values []string) (coding string, single bool) {
	count := 0
	for _, value := range values {
		for _, entry := range strings.Split(value, ",") {
			entry = strings.ToLower(strings.TrimSpace(entry))
			if entry == "" || entry == "identity" {
				continue
			}
			if entry == "x-gzip" {
				entry = "gzip"
			}
			coding = entry
			count++
		}
	}
	if count > 1 {
		return "", false
	}
	return coding, true
}
