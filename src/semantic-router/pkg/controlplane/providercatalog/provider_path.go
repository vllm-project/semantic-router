package providercatalog

import (
	"net/url"
	"strings"
)

// compiledPathForOrigin keeps Integration paths complete while avoiding API
// base segments that are already present at the end of an origin. The runtime
// appends the returned path to the origin verbatim.
func compiledPathForOrigin(origin, integrationPath string) string {
	parsed, err := url.Parse(origin)
	if err != nil {
		return integrationPath
	}
	originSegments := pathSegments(parsed.Path)
	integrationSegments := pathSegments(integrationPath)
	maximumOverlap := min(len(originSegments), len(integrationSegments)-1)
	for overlap := maximumOverlap; overlap > 0; overlap-- {
		originStart := len(originSegments) - overlap
		matched := true
		for index := 0; index < overlap; index++ {
			if originSegments[originStart+index] != integrationSegments[index] {
				matched = false
				break
			}
		}
		if matched {
			return "/" + strings.Join(integrationSegments[overlap:], "/")
		}
	}
	return integrationPath
}

func pathSegments(value string) []string {
	trimmed := strings.Trim(value, "/")
	if trimmed == "" {
		return nil
	}
	return strings.Split(trimmed, "/")
}
