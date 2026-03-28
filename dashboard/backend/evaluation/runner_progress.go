package evaluation

import (
	"fmt"
	"strings"
)

// tqdmPercentFromLine extracts a percentage from tqdm-style progress output.
func tqdmPercentFromLine(line string) (percent int, ok bool) {
	if !strings.Contains(line, "%|") && !strings.Contains(line, "% |") {
		return 0, false
	}
	for i := 0; i < len(line); i++ {
		if line[i] != '%' || i == 0 {
			continue
		}
		start := i - 1
		for start > 0 && line[start-1] >= '0' && line[start-1] <= '9' {
			start--
		}
		if start >= i {
			continue
		}
		var p int
		if _, scanErr := fmt.Sscanf(line[start:i], "%d", &p); scanErr != nil {
			continue
		}
		if p > 0 && p <= 100 {
			return p, true
		}
	}
	return 0, false
}
