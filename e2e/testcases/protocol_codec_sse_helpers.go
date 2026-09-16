package testcases

import (
	"fmt"
	"strings"
)

func protocolSSEDataFrames(body []byte) []string {
	frames := strings.Split(string(body), "\n\n")
	dataFrames := make([]string, 0, len(frames))
	for _, frame := range frames {
		if data := protocolSSEFrameData(frame); data != "" {
			dataFrames = append(dataFrames, data)
		}
	}
	return dataFrames
}

func protocolSSEFrameData(frame string) string {
	for _, line := range strings.Split(frame, "\n") {
		if strings.HasPrefix(line, "data:") {
			return strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
	}
	return ""
}

func rejectStreamFragments(stream, protocol string, fragments []string) error {
	for _, fragment := range fragments {
		if strings.Contains(stream, fragment) {
			return fmt.Errorf(
				"%s stream leaked backend protocol fragment %q: %s",
				protocol,
				fragment,
				truncateString(stream, 1200),
			)
		}
	}
	return nil
}

func validateOrderedStreamMarkers(stream string, markers []string) error {
	previousIndex := -1
	for _, marker := range markers {
		index := strings.Index(stream, marker)
		if index < 0 {
			return fmt.Errorf("missing stream marker %q: %s", marker, truncateString(stream, 1200))
		}
		if index < previousIndex {
			return fmt.Errorf("stream marker %q is out of order: %s", marker, truncateString(stream, 1200))
		}
		previousIndex = index
	}
	return nil
}
