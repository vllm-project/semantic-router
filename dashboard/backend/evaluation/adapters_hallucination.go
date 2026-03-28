package evaluation

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// tryHallucinationMetricsFromOutputFile loads the latest results_*.json next to outputPath.
func tryHallucinationMetricsFromOutputFile(outputPath string) (map[string]interface{}, bool) {
	if outputPath == "" {
		return nil, false
	}
	dir := filepath.Dir(outputPath)
	latest := findLatestResultsJSONPath(dir)
	if latest == "" {
		return nil, false
	}
	data, err := os.ReadFile(latest)
	if err != nil {
		return nil, false
	}
	var result map[string]interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, false
	}
	return result, true
}

func findLatestResultsJSONPath(dir string) string {
	files, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var latestFile string
	var latestTime int64
	for _, f := range files {
		if !strings.HasPrefix(f.Name(), "results_") || !strings.HasSuffix(f.Name(), ".json") {
			continue
		}
		filePath := filepath.Join(dir, f.Name())
		info, infoErr := f.Info()
		if infoErr != nil {
			continue
		}
		modTime := info.ModTime().UnixNano()
		if modTime > latestTime {
			latestTime = modTime
			latestFile = filePath
		}
	}
	return latestFile
}

func parseHallucinationLineFromStdout(line string) (map[string]interface{}, bool) {
	line = strings.TrimSpace(line)
	if !strings.HasPrefix(line, "{") || !strings.HasSuffix(line, "}") {
		return nil, false
	}
	var result map[string]interface{}
	if err := json.Unmarshal([]byte(line), &result); err != nil {
		return nil, false
	}
	return result, true
}
