// +build ignore

// generate_source_dataset reads Go source files under pkg/ and produces
// evaluation_dataset.json with real source code chunks as entries.
//
// Usage:
//   go run testdata/generate_source_dataset.go [-out testdata/evaluation_source.json] [-chunksize 300] [-max 120]
//
// Each source file is split into chunks of ~chunksize characters at function
// boundaries (or line boundaries). The package directory is the cluster label.
// Queries are auto-generated: one per cluster asking about that package's purpose.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type entry struct {
	ID      string `json:"id"`
	Cluster string `json:"cluster"`
	Content string `json:"content"`
}

type queryDef struct {
	Query         string `json:"query"`
	TargetCluster string `json:"target_cluster"`
}

type config struct {
	K              int       `json:"k"`
	Threshold      float32   `json:"threshold"`
	Alphas         []float32 `json:"alphas"`
	MaxDepth       int       `json:"max_depth"`
	ScorePropAlpha float32   `json:"score_prop_alpha"`
}

type dataset struct {
	Description string     `json:"description"`
	Config      config     `json:"config"`
	Entries     []entry    `json:"entries"`
	Queries     []queryDef `json:"queries"`
}

func main() {
	outPath := flag.String("out", "testdata/evaluation_source.json", "output JSON path")
	chunkSize := flag.Int("chunksize", 400, "target chunk size in characters")
	maxEntries := flag.Int("max", 120, "max total entries (sampled evenly across clusters)")
	srcRoot := flag.String("src", ".", "root directory containing pkg/")
	flag.Parse()

	pkgDir := filepath.Join(*srcRoot, "pkg")

	// Collect all .go files (non-test) grouped by package.
	byPkg := map[string][]string{}
	err := filepath.Walk(pkgDir, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return err
		}
		if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
			return nil
		}
		rel, _ := filepath.Rel(pkgDir, path)
		pkg := filepath.Dir(rel)
		byPkg[pkg] = append(byPkg[pkg], path)
		return nil
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "walk error: %v\n", err)
		os.Exit(1)
	}

	// Sort package names for determinism.
	var pkgs []string
	for p := range byPkg {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)

	// Chunk files into entries.
	var allEntries []entry
	for _, pkg := range pkgs {
		files := byPkg[pkg]
		sort.Strings(files)
		for _, fpath := range files {
			chunks := chunkFile(fpath, *chunkSize)
			base := strings.TrimSuffix(filepath.Base(fpath), ".go")
			for ci, c := range chunks {
				allEntries = append(allEntries, entry{
					ID:      fmt.Sprintf("%s/%s-%d", pkg, base, ci),
					Cluster: pkg,
					Content: c,
				})
			}
		}
	}

	fmt.Fprintf(os.Stderr, "Total chunks before sampling: %d across %d packages\n", len(allEntries), len(pkgs))

	// Sample evenly across clusters.
	sampled := sampleEvenly(allEntries, pkgs, *maxEntries)
	fmt.Fprintf(os.Stderr, "Sampled %d entries across %d packages\n", len(sampled), len(pkgs))

	// Count entries per cluster for queries.
	clusterCounts := map[string]int{}
	for _, e := range sampled {
		clusterCounts[e.Cluster]++
	}

	// Generate one query per cluster that has entries.
	var queries []queryDef
	for _, pkg := range pkgs {
		if clusterCounts[pkg] == 0 {
			continue
		}
		queries = append(queries, queryDef{
			Query:         fmt.Sprintf("How does the %s package work and what are its main functions?", pkg),
			TargetCluster: pkg,
		})
	}

	ds := dataset{
		Description: fmt.Sprintf("Auto-generated from %d Go source files under pkg/. %d chunks sampled across %d packages.",
			countFiles(byPkg), len(sampled), len(clusterCounts)),
		Config: config{
			K:              5,
			Threshold:      0.3,
			Alphas:         []float32{1.0, 0.7, 0.5, 0.3},
			MaxDepth:       3,
			ScorePropAlpha: 0.6,
		},
		Entries: sampled,
		Queries: queries,
	}

	data, err := json.MarshalIndent(ds, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal error: %v\n", err)
		os.Exit(1)
	}

	if err := os.WriteFile(*outPath, data, 0644); err != nil {
		fmt.Fprintf(os.Stderr, "write error: %v\n", err)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "Wrote %s (%d entries, %d queries)\n", *outPath, len(sampled), len(queries))
}

// chunkFile splits a Go source file into chunks at function boundaries.
func chunkFile(path string, targetSize int) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	content := string(data)
	lines := strings.Split(content, "\n")

	var chunks []string
	var current strings.Builder

	for _, line := range lines {
		current.WriteString(line)
		current.WriteString("\n")

		// Split at function boundaries when we exceed target size.
		if current.Len() >= targetSize && isBoundary(line) {
			text := strings.TrimSpace(current.String())
			if text != "" {
				chunks = append(chunks, text)
			}
			current.Reset()
		}
	}

	if current.Len() > 0 {
		text := strings.TrimSpace(current.String())
		if text != "" {
			if len(chunks) > 0 && len(text) < targetSize/3 {
				// Merge short tail into last chunk.
				chunks[len(chunks)-1] += "\n" + text
			} else {
				chunks = append(chunks, text)
			}
		}
	}

	return chunks
}

func isBoundary(line string) bool {
	trimmed := strings.TrimSpace(line)
	return trimmed == "}" || trimmed == "" || strings.HasPrefix(trimmed, "func ") || strings.HasPrefix(trimmed, "// ")
}

func sampleEvenly(entries []entry, pkgs []string, max int) []entry {
	if len(entries) <= max {
		return entries
	}

	byCluster := map[string][]entry{}
	for _, e := range entries {
		byCluster[e.Cluster] = append(byCluster[e.Cluster], e)
	}

	nClusters := len(byCluster)
	perCluster := max / nClusters
	if perCluster < 1 {
		perCluster = 1
	}

	var result []entry
	for _, pkg := range pkgs {
		items := byCluster[pkg]
		if len(items) == 0 {
			continue
		}
		n := perCluster
		if n > len(items) {
			n = len(items)
		}
		// Take evenly spaced samples.
		step := len(items) / n
		if step < 1 {
			step = 1
		}
		for i := 0; i < n && i*step < len(items); i++ {
			result = append(result, items[i*step])
		}
	}

	return result
}

func countFiles(m map[string][]string) int {
	n := 0
	for _, v := range m {
		n += len(v)
	}
	return n
}
