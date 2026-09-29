package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"text/tabwriter"
)

func main() {
	threshold := flag.Float64("threshold", 0.10,
		"minimum similarity; 0.10 is what the memory integration config pairs with deterministic embeddings")
	limit := flag.Int("limit", 5, "memories retrieved per probe, the router default")
	jsonPath := flag.String("json", "", "also write the full report, with every retrieved memory, to this path")
	flag.Parse()

	if err := run(*threshold, *limit, *jsonPath, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "memory-coldstart:", err)
		os.Exit(1)
	}
}

func run(threshold float64, limit int, jsonPath string, out io.Writer) error {
	if err := os.Setenv(deterministicEmbeddingsEnv, "1"); err != nil {
		return err
	}
	rep, err := replay(context.Background(), builtinScenario, replayOptions{
		Threshold: float32(threshold),
		Limit:     limit,
	})
	if err != nil {
		return err
	}
	if err = printReport(out, rep); err != nil {
		return err
	}
	if jsonPath == "" {
		return nil
	}
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(jsonPath, append(data, '\n'), 0o600)
}

func printReport(out io.Writer, rep report) error {
	w := tabwriter.NewWriter(out, 0, 0, 2, ' ', 0)
	fmt.Fprintf(w, "Router Memory cold-start replay: %d sessions over %d days, %d probes, %d memories stored\n",
		len(rep.Sessions), rep.Sessions[len(rep.Sessions)-1].Day, len(rep.Probes), rep.Memories)
	fmt.Fprintf(w, "in-memory store, %s embeddings, threshold %.2f, limit %d, default memory filter\n\n",
		rep.Embeddings, rep.Threshold, rep.Limit)

	fmt.Fprintln(w, "phase\tprobes\thit\tright\ttop-1\tstale\tungrounded")
	for _, p := range rep.Phases {
		fmt.Fprintf(w, "%s\t%d\t%s\t%s\t%s\t%s\t%s\n", p.Phase, p.Probes,
			ratio(p.Hit, p.Probes), ratio(p.Right, p.Probes), ratio(p.Top1, p.Probes),
			ratio(p.Stale, p.Probes), ratio(p.Ungrounded, p.Probes))
	}

	fmt.Fprintln(w, "\nday\tmemories\tprobes\thit\tright")
	for _, s := range rep.Sessions {
		fmt.Fprintf(w, "%d\t%d\t%d\t%s\t%s\n", s.Day, s.Memories, s.Probes,
			ratio(s.Hit, s.Probes), ratio(s.Right, s.Probes))
	}

	fmt.Fprintln(w, "\nday\tphase\thit\tright\ttop-1\tstale\tprobe")
	for _, p := range rep.Probes {
		fmt.Fprintf(w, "%d\t%s\t%s\t%s\t%s\t%s\t%s\n", p.Day, p.Phase,
			mark(p.Hit, "hit"), mark(p.Right, "right"), mark(p.Top1, "top-1"), mark(p.Stale, "stale"), p.Query)
		if note := probeNote(p); note != "" {
			fmt.Fprintf(w, "\t\t\t\t\t\t  %s\n", note)
		}
	}
	return w.Flush()
}

// probeNote names the memory behind a stale injection or a missed top rank.
// Session-window chunks hold several turns, so a stale memory is shown by the
// line that carries the superseded fact.
func probeNote(p probeResult) string {
	if p.Stale {
		for rank, r := range p.Retrieved {
			for _, line := range strings.Split(r.Content, "\n") {
				if containsAny(strings.ToLower(line), p.Superseded) {
					return fmt.Sprintf("stale at rank %d: %s", rank+1, line)
				}
			}
		}
	}
	if p.Hit && !p.Top1 {
		return "top: " + firstLine(p.Retrieved[0].Content)
	}
	return ""
}

func ratio(count, total int) string {
	return fmt.Sprintf("%d/%d", count, total)
}

func mark(set bool, label string) string {
	if set {
		return label
	}
	return "-"
}

func firstLine(content string) string {
	line, _, _ := strings.Cut(content, "\n")
	return line
}
