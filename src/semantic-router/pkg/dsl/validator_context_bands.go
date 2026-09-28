package dsl

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// checkContextSignalBands warns when context token bands partially overlap or
// leave a gap. Both are allowed: overlapping bands all match and are reported
// in declaration order, and a gap means requests in that range carry no
// context signal. A band that fully contains another is a common intentional
// layout and is not reported. Bands with invalid limits are reported by
// checkSignalConstraints.
func (v *Validator) checkContextSignalBands() {
	positions := make(map[string]Position)
	var bands []config.NamedContextBand
	for _, s := range v.prog.Signals {
		if s.SignalType != config.SignalTypeContext {
			continue
		}
		bounds, err := contextSignalBounds(s)
		if err != nil {
			continue
		}
		positions[s.Name] = s.Pos
		bands = append(bands, config.NamedContextBand{Name: s.Name, Bounds: bounds})
	}
	if len(bands) < 2 {
		return
	}

	overlaps, gaps := config.ContextBandIssues(bands)
	for _, overlap := range overlaps {
		if overlap.Contains {
			continue
		}
		v.addDiag(DiagWarning, positions[overlap.Inner.Name],
			fmt.Sprintf("SIGNAL context %s %s overlaps SIGNAL context %s %s; requests in the shared range match both",
				overlap.Inner.Name, overlap.Inner.Bounds, overlap.Outer.Name, overlap.Outer.Bounds),
			nil,
		)
	}
	for _, gap := range gaps {
		v.addDiag(DiagWarning, positions[gap.Before.Name],
			fmt.Sprintf("No context signal covers token counts %d to %d before SIGNAL context %s; requests in that range match no context signal",
				gap.From, gap.To, gap.Before.Name),
			nil,
		)
	}
}
