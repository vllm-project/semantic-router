package logo

import (
	"fmt"
)

// ANSI color codes
const (
	colorReset = "\033[0m"
	colorWhite = "\033[97m"
	colorMuted = "\033[38;2;145;158;171m"
)

func buildVLLMLogoLines() []string {
	return []string{
		"",
		colorWhite + `       █     █     █▄   ▄█` + colorReset,
		colorWhite + ` ▄▄ ▄█ █     █     █ ▀▄▀ █` + colorReset,
		colorWhite + `  █▄█▀ █     █     █     █` + colorReset,
		colorWhite + `   ▀▀  ▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀` + colorReset,
		colorWhite + `  Semantic Router` + colorReset,
		colorMuted + `  local runtime` + colorReset,
		"",
	}
}

// PrintVLLMLogo prints the vLLM Semantic Router serve banner.
func PrintVLLMLogo() {
	for _, line := range buildVLLMLogoLines() {
		fmt.Println(line)
	}
}
