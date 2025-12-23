package logo

import (
	"fmt"
)

// ANSI color codes
const (
	colorReset  = "\033[0m"
	colorOrange = "\033[38;2;254;181;22m" // #FEB516 - vLLM V left side
	colorBlue   = "\033[38;2;48;162;255m" // #30A2FF - vLLM V right side
	colorWhite  = "\033[97m"              // White - for LLM and SR
)

// PrintVLLMLogo prints the vLLM SR logo with colors
func PrintVLLMLogo() {
	// Logo design: vLLM SR
	// v = left side orange, right side blue
	// LLM SR = white
	logo := []string{
		"",
		colorOrange + `##` + colorWhite + `          ` + colorBlue + `##` + colorWhite + `  ##        ##        ##      ##    ######    ########` + colorReset,
		colorOrange + ` ##` + colorWhite + `        ` + colorBlue + `##` + colorWhite + `   ##        ##        ###    ###   ##    ##   ##    ##` + colorReset,
		colorOrange + `  ##` + colorWhite + `      ` + colorBlue + `##` + colorWhite + `    ##        ##        ####  ####   ##         ##    ##` + colorReset,
		colorOrange + `   ##` + colorWhite + `    ` + colorBlue + `##` + colorWhite + `     ##        ##        ## #### ##    ####     ########` + colorReset,
		colorOrange + `    ##` + colorWhite + `  ` + colorBlue + `##` + colorWhite + `      ##        ##        ##  ##  ##       ##    ##  ##` + colorReset,
		colorOrange + `     ##` + colorBlue + `##` + colorWhite + `       ##        ##        ##      ##   ##    ##   ##   ##` + colorReset,
		colorOrange + `      ` + colorBlue + `##` + colorWhite + `        ########  ########  ##      ##    ######    ##    ##` + colorReset,
		"",
	}

	for _, line := range logo {
		fmt.Println(line)
	}
}
