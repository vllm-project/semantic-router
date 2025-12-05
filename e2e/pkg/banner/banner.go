package banner

import (
	"fmt"
	"strings"
	"time"
)

// Color codes
const (
	Reset  = "\033[0m"
	Cyan   = "\033[36m"
	Purple = "\033[35m"
	Green  = "\033[32m"
	Yellow = "\033[33m"
	Bold   = "\033[1m"
)

// ASCII art for vLLM Semantic Router
var logo = []string{
	"        ██╗   ██╗██╗     ██╗     ███╗   ███╗",
	"        ██║   ██║██║     ██║     ████╗ ████║",
	"        ██║   ██║██║     ██║     ██╔████╔██║",
	"        ╚██╗ ██╔╝██║     ██║     ██║╚██╔╝██║",
	"         ╚████╔╝ ███████╗███████╗██║ ╚═╝ ██║",
	"          ╚═══╝  ╚══════╝╚══════╝╚═╝     ╚═╝",
	"",
	"      ███████╗███████╗███╗   ███╗ █████╗ ███╗   ██╗████████╗██╗ ██████╗",
	"      ██╔════╝██╔════╝████╗ ████║██╔══██╗████╗  ██║╚══██╔══╝██║██╔════╝",
	"      ███████╗█████╗  ██╔████╔██║███████║██╔██╗ ██║   ██║   ██║██║     ",
	"      ╚════██║██╔══╝  ██║╚██╔╝██║██╔══██║██║╚██╗██║   ██║   ██║██║     ",
	"      ███████║███████╗██║ ╚═╝ ██║██║  ██║██║ ╚████║   ██║   ██║╚██████╗",
	"      ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝",
	"",
	"       ██████╗  ██████╗ ██╗   ██╗████████╗███████╗██████╗",
	"       ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝██╔══██╗",
	"       ██████╔╝██║   ██║██║   ██║   ██║   █████╗  ██████╔╝",
	"       ██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝  ██╔══██╗",
	"       ██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗██║  ██║",
	"       ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝",
}

// Show displays the animated banner
func Show(version string) {
	fmt.Println()
	fmt.Println()

	// Animate logo line by line
	for _, line := range logo {
		fmt.Printf("%s%s%s%s\n", Bold, Cyan, line, Reset)
		time.Sleep(30 * time.Millisecond)
	}

	fmt.Println()

	// Show version and info with animation
	info := []string{
		fmt.Sprintf("%s%s                    E2E Testing Framework%s", Bold, Purple, Reset),
		fmt.Sprintf("%s                    Version: %s%s", Yellow, version, Reset),
		"",
	}

	for _, line := range info {
		fmt.Println(line)
		time.Sleep(50 * time.Millisecond)
	}

	// Loading animation
	loadingChars := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
	fmt.Print(Green + "                    Initializing" + Reset)
	for i := 0; i < 15; i++ {
		fmt.Printf("\r%s                    Initializing %s%s", Green, loadingChars[i%len(loadingChars)], Reset)
		time.Sleep(80 * time.Millisecond)
	}
	fmt.Printf("\r%s                    Initializing ✓%s\n", Green, Reset)

	fmt.Println()
	fmt.Println(strings.Repeat("═", 80))
	fmt.Println()
}

// ShowQuick displays a quick version without animation (for CI/non-interactive)
func ShowQuick(version string) {
	fmt.Println()
	fmt.Println()

	for _, line := range logo {
		fmt.Printf("%s%s%s%s\n", Bold, Cyan, line, Reset)
	}

	fmt.Println()
	fmt.Printf("%s%s                    E2E Testing Framework%s\n", Bold, Purple, Reset)
	fmt.Printf("%s                    Version: %s%s\n", Yellow, version, Reset)
	fmt.Println()
	fmt.Println(strings.Repeat("═", 80))
	fmt.Println()
}
