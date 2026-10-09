package modelservice

import "testing"

func TestCPUThreadBudgetRespectsCapacityAndOperatorChoice(t *testing.T) {
	for _, test := range []struct {
		name     string
		cores    int
		override string
		want     int
	}{
		{name: "unknown capacity", cores: 0, want: 0},
		{name: "invalid capacity", cores: -1, want: 0},
		{name: "single core", cores: 1, want: 1},
		{name: "two cores", cores: 2, want: 1},
		{name: "odd small capacity", cores: 3, want: 1},
		{name: "small container", cores: 4, want: 2},
		{name: "odd capacity", cores: 7, want: 3},
		{name: "sixteen cores", cores: 16, want: 8},
		{name: "large host", cores: 128, want: 16},
		{name: "operator reduces concurrency cost", cores: 128, override: "2", want: 2},
		{name: "operator increases budget", cores: 128, override: "32", want: 32},
		{name: "quota bounds override", cores: 4, override: "32", want: 4},
		{name: "operator uses full small host", cores: 4, override: "4", want: 4},
		{name: "invalid override", cores: 32, override: "many", want: 16},
		{name: "zero override", cores: 32, override: "0", want: 16},
		{name: "negative override", cores: 32, override: "-4", want: 16},
		{name: "overflow override", cores: 32, override: "999999999999999999999999", want: 16},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv(CPUThreadsEnv, test.override)
			if got := cpuThreads(test.cores); got != test.want {
				t.Fatalf("cpuThreads(%d) with %q = %d, want %d", test.cores, test.override, got, test.want)
			}
		})
	}
}
