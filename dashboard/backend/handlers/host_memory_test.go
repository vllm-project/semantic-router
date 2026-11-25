package handlers

import (
	"os"
	"path/filepath"
	"testing"
)

func writeHostFile(t *testing.T, root, name, content string) {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

const meminfo8GiB = "MemTotal:        8074240 kB\nMemFree:         1000000 kB\n"

func TestHostMemoryBytes(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name  string
		files map[string]string
		want  int64
	}{
		{"nothing readable", nil, 0},
		{"meminfo only", map[string]string{"proc/meminfo": meminfo8GiB}, 8074240 * 1024},
		{
			"unlimited cgroup v2 keeps MemTotal",
			map[string]string{"proc/meminfo": meminfo8GiB, "sys/fs/cgroup/memory.max": "max\n"},
			8074240 * 1024,
		},
		{
			"cgroup v2 limit below MemTotal wins",
			map[string]string{"proc/meminfo": meminfo8GiB, "sys/fs/cgroup/memory.max": "4294967296\n"},
			4294967296,
		},
		{
			"cgroup v1 unlimited sentinel is undercut by MemTotal",
			map[string]string{
				"proc/meminfo": meminfo8GiB,
				"sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
			},
			8074240 * 1024,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			root := t.TempDir()
			for name, content := range tc.files {
				writeHostFile(t, root, name, content)
			}
			if got := hostMemoryBytesAt(root); got != tc.want {
				t.Fatalf("hostMemoryBytesAt() = %d, want %d", got, tc.want)
			}
		})
	}
}
