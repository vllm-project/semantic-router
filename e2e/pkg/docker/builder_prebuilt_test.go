package docker

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrebuiltImageIsInspectedTaggedAndLoadedWithoutBuild(t *testing.T) {
	for _, missing := range []bool{false, true} {
		t.Run(map[bool]string{false: "available", true: "missing"}[missing], func(t *testing.T) {
			directory := t.TempDir()
			log := filepath.Join(directory, "commands")
			t.Setenv("COMMAND_LOG", log)
			t.Setenv("PATH", directory+string(os.PathListSeparator)+os.Getenv("PATH"))
			script := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$COMMAND_LOG\"\n"
			if missing {
				script += "if [ \"$1 $2\" = 'image inspect' ]; then exit 1; fi\n"
			}
			for _, name := range []string{"docker", "kind"} {
				if err := os.WriteFile(filepath.Join(directory, name), []byte(script), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			err := NewBuilder(false).LoadPrebuilt(context.Background(), "owned-cluster", "verified:sha", "profile:test")
			raw, readErr := os.ReadFile(log)
			if readErr != nil {
				t.Fatal(readErr)
			}
			commands := string(raw)
			if missing {
				if err == nil || strings.Contains(commands, "tag ") || strings.Contains(commands, "load ") {
					t.Fatalf("missing image was consumed: %v %s", err, commands)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			for _, want := range []string{"image inspect verified:sha", "tag verified:sha profile:test", "load docker-image profile:test --name owned-cluster"} {
				if !strings.Contains(commands, want) {
					t.Fatalf("missing %q: %s", want, commands)
				}
			}
			if strings.Contains(commands, "build") {
				t.Fatal("prebuilt path rebuilt image")
			}
		})
	}
}
