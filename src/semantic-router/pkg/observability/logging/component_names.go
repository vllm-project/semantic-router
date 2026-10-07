package logging

import (
	"maps"
	"sync"
	"sync/atomic"
)

// renamedComponents maps a component to the name its events carry instead,
// for the whole process. A standalone Router names its routing core "router",
// since no Envoy ext_proc stream is involved.
var (
	renamedComponents atomic.Pointer[map[string]string]
	renameMu          sync.Mutex
)

// RenameComponent makes every later event of component carry name instead.
func RenameComponent(component, name string) {
	renameMu.Lock()
	defer renameMu.Unlock()
	next := map[string]string{}
	if current := renamedComponents.Load(); current != nil {
		maps.Copy(next, *current)
	}
	next[component] = name
	renamedComponents.Store(&next)
}

func componentName(component string) string {
	if names := renamedComponents.Load(); names != nil {
		if name, ok := (*names)[component]; ok {
			return name
		}
	}
	return component
}
