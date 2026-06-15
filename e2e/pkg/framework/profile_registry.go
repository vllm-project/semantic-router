package framework

import (
	"fmt"
	"sort"
	"sync"
)

// LocalImageBuild describes a locally built image a profile needs before setup.
type LocalImageBuild struct {
	Dockerfile   string
	Tag          string
	BuildContext string
}

// ProfileCapabilities declares runner-level behavior a profile requires.
type ProfileCapabilities struct {
	RequiresGPU bool
	LocalImages []LocalImageBuild
}

// ProfileRegistration is the self-registration contract for runnable profiles.
type ProfileRegistration struct {
	Name         string
	Factory      func() Profile
	Capabilities ProfileCapabilities
}

var (
	profileRegistryMu sync.RWMutex
	profileRegistry   = make(map[string]ProfileRegistration)
)

// RegisterProfile adds a profile factory to the global registry.
func RegisterProfile(reg ProfileRegistration) error {
	if reg.Name == "" {
		return fmt.Errorf("profile registration requires a name")
	}
	if reg.Factory == nil {
		return fmt.Errorf("profile %q registration requires a factory", reg.Name)
	}

	profileRegistryMu.Lock()
	defer profileRegistryMu.Unlock()

	if _, exists := profileRegistry[reg.Name]; exists {
		return fmt.Errorf("profile %q already registered", reg.Name)
	}

	profileRegistry[reg.Name] = reg
	return nil
}

// MustRegisterProfile adds a profile factory or panics.
func MustRegisterProfile(reg ProfileRegistration) {
	if err := RegisterProfile(reg); err != nil {
		panic(err)
	}
}

// LookupProfileRegistration returns metadata for a registered profile.
func LookupProfileRegistration(name string) (ProfileRegistration, bool) {
	profileRegistryMu.RLock()
	defer profileRegistryMu.RUnlock()

	reg, ok := profileRegistry[name]
	return reg, ok
}

// NewProfileByName constructs a registered profile by name.
func NewProfileByName(name string) (Profile, error) {
	reg, ok := LookupProfileRegistration(name)
	if !ok {
		return nil, fmt.Errorf("unknown profile: %s (available: %v)", name, RegisteredProfileNames())
	}
	return reg.Factory(), nil
}

// RegisteredProfileNames returns the sorted set of registered profile names.
func RegisteredProfileNames() []string {
	profileRegistryMu.RLock()
	defer profileRegistryMu.RUnlock()

	names := make([]string, 0, len(profileRegistry))
	for name := range profileRegistry {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
