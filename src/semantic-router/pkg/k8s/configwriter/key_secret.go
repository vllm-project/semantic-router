package configwriter

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// KeySecretNameEnv names the Secret, in the ConfigMap's namespace, that keeps
// the keys a ConfigMap save moves out of the configuration. The chart loads
// its entries into the Router's and the Dashboard's environment, so the saved
// configuration names each key by its entry: auth.api_key_env for a backend,
// a "${NAME}" entry in api_keys for a listener. A pod reads them when it
// starts, which is when a ConfigMap save takes effect.
const KeySecretNameEnv = "VLLM_SR_K8S_KEY_SECRET_NAME" //nolint:gosec // G101: a variable name, not a credential.

// KeySecretEntryPrefix begins the name of each key the Secret holds, which is
// also the environment variable a configuration reads it from.
const KeySecretEntryPrefix = "VSR_KEY_"

const keySecretTimeout = 10 * time.Second

// envName is an entry name that envFrom loads as an environment variable.
var envName = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// KeySecret is the config.KeyStore of a key Secret. A Secret that is missing,
// or that this pod may not read and update, takes no key: the key stays in
// the ConfigMap as written, and the save warns.
type KeySecret struct {
	Client    kubernetes.Interface
	Namespace string
	Name      string
}

// Ref returns the entry that holds key, adding one when create is set.
func (s KeySecret) Ref(key string, create bool) (config.KeyRef, bool, error) {
	var entry string
	created := false
	err := s.update(func(data map[string][]byte) bool {
		for _, name := range slices.Sorted(maps.Keys(data)) {
			if envName.MatchString(name) && strings.TrimSpace(string(data[name])) == key {
				entry = name
				return false
			}
		}
		if entry, created = "", create; !create {
			return false
		}
		suffix := make([]byte, 8)
		_, _ = rand.Read(suffix) // crypto/rand.Read never fails since Go 1.24.
		entry = KeySecretEntryPrefix + strings.ToUpper(hex.EncodeToString(suffix))
		data[entry] = []byte(key)
		return true
	})
	if apierrors.IsNotFound(err) || apierrors.IsForbidden(err) {
		return config.KeyRef{}, false, nil
	}
	if err != nil || entry == "" {
		return config.KeyRef{}, false, err
	}
	return config.KeyRef{Env: entry}, created, nil
}

// Remove deletes an entry Ref added.
func (s KeySecret) Remove(ref config.KeyRef) error {
	return s.update(func(data map[string][]byte) bool {
		_, found := data[ref.Env]
		delete(data, ref.Env)
		return found
	})
}

// update applies change to the Secret's data, and writes the Secret when
// change reports that it changed the data.
func (s KeySecret) update(change func(map[string][]byte) bool) error {
	ctx, cancel := context.WithTimeout(context.Background(), keySecretTimeout)
	defer cancel()
	secrets := s.Client.CoreV1().Secrets(s.Namespace)
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		current, err := secrets.Get(ctx, s.Name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("get key Secret %s/%s: %w", s.Namespace, s.Name, err)
		}
		updated := current.DeepCopy()
		if updated.Data == nil {
			updated.Data = map[string][]byte{}
		}
		if !change(updated.Data) {
			return nil
		}
		if _, err = secrets.Update(ctx, updated, metav1.UpdateOptions{}); err != nil {
			return fmt.Errorf("update key Secret %s/%s: %w", s.Namespace, s.Name, err)
		}
		return nil
	})
}

// KeyStore is where a save of the configuration at configPath keeps the keys
// it moves out of the document: the key Secret on a ConfigMap target, files
// in the directory dir beside the configuration file otherwise, shared with
// the group config.KeyFileGroupEnv names. A ConfigMap target without a key
// Secret keeps every key as written.
func KeyStore(configPath, dir string, writer func() (*ConfigMapWriter, error)) (config.KeyStore, error) {
	if _, ok := ConfigMapTargetFromEnv(); !ok {
		store := config.KeyDir{Dir: filepath.Join(filepath.Dir(configPath), dir)}
		if group := strings.TrimSpace(os.Getenv(config.KeyFileGroupEnv)); group != "" {
			gid, err := strconv.Atoi(group)
			if err != nil || gid <= 0 {
				return nil, fmt.Errorf("%s must be a group id, not %q", config.KeyFileGroupEnv, group)
			}
			store.Group = gid
		}
		return store, nil
	}
	name := strings.TrimSpace(os.Getenv(KeySecretNameEnv))
	if name == "" {
		return keptKeys{}, nil
	}
	resolved, err := writer()
	if err != nil {
		return nil, fmt.Errorf("resolve key Secret client: %w", err)
	}
	return KeySecret{Client: resolved.clientset, Namespace: strings.TrimSpace(os.Getenv(ConfigMapNamespaceEnv)), Name: name}, nil
}

// keptKeys takes no key.
type keptKeys struct{}

func (keptKeys) Ref(string, bool) (config.KeyRef, bool, error) { return config.KeyRef{}, false, nil }
func (keptKeys) Remove(config.KeyRef) error                    { return nil }
