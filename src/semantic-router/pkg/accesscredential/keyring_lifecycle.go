package accesscredential

// Clone returns a process-owned copy of the pepper keyring. Callers retaining
// a keyring beyond construction must clone it so the source owner can erase or
// rotate its copy without changing the live service.
func (k PepperKeyring) Clone() PepperKeyring {
	return PepperKeyring{ActiveVersion: k.ActiveVersion, Keys: cloneKeyBytes(k.Keys)}
}

// Close erases the pepper key material owned by this keyring.
func (k *PepperKeyring) Close() {
	if k == nil {
		return
	}
	zeroKeyBytes(k.Keys)
	*k = PepperKeyring{}
}

// Clone returns a process-owned copy of the envelope keyring.
func (k KEKKeyring) Clone() KEKKeyring {
	return KEKKeyring{ActiveVersion: k.ActiveVersion, Keys: cloneKeyBytes(k.Keys)}
}

// Close erases the envelope key material owned by this keyring.
func (k *KEKKeyring) Close() {
	if k == nil {
		return
	}
	zeroKeyBytes(k.Keys)
	*k = KEKKeyring{}
}

func cloneKeyBytes(source map[string][]byte) map[string][]byte {
	if source == nil {
		return nil
	}
	cloned := make(map[string][]byte, len(source))
	for version, key := range source {
		cloned[version] = append([]byte(nil), key...)
	}
	return cloned
}

func zeroKeyBytes(keys map[string][]byte) {
	for version, key := range keys {
		zero(key)
		delete(keys, version)
	}
}
