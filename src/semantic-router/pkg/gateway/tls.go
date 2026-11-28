package gateway

import (
	"crypto/tls"
	"os"
	"runtime"
	"sync"
	"time"
)

// keyPairCheckInterval bounds how often handshakes look at a listener's
// certificate files.
var keyPairCheckInterval = time.Second

// LoadTLS returns the TLS settings of a listener that serves the certificate
// chain in certFile with the private key in keyFile. Like Envoy's downstream
// TLS, it accepts TLS 1.2 and later.
//
// The pair reloads when either file changes, as a renewed Kubernetes secret or
// a cert-manager rotation changes them, so new connections get the new
// certificate without a restart. A pair that fails to load leaves the previous
// one serving. reloaded, when set, learns the outcome of each reload, and of a
// file that can no longer be read, once until it can again.
func LoadTLS(certFile, keyFile string, reloaded func(error)) (*tls.Config, error) {
	pair := &keyPair{certFile: certFile, keyFile: keyFile, reloaded: reloaded}
	stamp, err := pair.stamp()
	if err == nil {
		err = pair.load()
	}
	if err != nil {
		return nil, err
	}
	pair.attempted = stamp
	return &tls.Config{GetCertificate: pair.certificate, MinVersion: tls.VersionTLS12}, nil
}

type keyPair struct {
	certFile, keyFile string
	reloaded          func(error)

	mu         sync.Mutex
	current    *tls.Certificate
	attempted  [2]os.FileInfo
	unreadable bool
	checked    time.Time
}

func (p *keyPair) certificate(*tls.ClientHelloInfo) (*tls.Certificate, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if now := time.Now(); now.Sub(p.checked) >= keyPairCheckInterval {
		p.checked = now
		p.reloadIfChanged()
	}
	return p.current, nil
}

func (p *keyPair) reloadIfChanged() {
	stamp, err := p.stamp()
	if err != nil {
		if !p.unreadable {
			p.unreadable = true
			p.report(err)
		}
		return
	}
	p.unreadable = false
	if sameTLSFiles(stamp, p.attempted) {
		return
	}
	p.attempted = stamp
	p.report(p.load())
}

func (p *keyPair) load() error {
	cert, err := tls.LoadX509KeyPair(p.certFile, p.keyFile)
	if err != nil {
		return err
	}
	p.current = &cert
	return nil
}

func (p *keyPair) stamp() ([2]os.FileInfo, error) {
	var stamp [2]os.FileInfo
	for i, name := range []string{p.certFile, p.keyFile} {
		info, err := os.Stat(name)
		if err != nil {
			return stamp, err
		}
		// Windows loads identity on the first SameFile call. Capture it
		// while the path still names this version of the file.
		if runtime.GOOS == "windows" {
			_ = os.SameFile(info, info)
		}
		stamp[i] = info
	}
	return stamp, nil
}

func sameTLSFiles(current, previous [2]os.FileInfo) bool {
	for i, file := range current {
		old := previous[i]
		if file.Size() != old.Size() || !file.ModTime().Equal(old.ModTime()) || !os.SameFile(file, old) {
			return false
		}
	}
	return true
}

func (p *keyPair) report(err error) {
	if p.reloaded != nil {
		p.reloaded(err)
	}
}
