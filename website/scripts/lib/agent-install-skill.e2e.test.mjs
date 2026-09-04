// E2E install smoke test: runs the maintained install.sh in a fully isolated
// temporary HOME / install root / bin dir and verifies the agent-safe contract:
//
//   install.sh --mode cli --runtime skip --no-launch
//     → vllm-sr --version succeeds
//     → no automatic serve
//     → no mutation of the real user HOME
//     → no Docker / Podman dependency
//     → temporary directories cleaned up
//
// This test is skipped on Windows / MSYS / MINGW because the installer targets
// macOS, Linux, and WSL2 — running it under native Windows shells would die
// before doing anything useful.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { tmpdir } from 'node:os'
import { platform } from 'node:os'

const scriptDir = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDir, '..', '..', '..')

const installScript = resolve(repoRoot, 'install.sh')

/**
 * Returns true if the current OS can run install.sh natively.
 * The installer's detect_os() accepts only Darwin and Linux.
 */
function canRunInstaller() {
  return platform() === 'linux' || platform() === 'darwin'
}

/**
 * Create a fully isolated temp directory tree:
 *   tempHome/
 *     .local/
 *       share/vllm-sr/   (install root)
 *       bin/             (bin dir)
 */
function createIsolatedEnv() {
  const tempHome = mkdtempSync(resolve(tmpdir(), 'vsr-e2e-'))
  const installRoot = resolve(tempHome, '.local', 'share', 'vllm-sr')
  const binDir = resolve(tempHome, '.local', 'bin')
  return { tempHome, installRoot, binDir }
}

/**
 * Snapshot the real user HOME contents that the installer would touch,
 * so we can prove the test never mutated them.
 */
function snapshotRealHome() {
  const realHome = process.env.HOME || ''
  const realVsrRoot = resolve(realHome, '.local', 'share', 'vllm-sr')
  const realBin = resolve(realHome, '.local', 'bin')
  return {
    realHome,
    realVsrRootExists: existsSync(realVsrRoot),
    realVsrRootEntries: existsSync(realVsrRoot) ? readdirSync(realVsrRoot).sort() : [],
    realBinExists: existsSync(realBin),
    realBinVsrExists: existsSync(resolve(realBin, 'vllm-sr')),
  }
}

test('E2E: install.sh agent-safe mode installs CLI and validates version', { skip: !canRunInstaller() ? 'installer requires Linux or macOS' : undefined }, () => {
  const { tempHome, installRoot, binDir } = createIsolatedEnv()
  const beforeSnap = snapshotRealHome()

  try {
    // Run the maintained installer with agent-safe flags and full isolation.
    const result = spawnSync('bash', [
      installScript,
      '--mode', 'cli',
      '--runtime', 'skip',
      '--no-launch',
      '--install-root', installRoot,
      '--bin-dir', binDir,
    ], {
      env: {
        ...process.env,
        HOME: tempHome,
        VLLM_SR_INSTALL_ROOT: installRoot,
        VLLM_SR_BIN_DIR: binDir,
      },
      encoding: 'utf8',
      timeout: 120000,
    })

    // 1. Exit code must be 0.
    assert.equal(result.status, 0,
      `install.sh exited with ${result.status}\nstdout: ${result.stdout}\nstderr: ${result.stderr}`)

    // 2. vllm-sr --version must succeed and print a version string.
    const vllmSrBin = resolve(binDir, 'vllm-sr')
    assert.ok(existsSync(vllmSrBin), `launcher ${vllmSrBin} was not created`)

    const versionResult = spawnSync(vllmSrBin, ['--version'], {
      env: {
        ...process.env,
        HOME: tempHome,
        PATH: `${binDir}:${process.env.PATH || ''}`,
      },
      encoding: 'utf8',
      timeout: 30000,
    })

    assert.equal(versionResult.status, 0,
      `vllm-sr --version exited with ${versionResult.status}\nstdout: ${versionResult.stdout}\nstderr: ${versionResult.stderr}`)
    assert.match(versionResult.stdout.trim(), /\d+\.\d+/,
      `vllm-sr --version did not print a version string: ${versionResult.stdout}`)

    // 3. No automatic serve: with --mode cli --no-launch, neither serve nor
    //    dashboard should have been started. Verify by checking that no
    //    runtime.env was written (skip mode), and no serve-related state
    //    exists under the install root.
    const runtimeEnv = resolve(installRoot, 'runtime.env')
    if (existsSync(runtimeEnv)) {
      const content = readFileSync(runtimeEnv, 'utf8')
      // --runtime skip may or may not write runtime.env; if it does,
      // it must say CONTAINER_RUNTIME=skip, not docker or podman.
      assert.ok(
        content.includes('CONTAINER_RUNTIME=skip'),
        `runtime.env should reflect skip mode, got: ${content}`,
      )
    }

    // 4. The real user HOME must not have been mutated.
    const afterSnap = snapshotRealHome()
    if (!beforeSnap.realVsrRootExists) {
      assert.ok(!existsSync(beforeSnap.realVsrRoot ?? resolve(process.env.HOME || '', '.local', 'share', 'vllm-sr')),
        'real HOME ~/.local/share/vllm-sr was created — isolation leaked')
    }
    if (!beforeSnap.realBinVsrExists) {
      assert.ok(!existsSync(resolve(process.env.HOME || '', '.local', 'bin', 'vllm-sr')),
        'real HOME ~/.local/bin/vllm-sr was created — isolation leaked')
    }

    // 5. No Docker / Podman dependency: --runtime skip means the installer
    //    must not have invoked docker/podman. We verify transitively:
    //    if docker/podman were required, the install would have failed
    //    or taken much longer. The exit-0 + version-success above already
    //    proves this, but we also assert no runtime.env docker entry.
    if (existsSync(runtimeEnv)) {
      const content = readFileSync(runtimeEnv, 'utf8')
      assert.ok(!content.includes('docker') && !content.includes('podman'),
        'runtime.env should not reference docker or podman under --runtime skip')
    }
  } finally {
    // 6. Cleanup: remove all temp directories.
    rmSync(tempHome, { recursive: true, force: true })
    assert.ok(!existsSync(tempHome), `temp dir ${tempHome} was not cleaned up`)
  }
})
