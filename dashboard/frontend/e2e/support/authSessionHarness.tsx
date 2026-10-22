import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from '../../src/contexts/AuthContext'
import AuthGate from '../../src/app/AuthGate'

// Mount the production provider and gate without unrelated page API fixtures.
export function SessionControls() {
  const { isAuthenticated, refreshSession } = useAuth()
  return (
    <>
      <output data-testid="session-state">
        {isAuthenticated ? 'Authenticated' : 'Unverified'}
      </output>
      <button
        onClick={() => {
          void refreshSession()
        }}
      >
        Check session
      </button>
    </>
  )
}

export function Workspace() {
  const { user } = useAuth()
  const [draft, setDraft] = useState('')
  return (
    <>
      <h1>Workspace for {user?.name}</h1>
      <label>
        Draft
        <textarea value={draft} onChange={(event) => setDraft(event.target.value)} />
      </label>
    </>
  )
}

createRoot(document.getElementById('root')!).render(
  <AuthProvider>
    <SessionControls />
    <MemoryRouter initialEntries={['/workspace']}>
      <Routes>
        <Route path="/login" element={<h1>Sign in</h1>} />
        <Route element={<AuthGate />}>
          <Route path="/workspace" element={<Workspace />} />
        </Route>
      </Routes>
    </MemoryRouter>
  </AuthProvider>,
)
