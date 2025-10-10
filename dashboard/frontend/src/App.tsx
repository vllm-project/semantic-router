import React from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import MonitoringPage from './pages/MonitoringPage'
import ConfigPage from './pages/ConfigPage'
import PlaygroundPage from './pages/PlaygroundPage'

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/monitoring" replace />} />
          <Route path="/monitoring" element={<MonitoringPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/playground" element={<PlaygroundPage />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}

export default App
