import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { preloadDashboardRoute } from './app/routeLoaders'
import './index.css'
import './productDialog.css'
import 'highlight.js/styles/github-dark.css'

// Fetch only the requested route's static module while session/setup checks
// run. AppRouter still owns authorization and whether that page may render.
void preloadDashboardRoute(window.location.pathname)

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
