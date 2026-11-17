// Applies the saved or system theme before the first paint (no inline script, CSP friendly).
;(function () {
  try {
    var pref = localStorage.getItem('vsr-theme') || 'system'
    var dark = pref === 'dark' || (pref === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)
    if (dark) document.documentElement.classList.add('dark')
    document.documentElement.dataset.accent = localStorage.getItem('vsr-accent') === 'gold' ? 'gold' : 'blue'
  } catch (e) {}
})()
