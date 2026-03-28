# Dashboard Frontend Shell Notes

## Scope

- `dashboard/frontend/src/**`

## Responsibilities

- Keep the frontend app shell focused on route registration, auth/setup gating, shared providers, and top-level layout composition.
- Keep page-specific data shaping, config-section mapping, and interaction logic out of `App.tsx` when a page or sibling support module can own it.
- Let deeper local rules under `pages/` and `components/` own page and container hotspots; this file covers the shared route-shell boundary above them.

## Change Rules

- `App.tsx` is the route-shell hotspot. Keep auth/setup routing and shared layout composition there, but move page-specific maps, repeated route wrappers, and feature-local helpers into sibling support modules before the file grows further.
- Do not add config-page helpers, overview fetch logic, or chat/editor support code into the app shell.
- If a change touches auth/setup gating and one page or container hotspot at once, treat that as a design smell and look for a narrower route-support seam first.
