import type { ReactNode, SVGProps } from 'react'

export type ProductIconName =
  | 'activity'
  | 'alert'
  | 'arrow-left'
  | 'arrow-right'
  | 'attachment'
  | 'audit'
  | 'budget'
  | 'chart'
  | 'check'
  | 'chevron-left'
  | 'chevron-down'
  | 'chevron-right'
  | 'chevron-up'
  | 'claw'
  | 'close'
  | 'code'
  | 'compute'
  | 'copy'
  | 'dashboard'
  | 'database'
  | 'decision'
  | 'download'
  | 'edit'
  | 'evaluation'
  | 'eye'
  | 'eye-off'
  | 'expand'
  | 'fleet'
  | 'fullscreen'
  | 'globe'
  | 'inbox'
  | 'info'
  | 'insight'
  | 'key'
  | 'label'
  | 'logs'
  | 'minus'
  | 'mixture'
  | 'model'
  | 'more'
  | 'playground'
  | 'play'
  | 'plug'
  | 'plus'
  | 'power'
  | 'projection'
  | 'puzzle'
  | 'refresh'
  | 'search'
  | 'server'
  | 'settings'
  | 'shield'
  | 'signal'
  | 'status'
  | 'stop'
  | 'team'
  | 'topology'
  | 'tool'
  | 'trace'
  | 'trash'
  | 'undo'
  | 'redo'
  | 'user'

interface Props extends Omit<SVGProps<SVGSVGElement>, 'name'> {
  name: ProductIconName
}

function glyph(name: ProductIconName): ReactNode {
  switch (name) {
    case 'activity':
      return <path d="M3 12h4l2.2-5.5 4.1 11L16 12h5" />
    case 'alert':
      return (
        <>
          <path d="M12 3 22 20H2L12 3Z" />
          <path d="M12 9v5m0 3h.01" />
        </>
      )
    case 'arrow-left':
      return <path d="M19 12H5m6-6-6 6 6 6" />
    case 'arrow-right':
      return <path d="M5 12h14m-6-6 6 6-6 6" />
    case 'attachment':
      return (
        <path d="m8.5 12.5 5.9-5.9a3 3 0 0 1 4.2 4.2l-7.8 7.8a5 5 0 0 1-7.1-7.1l8.1-8.1m-6.3 9.9 7.1-7.1" />
      )
    case 'audit':
      return (
        <>
          <path d="M7 3.5h10v17H7z" />
          <path d="M9.5 8h5M9.5 12h5M9.5 16h3" />
        </>
      )
    case 'budget':
      return (
        <>
          <rect x="3" y="5" width="18" height="14" rx="3" />
          <path d="M16 10h5v4h-5a2 2 0 0 1 0-4Z" />
        </>
      )
    case 'chart':
      return <path d="M4 19V9m5 10V5m5 14v-7m5 7V3M3 21h18" />
    case 'check':
      return <path d="m5 12.5 4.2 4.2L19 7" />
    case 'chevron-left':
      return <path d="m14.5 6-6 6 6 6" />
    case 'chevron-down':
      return <path d="m6 9.5 6 6 6-6" />
    case 'chevron-right':
      return <path d="m9.5 6 6 6-6 6" />
    case 'chevron-up':
      return <path d="m6 14.5 6-6 6 6" />
    case 'claw':
      return (
        <>
          <path d="M7.5 18.5c-2.2-1.4-3.5-3.7-3.5-6.3 0-3.7 2.6-6.8 6.1-7.5" />
          <path d="M16.5 18.5c2.2-1.4 3.5-3.7 3.5-6.3 0-3.7-2.6-6.8-6.1-7.5" />
          <path d="M9.5 9.5 12 12l2.5-2.5M8.5 21l3.5-5 3.5 5" />
        </>
      )
    case 'close':
      return <path d="m7 7 10 10M17 7 7 17" />
    case 'code':
      return <path d="m8.5 7-5 5 5 5M15.5 7l5 5-5 5M14 4l-4 16" />
    case 'compute':
      return (
        <>
          <rect x="5" y="5" width="14" height="14" rx="2" />
          <rect x="9" y="9" width="6" height="6" rx="1" />
          <path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3" />
        </>
      )
    case 'copy':
      return (
        <>
          <rect x="8" y="8" width="11" height="11" rx="2" />
          <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
        </>
      )
    case 'dashboard':
      return (
        <>
          <rect x="3" y="3" width="7" height="7" rx="1.5" />
          <rect x="14" y="3" width="7" height="4" rx="1.5" />
          <rect x="14" y="11" width="7" height="10" rx="1.5" />
          <rect x="3" y="14" width="7" height="7" rx="1.5" />
        </>
      )
    case 'database':
      return (
        <>
          <ellipse cx="12" cy="5" rx="8" ry="3" />
          <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" />
        </>
      )
    case 'download':
      return (
        <>
          <path d="M12 3v12m-5-5 5 5 5-5" />
          <path d="M4 19v2h16v-2" />
        </>
      )
    case 'decision':
      return (
        <>
          <path d="M12 3v5m0 0-5 5m5-5 5 5" />
          <circle cx="12" cy="3" r="1.5" />
          <circle cx="7" cy="15" r="2" />
          <circle cx="17" cy="15" r="2" />
          <path d="M7 17v4m10-4v4" />
        </>
      )
    case 'edit':
      return (
        <>
          <path d="M4 20h4l11-11-4-4L4 16v4Z" />
          <path d="m13.5 6.5 4 4" />
        </>
      )
    case 'expand':
      return (
        <>
          <path d="M8.5 3H3v5.5M15.5 3H21v5.5M8.5 21H3v-5.5M15.5 21H21v-5.5" />
          <path d="m3 8 5-5m8 0 5 5M3 16l5 5m8 0 5-5" />
        </>
      )
    case 'evaluation':
      return (
        <>
          <path d="M5 4h14v16H5zM8 8h5" />
          <path d="m8 14 2 2 5-5" />
        </>
      )
    case 'eye':
      return (
        <>
          <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" />
          <circle cx="12" cy="12" r="2.5" />
        </>
      )
    case 'eye-off':
      return (
        <>
          <path d="M4.3 4.3 19.7 19.7M9.8 6.3A10.2 10.2 0 0 1 12 6c6 0 9.5 6 9.5 6a15.7 15.7 0 0 1-2.1 2.8M6.6 7.1C4 8.8 2.5 12 2.5 12s3.5 6 9.5 6c1 0 2-.2 2.8-.5M10.2 10.2a2.5 2.5 0 0 0 3.6 3.6" />
        </>
      )
    case 'fleet':
      return (
        <>
          <rect x="3" y="5" width="7" height="5" rx="1" />
          <rect x="14" y="5" width="7" height="5" rx="1" />
          <rect x="8.5" y="15" width="7" height="5" rx="1" />
          <path d="M6.5 10v2.5H18V10M12 12.5V15" />
        </>
      )
    case 'fullscreen':
      return (
        <>
          <path d="M9 4H4v5M15 4h5v5M9 20H4v-5M15 20h5v-5" />
        </>
      )
    case 'globe':
      return (
        <>
          <circle cx="12" cy="12" r="9" />
          <path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" />
        </>
      )
    case 'inbox':
      return (
        <>
          <path d="M4 5h16v14H4z" />
          <path d="M4 14h5l1.5 2h3L15 14h5" />
        </>
      )
    case 'info':
      return (
        <>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v6m0-10h.01" />
        </>
      )
    case 'insight':
      return (
        <>
          <path d="M9 18h6M10 22h4" />
          <path d="M8.2 15.5A7 7 0 1 1 15.8 15.5c-.8.7-1.2 1.4-1.3 2.5h-5c-.1-1.1-.5-1.8-1.3-2.5Z" />
        </>
      )
    case 'key':
      return (
        <>
          <circle cx="8" cy="12" r="4" />
          <path d="M12 12h9M17 12v3M20 12v2" />
        </>
      )
    case 'label':
      return (
        <>
          <path d="M4 4h7l9 9-7 7-9-9V4Z" />
          <circle cx="8" cy="8" r="1.3" />
        </>
      )
    case 'logs':
      return (
        <>
          <path d="M5 4h14v16H5z" />
          <path d="M8.5 8h7M8.5 12h7M8.5 16h4" />
        </>
      )
    case 'minus':
      return <path d="M5 12h14" />
    case 'mixture':
      return (
        <>
          <circle cx="6" cy="6" r="2.5" />
          <circle cx="18" cy="6" r="2.5" />
          <circle cx="12" cy="18" r="2.5" />
          <path d="m8 7.5 2.8 7.8m5.2-7.8-2.8 7.8M8.5 6h7" />
        </>
      )
    case 'more':
      return (
        <>
          <circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" />
          <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
          <circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" />
        </>
      )
    case 'model':
      return (
        <>
          <path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z" />
          <path d="m4 12 8 4.5 8-4.5M4 16.5l8 4.5 8-4.5" />
        </>
      )
    case 'playground':
      return (
        <>
          <path d="M8 4h8l1 5 3 6a3 3 0 0 1-5.2 3L13 16h-2l-1.8 2A3 3 0 0 1 4 15l3-6 1-5Z" />
          <path d="M8 10h4M10 8v4M16.5 9.5h.01M18.5 11.5h.01" />
        </>
      )
    case 'play':
      return <path d="m8 5 11 7-11 7V5Z" />
    case 'plug':
      return <path d="M8 3v6m8-6v6M5 9h14v2a7 7 0 0 1-7 7v3m-4-8h8" />
    case 'plus':
      return <path d="M12 5v14M5 12h14" />
    case 'power':
      return (
        <>
          <path d="M12 3v9" />
          <path d="M7 5.8a8 8 0 1 0 10 0" />
        </>
      )
    case 'projection':
      return (
        <>
          <circle cx="6" cy="12" r="3" />
          <circle cx="18" cy="6" r="2" />
          <circle cx="18" cy="18" r="2" />
          <path d="M9 11 16 7M9 13l7 4" />
        </>
      )
    case 'puzzle':
      return (
        <path d="M4 4h6a2.5 2.5 0 1 0 4 0h6v6a2.5 2.5 0 1 1 0 4v6h-6a2.5 2.5 0 1 0-4 0H4v-6a2.5 2.5 0 1 1 0-4V4Z" />
      )
    case 'refresh':
      return (
        <>
          <path d="M20 7v5h-5" />
          <path d="M18.2 16.5A8 8 0 1 1 20 12" />
        </>
      )
    case 'redo':
      return <path d="M20 7v6h-6M4 17a8 8 0 0 1 13.7-5.6L20 13" />
    case 'search':
      return (
        <>
          <circle cx="10.5" cy="10.5" r="6.5" />
          <path d="m15.5 15.5 4.5 4.5" />
        </>
      )
    case 'server':
      return (
        <>
          <rect x="3" y="4" width="18" height="6" rx="2" />
          <rect x="3" y="14" width="18" height="6" rx="2" />
          <path d="M7 7h.01M7 17h.01M11 7h7M11 17h7" />
        </>
      )
    case 'settings':
      return (
        <>
          <circle cx="12" cy="12" r="3" />
          <path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.4-2.4 1a8 8 0 0 0-1.7-1L14.5 3h-5l-.4 3.1a8 8 0 0 0-1.7 1l-2.4-1-2 3.4L5 11a7 7 0 0 0 0 2l-2 1.5 2 3.4 2.4-1a8 8 0 0 0 1.7 1l.4 3.1h5l.4-3.1a8 8 0 0 0 1.7-1l2.4 1 2-3.4L18.9 13a7 7 0 0 0 .1-1Z" />
        </>
      )
    case 'shield':
      return (
        <>
          <path d="M12 3 20 6v5c0 5-3.4 8.3-8 10-4.6-1.7-8-5-8-10V6l8-3Z" />
          <path d="m8.5 12 2.2 2.2 4.8-5" />
        </>
      )
    case 'signal':
      return (
        <>
          <path d="M5 16a10 10 0 0 1 0-8M9 13a5 5 0 0 1 0-2m10 5a10 10 0 0 0 0-8m-4 5a5 5 0 0 0 0-2" />
          <circle cx="12" cy="12" r="2" />
        </>
      )
    case 'status':
      return (
        <>
          <circle cx="12" cy="12" r="9" />
          <path d="m8 12 2.5 2.5L16.5 8" />
        </>
      )
    case 'stop':
      return <rect x="6" y="6" width="12" height="12" rx="2" />
    case 'team':
      return (
        <>
          <circle cx="9" cy="9" r="3" />
          <circle cx="17" cy="10" r="2" />
          <path d="M3.5 19c.5-3.2 2.4-5 5.5-5s5 1.8 5.5 5M14 15c3.5-.6 5.5.8 6.5 3.5" />
        </>
      )
    case 'topology':
      return (
        <>
          <circle cx="12" cy="5" r="2.5" />
          <circle cx="5" cy="18" r="2.5" />
          <circle cx="19" cy="18" r="2.5" />
          <path d="m10.8 7.2-4.6 8.6m7-8.6 4.6 8.6M7.5 18h9" />
        </>
      )
    case 'tool':
      return (
        <>
          <path d="M14.5 6.2a4.5 4.5 0 0 0-5.8 5.7L3.5 17a2.5 2.5 0 0 0 3.5 3.5l5.1-5.2a4.5 4.5 0 0 0 5.7-5.8l-2.7 2.7-3.3-.7-.7-3.3 2.7-2.7Z" />
        </>
      )
    case 'trace':
      return (
        <>
          <circle cx="5" cy="6" r="2" />
          <circle cx="19" cy="6" r="2" />
          <circle cx="12" cy="18" r="2" />
          <path d="M7 6h10M6 8l5 8m7-8-5 8" />
        </>
      )
    case 'trash':
      return (
        <>
          <path d="M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13" />
          <path d="M10 11v5M14 11v5" />
        </>
      )
    case 'undo':
      return <path d="M4 7v6h6M20 17A8 8 0 0 0 6.3 11.4L4 13" />
    case 'user':
      return (
        <>
          <circle cx="12" cy="8" r="4" />
          <path d="M4.5 20c.6-4.2 3.1-6.5 7.5-6.5s6.9 2.3 7.5 6.5" />
        </>
      )
  }
}

export default function ProductIcon({ name, ...props }: Props) {
  const hasAccessibleName = Boolean(props['aria-label'] || props['aria-labelledby'])

  return (
    <svg
      viewBox="0 0 24 24"
      width="1em"
      height="1em"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      focusable="false"
      {...props}
      aria-hidden={hasAccessibleName ? undefined : (props['aria-hidden'] ?? true)}
      role={hasAccessibleName ? (props.role ?? 'img') : props.role}
    >
      {glyph(name)}
    </svg>
  )
}
