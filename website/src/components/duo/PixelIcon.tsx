import React from 'react'

/* Duo's 8×8 pixel icons (mosaic.js), as SVG so they render on the server and stay crisp: "#" takes the text colour,
 * "*" the ember accent. */
const ICONS = {
  bolt: ['....##..', '...##...', '..##....', '.******.', '....##..', '...##...', '..##....', '.##.....'],
  dial: ['..####..', '.#....#.', '#......#', '#..#...#', '#...#..#', '#....*.#', '.#....#.', '..####..'],
  open: ['..####..', '.#....#.', '#......#', '#.....*.', '#.....*.', '#......#', '.#....#.', '..####..'],
  client: ['########', '#......#', '#.*....#', '#......#', '########', '...##...', '.######.', '........'],
  front: ['..####..', '.#....#.', '#......#', '#..*...#', '#...*..#', '#..*...#', '#......#', '########'],
  engine: ['........', '......*.', '.....#..', '*####...', '.....#..', '......*.', '........', '........'],
  model: ['..#..#..', '.######.', '#.#..#.#', '.#.**.#.', '.#.**.#.', '#.#..#.#', '.######.', '..#..#..'],
  pool: ['##.##.##', '##.##.##', '........', '##.##.**', '##.##.**', '........', '##.##.##', '##.##.##'],
  key: ['........', '.###....', '#...#...', '#...####', '#...#.#*', '.###....', '........', '........'],
  code: ['..#..#..', '.#....#.', '.#....#.', '#..**..#', '.#....#.', '.#....#.', '..#..#..', '........'],
  agent: ['...**...', '.######.', '#......#', '#.#..#.#', '#......#', '#..##..#', '.######.', '..#..#..'],
} as const

export type PixelIconName = keyof typeof ICONS

export default function PixelIcon({ name, size = 24, className }: {
  name: PixelIconName
  size?: number
  className?: string
}): JSX.Element {
  const cells: JSX.Element[] = []
  ICONS[name].forEach((line, y) => {
    for (let x = 0; x < 8; x++) {
      const ch = line[x]
      if (ch === '.') continue
      cells.push(
        <rect
          key={`${x}-${y}`}
          x={x * 5 + 0.5}
          y={y * 5 + 0.5}
          width={4}
          height={4}
          fill={ch === '*' ? 'var(--ember, #d9441f)' : 'currentColor'}
        />,
      )
    }
  })
  return (
    <svg
      className={className}
      viewBox="0 0 40 40"
      width={size}
      height={size}
      shapeRendering="crispEdges"
      aria-hidden="true"
      focusable="false"
    >
      {cells}
    </svg>
  )
}
