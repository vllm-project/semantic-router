declare module 'node:fs' {
  interface DirectoryEntry {
    name: string
    isDirectory(): boolean
    isFile(): boolean
  }

  interface NodeBuffer extends Uint8Array {
    equals(otherBuffer: Uint8Array): boolean
  }

  export function readdirSync(
    path: string | URL,
    options: { withFileTypes: true },
  ): DirectoryEntry[]

  export function readFileSync(path: string | URL, encoding: 'utf8'): string
  export function readFileSync(path: string | URL): NodeBuffer
}
