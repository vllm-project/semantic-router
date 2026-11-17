export interface Testimonial {
  image: string
  kind: 'quote' | 'perspective'
  // The organisation's mark from static/img/ecosystem: dark lettering for light mode, light lettering for dark.
  logo: { alt: string, light: string, dark: string }
  label: string
  name: string
  role: string
  sourceLabel: string
  sourceUrl: string
  statement: string
}

export const testimonials: Testimonial[] = [
  {
    image: '/img/testimonials/satya-nadella.jpg',
    kind: 'quote',
    label: 'Meta model',
    name: 'Satya Nadella',
    logo: { alt: 'Microsoft', light: '/img/ecosystem/microsoft.svg', dark: '/img/ecosystem/microsoft.svg' },
    role: 'Microsoft Chairman & CEO',
    sourceLabel: 'Morgan Stanley TMT Conference · 2026',
    sourceUrl: 'https://www.microsoft.com/en-us/investor/events/fy-2026/morgan-stanley-tmt-conference',
    statement: 'Think of it as a meta model, right? That’s the most important thing.',
  },
  {
    image: '/img/testimonials/clement-delangue.png',
    kind: 'quote',
    label: 'Open-source routers',
    name: 'Clément Delangue',
    logo: { alt: 'Hugging Face', light: '/img/ecosystem/hugging-face.svg', dark: '/img/ecosystem/hugging-face.svg' },
    role: 'Hugging Face Co-founder & CEO',
    sourceLabel: 'Public post on open-source router systems',
    sourceUrl: 'https://x.com/ClementDelangue/status/2071987200481202217',
    statement: 'The future is multi-models and you’ll want to customize your router the same way you customize your code!',
  },
  {
    image: '/img/testimonials/mark-papermaster.jpeg',
    kind: 'quote',
    label: 'System-level optimization',
    name: 'Mark Papermaster',
    logo: { alt: 'AMD', light: '/img/ecosystem/amd-light.svg', dark: '/img/ecosystem/amd.svg' },
    role: 'AMD CTO & EVP',
    sourceLabel: 'theCUBE at RAISE Summit · 2026',
    sourceUrl: 'https://www.thecube.net/events/nyse/raise-summit-26/content/Videos/9b85b557-8da2-4e68-8621-a9cec1a84674',
    statement: 'We’ve become system optimizers, still optimizing every component, but then equally looking at how you optimize how each of the pieces come together.',
  },
  {
    image: '/img/testimonials/chris-wright.jpeg',
    kind: 'perspective',
    label: 'Hybrid enterprise AI',
    name: 'Chris Wright',
    logo: { alt: 'Red Hat', light: '/img/ecosystem/red-hat.svg', dark: '/img/ecosystem/red-hat.svg' },
    role: 'Red Hat CTO',
    sourceLabel: 'Technically Speaking / Red Hat · 2026',
    sourceUrl: 'https://www.youtube.com/watch?v=ExbMEW-Os1I&t=386s',
    statement: 'Enterprise AI needs control over cost, privacy, and policy.',
  },
  {
    image: '/img/testimonials/jensen-huang.jpeg',
    kind: 'perspective',
    label: 'Smart router',
    name: 'Jensen Huang',
    logo: { alt: 'NVIDIA', light: '/img/ecosystem/nvidia-light.svg', dark: '/img/ecosystem/nvidia.svg' },
    role: 'NVIDIA Co-founder, President & CEO',
    sourceLabel: 'NVIDIA Live at CES · 2026',
    sourceUrl: 'https://www.youtube.com/live/0NBILspM4c4?t=2243s',
    statement: 'A smart router sends every task to its best-fit model.',
  },
]
