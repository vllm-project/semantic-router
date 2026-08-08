export type SponsorCategoryId = 'cash' | 'compute' | 'slack'

/** mono = single-color mark (inverted on dark); brand = full-color artwork */
export type SponsorLogoStyle = 'mono' | 'brand'

export type Sponsor = {
  id: string
  name: string
  url: string
  logo: string
  logoStyle?: SponsorLogoStyle
}

export type SponsorCategory = {
  id: SponsorCategoryId
  sponsors: Sponsor[]
}

/** Aligned with the vLLM community sponsor roster (https://vllm.ai/). */
export const sponsorCategories: SponsorCategory[] = [
  {
    id: 'cash',
    sponsors: [
      {
        id: 'a16z',
        name: 'a16z',
        url: 'https://a16z.com/',
        logo: '/img/sponsors/a16z-icon.png',
        logoStyle: 'brand',
      },
      {
        id: 'sequoia',
        name: 'Sequoia Capital',
        url: 'https://www.sequoiacap.com/',
        logo: '/img/sponsors/sequoia-icon.png',
        logoStyle: 'brand',
      },
      {
        id: 'skywork',
        name: 'Skywork AI',
        url: 'https://skywork.ai/',
        logo: '/img/sponsors/skywork-icon.png',
        logoStyle: 'brand',
      },
      {
        id: 'zhenfund',
        name: 'ZhenFund',
        url: 'https://www.zhenfund.com/',
        logo: '/img/sponsors/zhenfund-icon.png',
        logoStyle: 'brand',
      },
    ],
  },
  {
    id: 'compute',
    sponsors: [
      {
        id: 'alibaba-cloud',
        name: 'Alibaba Cloud',
        url: 'https://www.alibabacloud.com/',
        logo: '/img/sponsors/alibaba-cloud.svg',
        logoStyle: 'brand',
      },
      {
        id: 'amd',
        name: 'AMD',
        url: 'https://www.amd.com/',
        logo: '/img/sponsors/amd.svg',
        logoStyle: 'brand',
      },
      {
        id: 'anyscale',
        name: 'Anyscale',
        url: 'https://www.anyscale.com/',
        logo: '/img/sponsors/anyscale-icon.png',
        logoStyle: 'brand',
      },
      {
        id: 'aws',
        name: 'AWS',
        url: 'https://aws.amazon.com/',
        logo: '/img/sponsors/aws.svg',
        logoStyle: 'brand',
      },
      {
        id: 'crusoe',
        name: 'Crusoe Cloud',
        url: 'https://www.crusoe.ai/',
        logo: '/img/sponsors/crusoe-icon.png',
        logoStyle: 'brand',
      },
      {
        id: 'google-cloud',
        name: 'Google Cloud',
        url: 'https://cloud.google.com/',
        logo: '/img/sponsors/google-cloud.svg',
        logoStyle: 'brand',
      },
      {
        id: 'ibm',
        name: 'IBM',
        url: 'https://www.ibm.com/',
        logo: '/img/sponsors/ibm.svg',
        logoStyle: 'brand',
      },
      {
        id: 'intel',
        name: 'Intel',
        url: 'https://www.intel.com/',
        logo: '/img/sponsors/intel.svg',
        logoStyle: 'brand',
      },
      {
        id: 'lambda',
        name: 'Lambda Lab',
        url: 'https://lambdalabs.com/',
        logo: '/img/sponsors/lambda.svg',
        logoStyle: 'brand',
      },
      {
        id: 'microsoft',
        name: 'Microsoft',
        url: 'https://www.microsoft.com/',
        logo: '/img/sponsors/microsoft.svg',
        logoStyle: 'brand',
      },
      {
        id: 'nebius',
        name: 'Nebius',
        url: 'https://nebius.com/',
        logo: '/img/sponsors/nebius.svg',
        logoStyle: 'brand',
      },
      {
        id: 'novita',
        name: 'Novita AI',
        url: 'https://novita.ai/',
        logo: '/img/sponsors/novita-icon.png',
        logoStyle: 'brand',
      },
      {
        id: 'nvidia',
        name: 'NVIDIA',
        url: 'https://www.nvidia.com/',
        logo: '/img/sponsors/nvidia.svg',
        logoStyle: 'brand',
      },
      {
        id: 'redhat',
        name: 'Red Hat',
        url: 'https://www.redhat.com/',
        logo: '/img/sponsors/redhat.svg',
        logoStyle: 'brand',
      },
      {
        id: 'roblox',
        name: 'Roblox',
        url: 'https://www.roblox.com/',
        logo: '/img/sponsors/roblox.svg',
        logoStyle: 'brand',
      },
      {
        id: 'runpod',
        name: 'RunPod',
        url: 'https://www.runpod.io/',
        logo: '/img/sponsors/runpod-icon.png',
        logoStyle: 'brand',
      },
      {
        id: 'ucb',
        name: 'UC Berkeley',
        url: 'https://www.berkeley.edu/',
        logo: '/img/sponsors/berkeley.svg',
        logoStyle: 'brand',
      },
    ],
  },
  {
    id: 'slack',
    sponsors: [
      {
        id: 'inferact',
        name: 'Inferact',
        url: 'https://inferact.ai/',
        logo: '/img/sponsors/inferact-icon.png',
        logoStyle: 'brand',
      },
    ],
  },
]
