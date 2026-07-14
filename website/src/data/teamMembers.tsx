import React from 'react'
import Translate from '@docusaurus/Translate'

export interface MemberLink {
  label: string
  href: string
}

export interface TeamMember {
  name: string
  role: React.ReactNode
  company?: string
  avatar: string
  github?: string
  linkedin?: string
  externalLinks?: MemberLink[]
  bio: React.ReactNode
  expertise?: React.ReactNode[]
  memberType: 'steering' | 'committer'
  steeringTrack?: 'industry' | 'academic'
}

export const steeringCommitteeMembers: TeamMember[] = [
  {
    name: 'Xunzhuo Liu',
    role: <Translate id="team.members.XunzhuoLiu.role">LLM Routing @ vLLM</Translate>,
    avatar: '/img/team/xunzhuo.png',
    github: 'https://github.com/Xunzhuo',
    linkedin: 'http://linkedin.com/in/bitliu',
    externalLinks: [
      { label: 'Website', href: 'https://www.liuxunzhuo.com/' },
    ],
    bio: <Translate id="team.members.XunzhuoLiu.bio">LLM routing builder at vLLM.</Translate>,
    expertise: [
      <Translate id="team.members.XunzhuoLiu.expertise.routing">LLM routing</Translate>,
      <Translate id="team.members.XunzhuoLiu.expertise.gateway">Kubernetes AI Gateway</Translate>,
      <Translate id="team.members.XunzhuoLiu.expertise.opensource">Open-source infrastructure</Translate>,
    ],
    memberType: 'steering',
    steeringTrack: 'industry',
  },
  {
    name: 'Huamin Chen',
    role: <Translate id="team.members.HuaminChen.role">@Microsoft</Translate>,
    avatar: '/img/team/huamin.png',
    github: 'https://github.com/rootfs',
    linkedin: 'https://www.linkedin.com/in/huaminchen',
    externalLinks: [
      { label: 'Hugging Face', href: 'https://huggingface.co/HuaminChen' },
    ],
    bio: <Translate id="team.members.HuaminChen.bio">Long-term incubator of frontier infrastructure and AI systems across cloud-native platforms, open-source ecosystems, and model-serving stacks.</Translate>,
    expertise: [
      <Translate id="team.members.HuaminChen.expertise.cloud">Cloud-native platforms</Translate>,
      <Translate id="team.members.HuaminChen.expertise.serving">Model-serving stacks</Translate>,
      <Translate id="team.members.HuaminChen.expertise.ecosystem">Open-source ecosystems</Translate>,
    ],
    memberType: 'steering',
    steeringTrack: 'industry',
  },
  {
    name: 'Bowei He',
    role: <Translate id="team.members.BoweiHe.role">Postdoc @ MBZUAI / McGill</Translate>,
    avatar: 'https://agentic-in.ai/people/bowei-he.jpeg',
    linkedin: 'https://www.linkedin.com/in/bowei-he-8a9450199/',
    externalLinks: [
      { label: 'Google Scholar', href: 'https://scholar.google.com/citations?user=1cH0A9cAAAAJ&hl=zh-CN' },
    ],
    bio: <Translate id="team.members.BoweiHe.bio">PhD from CityUHK and former Hunyuan LLM under the Qingyun Talent program.</Translate>,
    expertise: [
      <Translate id="team.members.BoweiHe.expertise.research">Academic research</Translate>,
      <Translate id="team.members.BoweiHe.expertise.llm">LLM systems</Translate>,
      <Translate id="team.members.BoweiHe.expertise.hunyuan">Tencent Hunyuan LLM</Translate>,
    ],
    memberType: 'steering',
    steeringTrack: 'academic',
  },
  {
    name: 'Yankai Chen',
    role: <Translate id="team.members.YankaiChen.role">Postdoctoral Associate @ McGill University / MBZUAI</Translate>,
    avatar: 'https://agentic-in.ai/people/yankai-chen.jpg',
    linkedin: 'https://www.linkedin.com/in/yankai-chen-923001154/',
    externalLinks: [
      { label: 'Website', href: 'https://yankai-chen.github.io/' },
      { label: 'Google Scholar', href: 'https://scholar.google.com/citations?user=5ZOi7UAAAAAJ&hl=zh-CN' },
    ],
    bio: <Translate id="team.members.YankaiChen.bio">Research spanning agentic AI, human-centered AI, and knowledge mining.</Translate>,
    expertise: [
      <Translate id="team.members.YankaiChen.expertise.agentic">Agentic AI</Translate>,
      <Translate id="team.members.YankaiChen.expertise.hcai">Human-centered AI</Translate>,
      <Translate id="team.members.YankaiChen.expertise.mining">Knowledge mining</Translate>,
    ],
    memberType: 'steering',
    steeringTrack: 'academic',
  },
  {
    name: 'Fuyuan Lyu',
    role: <Translate id="team.members.FuyuanLyu.role">PhD Candidate @ McGill University / Mila</Translate>,
    avatar: 'https://agentic-in.ai/people/fuyuan-lv.jpeg',
    linkedin: 'https://www.linkedin.com/in/fuyuan-lyu-560756167/',
    externalLinks: [
      { label: 'Website', href: 'https://fuyuanlyu.github.io/' },
      { label: 'Google Scholar', href: 'https://scholar.google.com/citations?user=dOjmAVQAAAAJ&hl=en' },
    ],
    bio: <Translate id="team.members.FuyuanLyu.bio">Research in data-centric AI, automatic feature selection, and automatic labeling for deep learning and foundation models.</Translate>,
    expertise: [
      <Translate id="team.members.FuyuanLyu.expertise.data">Data-centric AI</Translate>,
      <Translate id="team.members.FuyuanLyu.expertise.features">Automatic feature selection</Translate>,
      <Translate id="team.members.FuyuanLyu.expertise.labeling">Automatic labeling</Translate>,
    ],
    memberType: 'steering',
    steeringTrack: 'academic',
  },
  {
    name: 'Steve Liu',
    role: <Translate id="team.members.SteveLiu.role">@MBZUAI / McGill / Mila</Translate>,
    avatar: 'https://agentic-in.ai/people/steve-liu.jpeg',
    linkedin: 'https://ca.linkedin.com/in/xueliu',
    externalLinks: [
      { label: 'MBZUAI', href: 'https://mbzuai.ac.ae/study/faculty/steve-liu/' },
      { label: 'Google Scholar', href: 'https://scholar.google.com/citations?user=rfLIRakAAAAJ&hl=en' },
    ],
    bio: <Translate id="team.members.SteveLiu.bio">Fellow CAE & IEEE; Associate VPR @ MBZUAI; Prof @ McGill; Mila; ex-VP R&D @ Samsung AI; Chair ACM SIGBED.</Translate>,
    expertise: [
      <Translate id="team.members.SteveLiu.expertise.aiml">AI and machine learning</Translate>,
      <Translate id="team.members.SteveLiu.expertise.systems">Intelligent systems</Translate>,
      <Translate id="team.members.SteveLiu.expertise.cps">Cyber-physical systems</Translate>,
    ],
    memberType: 'steering',
    steeringTrack: 'academic',
  },
]

export const industryTrackMembers = steeringCommitteeMembers.filter(
  member => member.steeringTrack === 'industry',
)

export const academicTrackMembers = steeringCommitteeMembers.filter(
  member => member.steeringTrack === 'academic',
)

export function getTeamMemberBadge(member: TeamMember): React.ReactNode {
  if (member.memberType === 'committer') {
    return <Translate id="team.badge.committer">Committer</Translate>
  }

  if (member.steeringTrack === 'industry') {
    return <Translate id="team.track.industry.title">Industry Track</Translate>
  }

  if (member.steeringTrack === 'academic') {
    return <Translate id="team.track.academic.title">Academic Track</Translate>
  }

  return <Translate id="team.badge.steering">Steering Committee</Translate>
}

export const topNewContributorMembers: TeamMember[] = [
  {
    name: 'FAUST',
    role: <Translate id="team.members.FAUST-BENCHOU.role">Cloud-native Open Source Contributor</Translate>,
    company: 'Tongji University',
    avatar: 'https://github.com/FAUST-BENCHOU.png',
    github: 'https://github.com/FAUST-BENCHOU',
    bio: <Translate id="team.members.FAUST-BENCHOU.bio">Cloud-native open source contributor across Karmada and Volcano.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'David Shrader',
    role: <Translate id="team.members.shraderdm.role">GTM Tech Lead</Translate>,
    company: 'Google',
    avatar: 'https://github.com/shraderdm.png',
    github: 'https://github.com/shraderdm',
    linkedin: 'https://www.linkedin.com/in/shraderdm/',
    externalLinks: [
      { label: 'Website', href: 'https://shrader.dev' },
    ],
    bio: <Translate id="team.members.shraderdm.bio">GTM Tech Lead at Google.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'yangw',
    role: <Translate id="team.members.drivebyer.role">Cloud-native Engineer</Translate>,
    company: 'DaoCloud',
    avatar: 'https://github.com/drivebyer.png',
    github: 'https://github.com/drivebyer',
    bio: <Translate id="team.members.drivebyer.bio">DaoCloud engineer and open-source contributor focused on practical infrastructure improvements.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Ramakrishnan Sathyavageeswaran',
    role: <Translate id="team.members.ramkrishs.role">Computer Science Engineer</Translate>,
    company: 'Intuit',
    avatar: 'https://github.com/ramkrishs.png',
    github: 'https://github.com/ramkrishs',
    externalLinks: [
      { label: 'Website', href: 'http://ramakrishnan.me' },
    ],
    bio: <Translate id="team.members.ramkrishs.bio">Computer science engineer at Intuit focused on software systems.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'WUKUNTAI',
    role: <Translate id="team.members.WUKUNTAI-0211.role">Software Engineer</Translate>,
    company: 'DELTA ELECTRONICS, INC.',
    avatar: 'https://github.com/WUKUNTAI-0211.png',
    github: 'https://github.com/WUKUNTAI-0211',
    bio: <Translate id="team.members.WUKUNTAI-0211.bio">Software engineer at Delta Electronics in Taiwan.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Aayush Saini',
    role: <Translate id="team.members.AayushSaini101.role">SDE, Data and AI</Translate>,
    company: 'Red Hat',
    avatar: 'https://github.com/AayushSaini101.png',
    github: 'https://github.com/AayushSaini101',
    bio: <Translate id="team.members.AayushSaini101.bio">SDE in Red Hat Data and AI, GSoC 2025 participant, and AsyncAPI Steering Committee member.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'siloteemu',
    role: <Translate id="team.members.siloteemu.role">Open Source Contributor</Translate>,
    avatar: 'https://github.com/siloteemu.png',
    github: 'https://github.com/siloteemu',
    bio: <Translate id="team.members.siloteemu.bio">Open-source contributor on GitHub.</Translate>,
    memberType: 'committer',
  },
]

export const committerMembers: TeamMember[] = [
  ...topNewContributorMembers,
  {
    name: 'Chen Wang',
    role: <Translate id="team.members.ChenWang.role">Senior Staff Research Scientist</Translate>,
    company: 'IBM',
    avatar: '/img/team/chen.png',
    github: 'https://github.com/wangchen615',
    linkedin: 'https://www.linkedin.com/in/chenw615/',
    bio: <Translate id="team.members.ChenWang.bio">Senior Staff Research Scientist at IBM, focusing on advanced AI systems and research.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Yue Zhu',
    role: <Translate id="team.members.YueZhu.role">Staff Research Scientist</Translate>,
    company: 'IBM',
    avatar: '/img/team/yue.png',
    github: 'https://github.com/yuezhu1',
    linkedin: 'https://www.linkedin.com/in/yue-zhu-b26526a3/',
    bio: <Translate id="team.members.YueZhu.bio">Staff Research Scientist at IBM, specializing in AI research and LLM Inference.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Senan Zedan',
    company: 'Red Hat',
    role: <Translate id="team.members.SenanZedan.role">R&D Manager</Translate>,
    linkedin: 'https://www.linkedin.com/in/senan-zedan-2041855b/',
    avatar: 'https://github.com/szedan-rh.png',
    github: 'https://github.com/szedan-rh',
    bio: <Translate id="team.members.SenanZedan.bio">A dynamic and hands-on Engineering Manager who thrives on building elite engineering teams and driving them to deliver exceptional results.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Yossi Ovadia',
    company: 'Red Hat',
    role: <Translate id="team.members.YossiOvadia.role">Senior Principal Engineer</Translate>,
    avatar: 'https://github.com/yossiovadia.png',
    github: 'https://github.com/yossiovadia',
    linkedin: 'https://www.linkedin.com/in/yossi-ovadia-336b314/',
    bio: <Translate id="team.members.YossiOvadia.bio">Making life easier for developers and customers through innovative tooling. From the Red Hat Office of the CTO.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'samzong',
    role: <Translate id="team.members.samzong.role">AI Infrastructure / Cloud-Native PM</Translate>,
    company: 'DaoCloud',
    avatar: 'https://github.com/samzong.png',
    github: 'https://github.com/samzong',
    linkedin: 'https://www.linkedin.com/in/samzong',
    bio: <Translate id="team.members.samzong.bio">Cloud-native AI infrastructure product leader. Focused on Kubernetes, GPU resource scheduling, and large-scale LLM serving platforms.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Liav Weiss',
    role: <Translate id="team.members.LiavWeiss.role">Software Engineer</Translate>,
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/74174727?v=4',
    github: 'https://github.com/liavweiss',
    linkedin: 'https://www.linkedin.com/in/liav-weiss-2a0428208',
    bio: <Translate id="team.members.LiavWeiss.bio">Software engineer, focused on backend and cloud-native systems, with hands-on experience exploring AI infrastructure, LLM-based systems, and RAG architectures.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Asaad Balum',
    role: <Translate id="team.members.AsaadBalum.role">Senior Software Engineer</Translate>,
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/154635253?s=400&u=6e7e87cce16b88346a3e54e96aad263318a1901a&v=4',
    github: 'https://github.com/asaadbalum',
    linkedin: 'https://www.linkedin.com/in/asaad-balum-0928771a9/',
    bio: <Translate id="team.members.AsaadBalum.bio">Senior software engineer with a research-driven mindset, specializing in cloud-native platforms, Kubernetes-based infrastructure, and AI enablement.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Yehudit',
    role: <Translate id="team.members.Yehudit.role">Software Engineer</Translate>,
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/u/34643974?s=400&v=4',
    github: 'https://github.com/yehuditkerido',
    linkedin: 'https://www.linkedin.com/in/yehuditkerido/',
    bio: <Translate id="team.members.Yehudit.bio">Software engineer with a research-driven mindset, focused on cloud-native platforms and AI infrastructure. Open-source contributor.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Noa Limoy',
    role: <Translate id="team.members.NoaLimoy.role">Software Engineer</Translate>,
    company: 'Red Hat',
    avatar: 'https://avatars.githubusercontent.com/noalimoy',
    github: 'https://github.com/noalimoy',
    linkedin: 'https://www.linkedin.com/in/noalimoy/',
    bio: <Translate id="team.members.NoaLimoy.bio">Software engineer with a research-driven mindset, focused on cloud-native platforms and AI infrastructure. Open-source contributor.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Marina Koushnir',
    role: <Translate id="team.members.MarinaKoushnir.role">Open Source Contributor</Translate>,
    company: 'Red Hat',
    avatar: 'https://github.com/mkoushni.png',
    github: 'https://github.com/mkoushni',
    bio: <Translate id="team.members.MarinaKoushnir.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'JaredforReal',
    company: 'Z.ai',
    role: <Translate id="team.members.JaredforReal.role">Software Engineer</Translate>,
    avatar: 'https://github.com/JaredforReal.png',
    github: 'https://github.com/JaredforReal',
    bio: <Translate id="team.members.JaredforReal.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Abdallah Samara',
    company: 'Red Hat',
    role: <Translate id="team.members.AbdallahSamara.role">Senior Software Engineer</Translate>,
    avatar: 'https://github.com/abdallahsamabd.png',
    github: 'https://github.com/abdallahsamabd',
    linkedin: 'https://www.linkedin.com/in/abdallah-samara',
    bio: <Translate id="team.members.AbdallahSamara.bio">Software engineer with a research-driven approach, focused on cloud-native platforms and AI infrastructure. Building semantic routing systems and contributing to open-source LLM orchestration projects.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Hen Schwartz',
    company: 'Red Hat',
    role: <Translate id="team.members.HenSchwartz.role">Software Engineer</Translate>,
    avatar: 'https://github.com/henschwartz.png',
    github: 'https://github.com/henschwartz',
    linkedin: 'https://www.linkedin.com/in/henschwartz',
    bio: <Translate id="team.members.HenSchwartz.bio">Software engineer with a research-driven approach, focused on cloud-native platforms and AI infrastructure. Building semantic routing systems and contributing to open-source LLM orchestration projects.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Srinivas A',
    role: <Translate id="team.members.SrinivasA.role">Software Engineer</Translate>,
    company: 'Yokogawa',
    avatar: 'https://avatars.githubusercontent.com/srini-abhiram',
    github: 'https://github.com/srini-abhiram',
    linkedin: 'https://www.linkedin.com/in/sriniabhiram',
    bio: <Translate id="team.members.SrinivasA.bio">Application software engineer with experience in Distributed Control Systems and Big data.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'carlory',
    role: <Translate id="team.members.carlory.role">Open Source Engineer</Translate>,
    company: 'DaoCloud',
    avatar: 'https://avatars.githubusercontent.com/u/28390961?v=4',
    github: 'https://github.com/carlory',
    bio: <Translate id="team.members.carlory.bio">Open Source Engineer at DaoCloud, focusing on container technology and cloud-native solutions. Passionate about contributing to vllm and other open source projects.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Jintao Zhang',
    company: 'Kong',
    role: <Translate id="team.members.JintaoZhang.role">Senior Software Engineer</Translate>,
    avatar: 'https://github.com/tao12345666333.png',
    github: 'https://github.com/tao12345666333',
    linkedin: 'https://www.linkedin.com/in/jintao-zhang-402645193/',
    bio: <Translate id="team.members.JintaoZhang.bio">Senior Software Engineer @ Kong Inc. | Microsoft MVP | CNCF Ambassador | Kubernetes Ingress-NGINX maintainer | PyCon China & KCD Beijing organizer.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'yuluo-yx',
    role: <Translate id="team.members.yuluo-yx.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/yuluo-yx.png',
    github: 'https://github.com/yuluo-yx',
    bio: <Translate id="team.members.yuluo-yx.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'cryo-zd',
    role: <Translate id="team.members.cryo-zd.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/cryo-zd.png',
    github: 'https://github.com/cryo-zd',
    bio: <Translate id="team.members.cryo-zd.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'OneZero-Y',
    role: <Translate id="team.members.OneZero-Y.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/OneZero-Y.png',
    github: 'https://github.com/OneZero-Y',
    bio: <Translate id="team.members.OneZero-Y.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'aeft',
    role: <Translate id="team.members.aeft.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/aeft.png',
    github: 'https://github.com/aeft',
    bio: <Translate id="team.members.aeft.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Hao Wu',
    role: <Translate id="team.members.HaoWu.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/haowu1234.png',
    github: 'https://github.com/haowu1234',
    bio: <Translate id="team.members.HaoWu.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
  {
    name: 'Qiping Pan',
    role: <Translate id="team.members.QipingPan.role">Individual Contributor</Translate>,
    avatar: 'https://github.com/ppppqp.png',
    github: 'https://github.com/ppppqp',
    linkedin: 'https://www.linkedin.com/in/qiping-pan-8662ab215/',
    bio: <Translate id="team.members.QipingPan.bio">Open source contributor to vLLM Semantic Router.</Translate>,
    memberType: 'committer',
  },
]
