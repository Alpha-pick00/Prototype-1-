import { useRef } from 'react';
import { motion, useScroll, useTransform } from 'motion/react';
import weCraftImage from '../../assets/about/we-craft.jpg';

import qwenLogo from '../../assets/about/logos/qwen.svg';
import deepseekLogo from '../../assets/about/logos/deepseek.svg';
import awsLogo from '../../assets/about/logos/aws.svg';

// "gpt" 에이전트 슬롯은 Qwen(DashScope)이 담당한다(backend/app/agents/gpt.py
// 참고) - qwen.svg는 정식 로고 에셋을 아직 못 구해 임시 텍스트 워드마크다.
// 실제 로고 파일이 생기면 교체할 것. 메인 검색은 11번가 오픈 API 기반이라
// (Tavily/다나와 스크래핑 제거) 11번가 로고 에셋을 구하면 이 자리에 추가할 것.
const poweredByClients = [
  { name: 'Qwen', url: 'https://qwenlm.ai', logo: qwenLogo },
  { name: 'DeepSeek', url: 'https://www.deepseek.com', logo: deepseekLogo },
  { name: 'Amazon AWS', url: 'https://aws.amazon.com', logo: awsLogo },
];

export const About = () => {
  const containerRef = useRef(null);

  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"]
  });

  const opacity = useTransform(scrollYProgress, [0, 0.3], [0, 1]);

  return (
    <section ref={containerRef} id="about" className="py-32 relative bg-white overflow-hidden">
      {/* Background Grid - Technical Texture */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#00000008_1px,transparent_1px),linear-gradient(to_bottom,#00000008_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] pointer-events-none" />

      <div className="container mx-auto px-6">
        
        {/* Section Header - Consistent Style */}
        <div className="flex items-center gap-6 mb-24">
           <div className="flex items-baseline gap-3">
              <span className="font-serif italic text-lg text-neutral-950">01</span>
              <span className="text-xs font-mono uppercase tracking-[0.3em] text-neutral-600">About <span style={{ color: 'rgb(64,117,38)' }}>αlpha Pick</span></span>
           </div>
           <div className="h-px w-32 bg-gradient-to-r from-black/30 to-transparent" />
        </div>

        <div className="grid lg:grid-cols-[1.2fr_0.8fr] gap-20 items-start">

          {/* Text Content */}
          <div className="relative z-10 min-w-0">
            <motion.h2 
              initial={{ opacity: 0, y: 100 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              className="text-5xl md:text-8xl font-medium tracking-tighter mb-12 leading-[0.9]"
            >
              We craft <br />
              <span className="italic font-serif" style={{ color: '#4ADE80' }}>smarter</span> choices.
            </motion.h2>

            <div className="grid sm:grid-cols-3 gap-10">
              {[
                {
                  label: '소음 대신 답',
                  text: '수십 개의 창, 수십 개의 가격. 우리는 그중 하나만 골라 보여드립니다.',
                },
                {
                  label: '근거 있는 추천',
                  text: '이유 없는 가격은 의미 없습니다. 모든 추천에는 근거가 함께합니다.',
                },
                {
                  label: '시간을 위한 서비스',
                  text: '돈만큼 시간도 소중하니까요. 확인은 저희가, 선택은 당신이 합니다.',
                },
              ].map((item, index) => (
                <motion.div
                  key={item.label}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.2 + index * 0.1, duration: 0.8 }}
                  className="space-y-3"
                >
                  <h4 className="text-xs font-mono uppercase tracking-widest" style={{ color: '#4ADE80' }}>
                    {item.label}
                  </h4>
                  <p className="text-xl font-light text-neutral-700 leading-snug">
                    {item.text}
                  </p>
                </motion.div>
              ))}
            </div>

            {/* Stats & Trust */}
            <div className="mt-16 pt-16 border-t border-black/5">
               <div className="grid grid-cols-3 gap-8 mb-16">
                 <div className="space-y-2 border-r border-black/5">
                   <h4 className="text-4xl font-light text-neutral-950">2026</h4>
                   <p className="text-xs uppercase tracking-widest text-neutral-500">Launching</p>
                 </div>
                 <div className="space-y-2 border-r border-black/5">
                   <h4 className="text-4xl font-light text-neutral-950">15<span className="text-neutral-400 text-lg">+</span></h4>
                   <p className="text-xs uppercase tracking-widest text-neutral-500">Platforms Compared</p>
                 </div>
                 <div className="space-y-2">
                   <h4 className="text-4xl font-light text-neutral-950">100K</h4>
                   <p className="text-xs uppercase tracking-widest text-neutral-500">Target Users by Year 3</p>
                 </div>
               </div>
            </div>
          </div>

          {/* Image Area */}
          <motion.div 
            style={{ opacity }}
            className="relative lg:mt-24"
          >
            <div className="relative z-10">
               <motion.div 
                 whileHover={{ scale: 0.98 }}
                 transition={{ duration: 0.5 }}
                 className="aspect-[4/5] overflow-hidden grayscale hover:grayscale-0 transition-all duration-700 ease-in-out bg-neutral-100"
               >
                 <img
                   src={weCraftImage}
                   alt="Workspace"
                   className="w-full h-full object-cover opacity-80"
                 />
                 <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent" />
               </motion.div>
               
               {/* Decorative Ring */}
               <div className="absolute -bottom-12 -left-12 w-48 h-48 border border-black/10 rounded-full flex items-center justify-center backdrop-blur-sm hidden md:flex" style={{ animation: 'spin 15s linear infinite' }}>
                 <style dangerouslySetInnerHTML={{__html: `
                   @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
                 `}} />
                 <svg className="w-full h-full p-2" viewBox="0 0 100 100">
                   <path id="circlePath" d="M 50, 50 m -37, 0 a 37,37 0 1,1 74,0 a 37,37 0 1,1 -74,0" fill="transparent" />
                   <text className="fill-neutral-500 text-[10px] uppercase tracking-widest font-mono">
                     <textPath href="#circlePath">
                       - Price Comparison • AI Curation • Smart Shopping
                     </textPath>
                   </text>
                 </svg>
               </div>
            </div>
          </motion.div>

        </div>

        {/* Powered By - AI 모델 + 인프라 로고를 한 줄 스캐닝 목록으로 보여준다 */}
        <div className="mt-20 pt-16 border-t border-black/5">
          <span className="text-base font-mono uppercase tracking-widest text-neutral-400 block mb-6">Powered by</span>
          <div className="relative w-full overflow-hidden [mask-image:linear-gradient(to_right,transparent,black_8%,black_92%,transparent)]">
            <div
              className="marquee-track flex w-max items-center gap-4 whitespace-nowrap"
              style={{ animation: 'marquee 40s linear infinite' }}
            >
              {[...poweredByClients, ...poweredByClients].map((client, i) => (
                <a
                  key={`${client.name}-${i}`}
                  href={client.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  title={client.name}
                  className="shrink-0 flex items-center gap-3 h-16 px-6 rounded-xl bg-neutral-100 grayscale opacity-70 hover:grayscale-0 hover:opacity-100 transition-all cursor-pointer"
                >
                  <img src={client.logo} alt="" className="h-6 md:h-7 w-auto max-w-[32px] object-contain shrink-0" />
                  <span className="text-sm font-light tracking-wide text-neutral-700 whitespace-nowrap">
                    {client.name}
                  </span>
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
