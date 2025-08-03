'use client';

import NavTabs from '../nav_tabs';
import Image from "next/image";

function AboutBanner() {
  return (
    <section className="relative bg-blue-50 py-24 text-center border-b overflow-hidden">
      <div className="absolute inset-0 opacity-5">
        <div className="absolute inset-0" style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000000' fill-opacity='0.1'%3E%3Ccircle cx='30' cy='30' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
        }} />
      </div>
      
      <div className="max-w-5xl mx-auto px-6">
        <div className="absolute left-6 top-6 z-20">
          <Image 
            src="/meddigest-logo.png" 
            alt="MedDigest Logo" 
            width={100} 
            height={100}
            className="drop-shadow-sm transform hover:scale-105 transition-all duration-300"
          />
        </div>
        
        <div className="relative z-10">
          <div className="inline-block bg-blue-50 text-blue-700 rounded-full px-6 py-2 mb-8 font-semibold text-sm border border-blue-100">
            MedDigest
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-8 leading-tight">
            About MedDigest
          </h1>
          
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Understanding the mission, relevance, and team behind our AI-powered medical research platform
          </p>
        </div>
      </div>
    </section>
  );
}

function SectionCard({ 
  title, 
  icon, 
  children 
}: { 
  title: string; 
  icon?: React.ReactNode; 
  children: React.ReactNode; 
}) {
  return (
    <div className="bg-white rounded-2xl shadow-sm p-16 border border-gray-100">
      <div className="flex items-center justify-center mb-8">
        {icon && (
          <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mr-6">
            {icon}
          </div>
        )}
        <h2 className="text-4xl font-bold text-gray-900">{title}</h2>
      </div>
      {children}
    </div>
  );
}

function RelevanceCard({ 
  icon, 
  title, 
  description 
}: { 
  icon: React.ReactNode; 
  title: string; 
  description: string[]; 
}) {
  return (
    <div className="p-6 bg-gray-50 rounded-xl border border-gray-100">
      <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-2xl font-semibold text-gray-900 mb-6">{title}</h3>
      {description.map((paragraph, index) => (
        <p key={index} className="text-gray-600 leading-relaxed mb-4 last:mb-0">
          {paragraph}
        </p>
      ))}
    </div>
  );
}

function TeamMember({ 
  name, 
  role, 
  description 
}: { 
  name: string; 
  role: string; 
  description: string; 
}) {
  return (
    <div className="text-center p-8 bg-gray-50 rounded-xl border border-gray-100">
      <div className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg className="w-12 h-12 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
      </div>
      <h3 className="text-2xl font-semibold text-gray-900 mb-2">{name}</h3>
      <p className="text-blue-600 font-semibold mb-4">{role}</p>
      <p className="text-gray-600 text-sm leading-relaxed">{description}</p>
    </div>
  );
}

const RELEVANCE_CARDS = [
  {
    icon: (
      <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    title: "The Research Explosion",
    description: [
      "Medical research is being published at an unprecedented rate. In 2024 alone, over 3 million scientific papers were published, with medical research accounting for a significant portion of this growth.",
      "This exponential growth makes it impossible for anyone to manually review even a fraction of the relevant research in their field of interest."
    ]
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
    title: "AI Revolution in Healthcare",
    description: [
      "Artificial intelligence has matured to the point where it can effectively analyze and summarize complex medical research, identifying key findings, clinical implications, and emerging trends.",
      "This technological advancement enables us to bridge the gap between research publication and clinical application, ensuring that valuable insights reach healthcare professionals when they need them most."
    ]
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
      </svg>
    ),
    title: "Patient Care Impact",
    description: [
      "When people have access to the latest research insights, they can make more informed decisions - whether that's healthcare professionals improving patient care, researchers building on existing work, or individuals understanding medical breakthroughs that might affect them or their families.",
      "By democratizing access to medical research, we're helping to ensure that the latest scientific discoveries reach everyone who can benefit from them."
    ]
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    title: "Time Efficiency",
    description: [
      "People are busy with their daily responsibilities - whether that's healthcare professionals with patient care, researchers with their own studies, or individuals with work and family commitments. Spending hours reading research papers is simply not feasible for most people.",
      "MedDigest provides a time-efficient solution that allows everyone to stay current with medical research without sacrificing their other important priorities."
    ]
  }
];

const TEAM_MEMBERS = [
  {
    name: "[Name]",
    role: "Co-Founder & Developer",
    description: "[Add background here.]"
  },
  {
    name: "[Name]",
    role: "Co-Founder & Developer", 
    description: "[Add background here.]"
  }
];

export default function About() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-gray-50 to-slate-100">
      <AboutBanner />
      <NavTabs />
      
      <main className="max-w-6xl mx-auto px-6 py-24">
        <section className="mb-24">
          <SectionCard
            title="The Idea Behind MedDigest"
          >
          <div className="max-w-4xl mx-auto">
            <p className="text-lg text-gray-600 leading-relaxed mb-6">
              MedDigest was born from a simple observation: the volume of medical research being published daily is overwhelming for everyone - from healthcare professionals 
              to researchers to curious individuals. With thousands of new papers appearing in repositories like arXiv every week, staying current with the latest developments 
              has become an impossible task.
            </p>
            <p className="text-lg text-gray-600 leading-relaxed mb-6">
              We recognized that while the research is valuable, the traditional approach of manually reading through papers is inefficient and often leads to 
              missed opportunities. People need a way to quickly identify and understand the most relevant research, whether they're healthcare professionals, 
              researchers, or simply interested in medical breakthroughs.
            </p>
            <p className="text-lg text-gray-600 leading-relaxed">
              Our solution leverages artificial intelligence to automatically analyze, summarize, and curate medical research, transforming complex academic papers 
              into accessible insights that anyone can understand and benefit from.
            </p>
          </div>
        </SectionCard>
        </section>

        <section className="mb-24">
          <SectionCard
            title="Why MedDigest is Relevant Today"
            icon={null}
          >
            <div className="grid md:grid-cols-2 gap-12">
              {RELEVANCE_CARDS.map((card, index) => (
                <RelevanceCard
                  key={index}
                  icon={card.icon}
                  title={card.title}
                  description={card.description}
                />
              ))}
            </div>
          </SectionCard>
        </section>

        <section className="mb-24">
          <div className="bg-white rounded-2xl shadow-sm p-16 border border-gray-100">
            <div className="text-center mb-12">
              <div className="w-16 h-16 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-6">
                <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                </svg>
              </div>
              <h2 className="text-4xl font-bold text-gray-900 mb-4">
                Developed By
              </h2>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Meet the team behind MedDigest
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-12 max-w-4xl mx-auto">
              {TEAM_MEMBERS.map((member, index) => (
                <TeamMember
                  key={index}
                  name={member.name}
                  role={member.role}
                  description={member.description}
                />
              ))}
            </div>
            
            <div className="text-center mt-12 pt-8 border-t border-gray-100">
              <p className="text-lg text-gray-600 leading-relaxed max-w-3xl mx-auto">
                Together, we're passionate about leveraging technology to democratize access to medical research and make scientific discoveries more accessible to everyone worldwide.
              </p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
} 