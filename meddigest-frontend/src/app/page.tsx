import NavTabs from './nav_tabs';
import Link from "next/link";
import Image from "next/image";
import SignupForm from './components/SignupForm';

function ModernBanner() {
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
            src="/images/meddigest-logo.png" 
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
            Stay Ahead of Medical Research
          </h1>
          
          <p className="text-xl text-gray-600 mb-12 max-w-3xl mx-auto leading-relaxed">
            AI-powered insights from medical research papers, delivered weekly to keep you at the forefront of healthcare innovation.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link
              href="/newspaper"
              className="inline-flex items-center px-8 py-4 bg-blue-600 text-white rounded-lg font-semibold text-lg shadow-sm hover:shadow-md transform hover:-translate-y-1 transition-all duration-200"
            >
              View Latest Digest
              <span className="ml-2">→</span>
            </Link>
            
            <div className="text-gray-500 font-medium text-sm">
              Powered by AI
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function BenefitCard({ 
  icon, 
  title, 
  description 
}: { 
  icon: React.ReactNode; 
  title: string; 
  description: string; 
}) {
  return (
    <div className="text-center md:text-left p-6 bg-gray-50 rounded-xl border border-gray-100">
      <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center mb-4 mx-auto md:mx-0">
        {icon}
      </div>
      <h4 className="text-xl font-semibold text-gray-900 mb-4">{title}</h4>
      <p className="text-gray-600 leading-relaxed">{description}</p>
    </div>
  );
}

function ProcessStep({ 
  step, 
  title, 
  description, 
  icon 
}: { 
  step: number; 
  title: string; 
  description: string; 
  icon: React.ReactNode; 
}) {
  return (
    <div className="text-center">
      <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
        {icon}
      </div>
      <h4 className="text-lg font-semibold text-gray-900 mb-2">{step}. {title}</h4>
      <p className="text-gray-600 text-sm leading-relaxed">{description}</p>
    </div>
  );
}

const BENEFITS = [
  {
    icon: (
      <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    title: "Time-Saving Research",
    description: "Save hours of research time by getting AI-curated insights instead of manually reading through hundreds of papers."
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    title: "Specialty-Focused",
    description: "Get research insights tailored to your medical specialty with organized, actionable summaries."
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
    title: "Evidence-Based Insights",
    description: "Access peer-reviewed research findings with clinical implications and practical applications."
  },
  {
    icon: (
      <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
    ),
    title: "Always Up-to-Date",
    description: "Stay current with weekly updates covering the latest breakthroughs and emerging trends."
  }
];

const PROCESS_STEPS = [
  {
    title: "Data Collection",
    description: "Automatically gather medical research papers from arXiv repositories",
    icon: (
      <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
      </svg>
    )
  },
  {
    title: "AI Analysis",
    description: "Advanced algorithms extract key findings, trends, and clinical implications",
    icon: (
      <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    )
  },
  {
    title: "Quality Filtering",
    description: "Filter for high-impact research with significant clinical relevance",
    icon: (
      <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    )
  },
  {
    title: "Digest Generation",
    description: "Compile insights into comprehensive, weekly research summaries",
    icon: (
      <svg className="w-8 h-8 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
      </svg>
    )
  }
];

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-gray-50 to-slate-100">
      <ModernBanner />
      <NavTabs />
      
      <main className="max-w-6xl mx-auto px-6 py-24">
        <div className="bg-white rounded-2xl shadow-sm p-16 mb-24 border border-gray-100">
          <div className="text-center mb-8">
            <h2 className="text-4xl font-bold text-gray-900">
              Welcome to MedDigest
            </h2>
          </div>
          <div className="max-w-4xl mx-auto">
            <p className="text-lg text-gray-600 leading-relaxed mb-12 text-center md:text-left">
              We automatically analyze medical research papers from arXiv&apos;s repositories, 
              extracting key findings and trends to help healthcare professionals stay current 
              with the latest developments. Our AI-powered platform transforms complex research 
              into actionable insights, saving you hours of manual review while ensuring you 
              never miss critical breakthroughs in your field.
            </p>
            
            <div className="grid md:grid-cols-2 gap-8 mb-12">
              {BENEFITS.map((benefit, index) => (
                <BenefitCard
                  key={index}
                  icon={benefit.icon}
                  title={benefit.title}
                  description={benefit.description}
                />
              ))}
            </div>
          </div>
        </div>

        <section className="mb-24">
          <div className="text-center mb-16">
            <h3 className="text-3xl font-bold text-gray-900 mb-4">
              How It Works
            </h3>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our AI-powered process transforms raw research into actionable insights
            </p>
          </div>
          
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8">
            <div className="grid md:grid-cols-4 gap-6">
              {PROCESS_STEPS.map((step, index) => (
                <ProcessStep
                  key={index}
                  step={index + 1}
                  title={step.title}
                  description={step.description}
                  icon={step.icon}
                />
              ))}
            </div>
          </div>
        </section>

        <section className="mb-24">
          <div className="bg-white rounded-2xl shadow-sm p-16 border border-gray-100 text-center">
            <h3 className="text-3xl font-bold text-gray-900 mb-6">
              Join the MedDigest Community
            </h3>
            <p className="text-lg text-gray-600 mb-12 max-w-2xl mx-auto leading-relaxed">
              Get personalized weekly research updates tailored to your medical interests. 
              Join healthcare professionals staying ahead of the latest discoveries.
            </p>
            <SignupForm />
          </div>
        </section>
      </main>
    </div>
  );
}