'use client';

import { useState, useEffect } from 'react';
import NavTabs from '../nav_tabs';
import Image from "next/image";

interface Paper {
  id: string;
  title: string;
  authors: string[];
  abstract: string;
  keywords: string[];
  focus: string;
  date: string;
  specialty: string;
}

interface NewsletterData {
  date_generated: string;
  total_papers: number;
  executive_summary?: string;
  key_discoveries?: string[];
  emerging_trends?: string;
  cross_specialty_insights?: string;
  clinical_implications?: string;
  research_gaps?: string;
  future_directions?: string;
  specialty_data?: Record<string, { papers: Paper[] }>;
}

function NewspaperBanner({ date, totalPapers }: { date: string; totalPapers: number }) {
  const formattedDate = new Date(date).toLocaleDateString('en-US', { 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric' 
  });

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
              Weekly Medical Research Digest
            </h1>
            
            <p className="text-xl text-gray-600 mb-12 max-w-3xl mx-auto leading-relaxed">
              AI-curated insights from the latest medical research papers
            </p>
            
            <div className="bg-white rounded-2xl shadow-sm px-8 py-6 inline-block border border-gray-200">
              <div className="text-sm font-semibold text-gray-600 mb-2">Current Issue</div>
              <div className="text-2xl font-bold text-gray-900">
                {formattedDate} | {totalPapers} Papers Analyzed
              </div>
            </div>
          </div>
        </div>
    </section>
  );
}

function NewsletterSection({ 
  title, 
  icon, 
  children, 
  fallback 
}: { 
  title: string; 
  icon: string; 
  children: React.ReactNode; 
  fallback?: string;
}) {
  return (
    <section className="bg-white rounded-2xl shadow-sm p-10 mb-12 border border-gray-100">
      <h2 className="text-3xl font-bold text-gray-900 mb-8 flex items-center">
        <span className="mr-4 w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center text-blue-600 text-xl">
          {icon}
        </span>
        {title}
      </h2>
      {children || (
        <div className="prose prose-gray max-w-none">
          <p className="text-gray-700 leading-relaxed text-lg">
            {fallback}
          </p>
        </div>
      )}
    </section>
  );
}

function KeyDiscoveriesList({ discoveries }: { discoveries: string[] }) {
  if (!discoveries?.length) {
    return (
      <div className="text-gray-600 italic bg-gray-50 p-6 rounded-xl">
        Key discoveries are being processed from the research papers in this digest.
      </div>
    );
  }

  return (
    <ul className="space-y-4">
      {discoveries.map((discovery, index) => (
        <li key={index} className="text-gray-700 leading-relaxed text-lg bg-gray-50 p-6 rounded-xl border-l-4 border-blue-500">
          {discovery}
        </li>
      ))}
    </ul>
  );
}

function ResearchCategories({ specialtyData }: { specialtyData: Record<string, { papers: Paper[] }> }) {
  const categories = Object.entries(specialtyData);
  const maxPapers = Math.max(...categories.map(([, data]) => data.papers.length));

  return (
    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
      {categories.map(([specialty, data]) => (
        <div key={specialty} className="bg-gray-50 p-6 rounded-xl border border-gray-200 hover:shadow-md transition-all duration-200">
          <h3 className="text-xl font-semibold text-gray-900 mb-3">
            {specialty}
          </h3>
          <p className="text-blue-600 font-bold text-lg">
            {data.papers.length} paper{data.papers.length !== 1 ? 's' : ''}
          </p>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-3">
            <div 
              className="bg-blue-600 h-2 rounded-full" 
              style={{
                width: `${Math.min(100, (data.papers.length / maxPapers) * 100)}%`
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function LoadingState() {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-sm p-12 text-center border border-gray-100">
        <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-blue-600 mx-auto mb-6" />
        <div className="text-3xl font-bold text-gray-900 mb-4">Loading MedDigest...</div>
        <div className="text-gray-600">Please wait while we fetch the latest newsletter</div>
      </div>
    </div>
  );
}

function ErrorState({ error }: { error: string }) {
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="bg-white rounded-2xl shadow-sm p-12 text-center border border-red-200">
        <div className="text-4xl mb-6">⚠️</div>
        <div className="text-3xl font-bold text-gray-900 mb-6">Error Loading Newsletter</div>
        <div className="text-gray-700 mb-6">{error}</div>
        <div className="text-sm text-gray-600 bg-gray-50 rounded-lg p-4">
          Make sure:
          <br />1. Python API server is running (python api.py)
          <br />2. You've run the Python script to generate the newsletter first (python main.py)
        </div>
      </div>
    </div>
  );
}

export default function Newspaper() {
  const [newsletterData, setNewsletterData] = useState<NewsletterData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchNewsletter = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/newsletter');
        if (!response.ok) {
          throw new Error('Failed to fetch newsletter data');
        }
        const data = await response.json();
        if (data.error) {
          throw new Error(data.error);
        }
        setNewsletterData(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchNewsletter();
  }, []);

  if (loading) return <LoadingState />;
  if (error) return <ErrorState error={error} />;
  if (!newsletterData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-sm p-12 text-center border border-gray-100">
          <div className="text-4xl mb-6">📄</div>
          <div className="text-3xl font-bold text-gray-900 mb-6">No Newsletter Data</div>
          <div className="text-gray-600">Please run the Python script to generate a newsletter first.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <NewspaperBanner date={newsletterData.date_generated} totalPapers={newsletterData.total_papers} />
      <NavTabs />
      
      <main className="max-w-6xl mx-auto px-6 py-16">
        <NewsletterSection 
          title="Executive Summary" 
          icon="📋"
          fallback="This week's digest includes research from various medical specialties with cutting-edge developments and clinical insights."
        >
          <div className="prose prose-gray max-w-none">
            <p className="text-gray-700 leading-relaxed text-lg">
              {newsletterData.executive_summary}
            </p>
          </div>
        </NewsletterSection>

        <NewsletterSection title="Key Discoveries" icon="🔬">
          <KeyDiscoveriesList discoveries={newsletterData.key_discoveries || []} />
        </NewsletterSection>

        <NewsletterSection 
          title="Emerging Trends" 
          icon="📈"
          fallback="AI and machine learning continue to advance medical research with innovative approaches to diagnosis and treatment."
        >
          <div className="prose prose-gray max-w-none">
            <p className="text-gray-700 leading-relaxed text-lg">
              {newsletterData.emerging_trends}
            </p>
          </div>
        </NewsletterSection>

        <NewsletterSection 
          title="Cross-Specialty Insights" 
          icon="🔗"
          fallback="Interdisciplinary collaboration continues to drive innovation across medical specialties."
        >
          <div className="prose prose-gray max-w-none">
            <p className="text-gray-700 leading-relaxed text-lg">
              {newsletterData.cross_specialty_insights}
            </p>
          </div>
        </NewsletterSection>

        <NewsletterSection 
          title="Clinical Implications" 
          icon="🏥"
          fallback="These research findings have potential implications for clinical practice and patient care."
        >
          <div className="prose prose-gray max-w-none">
            <p className="text-gray-700 leading-relaxed text-lg">
              {newsletterData.clinical_implications}
            </p>
          </div>
        </NewsletterSection>

        <NewsletterSection 
          title="Research Gaps" 
          icon="🔍"
          fallback="Analysis of current research gaps and areas requiring further investigation."
        >
          <div className="prose prose-gray max-w-none">
            <p className="text-gray-700 leading-relaxed text-lg">
              {newsletterData.research_gaps}
            </p>
          </div>
        </NewsletterSection>

        <NewsletterSection 
          title="Future Directions" 
          icon="🚀"
          fallback="Emerging directions and potential future developments in medical research."
        >
          <div className="prose prose-gray max-w-none">
            <p className="text-gray-700 leading-relaxed text-lg">
              {newsletterData.future_directions}
            </p>
          </div>
        </NewsletterSection>

        <NewsletterSection title="Research Categories" icon="📊">
          {newsletterData.specialty_data && (
            <ResearchCategories specialtyData={newsletterData.specialty_data} />
          )}
        </NewsletterSection>

        <footer className="text-center text-gray-500 py-12">
          <div className="bg-white rounded-2xl shadow-sm p-8 border border-gray-100">
            <p className="text-lg font-semibold text-gray-900">Thank you for reading MedDigest!</p>
            <p className="mt-2 text-gray-600">© 2025 MedDigest. All rights reserved.</p>
          </div>
        </footer>
      </main>
    </div>
  );
}
