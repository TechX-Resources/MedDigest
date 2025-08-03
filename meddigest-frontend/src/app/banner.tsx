type BannerProps = {
    date?: string;
    totalPapers?: number;
  };

export default function Hero({ date, totalPapers }: BannerProps) {    
    return (
      <section className="bg-blue-50 py-12 text-center border-b">
        <div className="max-w-3xl mx-auto">
          <div className="inline-block bg-white shadow rounded-full px-6 py-2 mb-6 font-semibold text-blue-700 text-lg">
            MedDigest
          </div>
          <h1 className="text-4xl font-bold text-blue-900 mb-4">
            Weekly Medical Research Digest
          </h1>
          <p className="text-lg text-blue-800 mb-6">
            Stay up to date with the latest insights from the world of medical research
          </p>
          {date && totalPapers && (
            <div className="text-sm text-gray-500 text-center mt-4">
              Date: {date} | Total Papers Analyzed: {totalPapers}
            </div>
          )}
        </div>
      </section>
    );
  }