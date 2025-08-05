import Link from "next/link";

export default function NavTabs() {
  return (
    <nav className="w-full bg-white border-b border-gray-200 shadow-sm sticky top-0 z-50">
      <div className="max-w-6xl mx-auto flex justify-center gap-1 py-4 px-6">
        <Link
          href="/"
          className="px-6 py-3 text-gray-700 font-semibold hover:text-blue-600 hover:bg-gray-50 rounded-lg transition-all duration-200"
        >
          Home
        </Link>
        
        <Link
          href="/about"
          className="px-6 py-3 text-gray-700 font-semibold hover:text-blue-600 hover:bg-gray-50 rounded-lg transition-all duration-200"
        >
          About
        </Link>
        
        <Link
          href="/newspaper"
          className="px-6 py-3 text-gray-700 font-semibold hover:text-blue-600 hover:bg-gray-50 rounded-lg transition-all duration-200"
        >
          Newsletter
        </Link>
      </div>
    </nav>
  );
}