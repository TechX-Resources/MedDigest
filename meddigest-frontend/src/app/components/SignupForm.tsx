'use client';

import { useState, useEffect, useRef } from 'react';

const INTERESTS = [
  "Cardiology", "Oncology", "Neurology", "Psychiatry", "Pediatrics", "Internal Medicine",
  "Surgery", "Emergency Medicine", "Radiology", "Pathology", "Anesthesiology", "Dermatology",
  "Endocrinology", "Gastroenterology", "Hematology", "Infectious Disease", "Nephrology",
  "Ophthalmology", "Orthopedics", "Otolaryngology", "Pulmonology", "Rheumatology",
  "Urology", "Obstetrics and Gynecology", "Family Medicine", "Preventive Medicine",
  "Public Health", "Epidemiology", "Biostatistics", "Medical Genetics", "Immunology",
  "Pharmacology", "Toxicology", "Medical Education", "Health Policy", "Medical Ethics",
  "Rehabilitation Medicine", "Sports Medicine", "Geriatrics", "Palliative Care",
  "Critical Care", "Intensive Care", "Trauma Surgery", "Plastic Surgery", "Neurosurgery",
  "Cardiothoracic Surgery", "Vascular Surgery", "Transplant Surgery", "Medical Imaging",
  "Nuclear Medicine", "Interventional Radiology", "Radiation Oncology", "Medical Oncology",
  "Surgical Oncology", "Gynecologic Oncology", "Pediatric Oncology", "Hematologic Oncology"
];

interface FormData {
  firstName: string;
  lastName: string;
  email: string;
  medicalInterests: string[];
}

function InterestDropdown({ 
  interests, 
  selectedInterests, 
  onToggleInterest, 
  isOpen, 
  onToggle, 
  searchTerm, 
  onSearchChange 
}: {
  interests: string[];
  selectedInterests: string[];
  onToggleInterest: (interest: string) => void;
  isOpen: boolean;
  onToggle: () => void;
  searchTerm: string;
  onSearchChange: (term: string) => void;
}) {
  const filteredInterests = interests.filter(interest =>
    interest.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="relative">
      <label className="block mb-2 font-medium text-gray-900">Medical Interests:</label>
      
      <button
        type="button"
        onClick={onToggle}
        className="w-full p-3 border rounded-lg text-gray-900 bg-white text-left flex justify-between items-center"
      >
        <span>
          {selectedInterests.length > 0 
            ? `${selectedInterests.length} selected` 
            : 'Select medical interests'}
        </span>
        <span className={`transform transition-transform ${isOpen ? 'rotate-180' : ''}`}>
          ▼
        </span>
      </button>

      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white border rounded-lg shadow-lg max-h-64 overflow-y-auto">
          <div className="sticky top-0 bg-white border-b p-2">
            <input
              type="text"
              placeholder="Search medical interests..."
              value={searchTerm}
              onChange={(e) => onSearchChange(e.target.value)}
              className="w-full p-2 border rounded text-gray-900 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              onClick={(e) => e.stopPropagation()}
            />
          </div>
          
          <div className="p-2">
            {filteredInterests.length > 0 ? (
              filteredInterests.map(interest => (
                <label
                  key={interest}
                  className="flex items-center p-2 hover:bg-gray-50 cursor-pointer rounded"
                >
                  <input
                    type="checkbox"
                    checked={selectedInterests.includes(interest)}
                    onChange={() => onToggleInterest(interest)}
                    className="mr-3"
                  />
                  <span className="text-gray-900">{interest}</span>
                </label>
              ))
            ) : (
              <div className="p-2 text-gray-500 text-sm text-center">
                No interests found matching "{searchTerm}"
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function SelectedInterests({ interests, onRemove }: { interests: string[]; onRemove: (interest: string) => void }) {
  if (interests.length === 0) return null;

  return (
    <div className="mt-2">
      <p className="text-sm text-gray-600 mb-1">Selected:</p>
      <div className="flex flex-wrap gap-1">
        {interests.map(interest => (
          <span
            key={interest}
            className="inline-flex items-center px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full"
          >
            {interest}
            <button
              type="button"
              onClick={() => onRemove(interest)}
              className="ml-1 hover:bg-blue-200 rounded-full w-4 h-4 flex items-center justify-center"
            >
              ×
            </button>
          </span>
        ))}
      </div>
    </div>
  );
}

function MessageDisplay({ message }: { message: string }) {
  if (!message) return null;

  const isSuccess = message.includes('Thanks');
  return (
    <div className={`p-3 rounded ${isSuccess ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
      {message}
    </div>
  );
}

export default function SignupForm() {
  const [formData, setFormData] = useState<FormData>({
    firstName: '',
    lastName: '',
    email: '',
    medicalInterests: []
  });
  const [message, setMessage] = useState('');
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      const response = await fetch('http://localhost:8000/api/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: formData.email,
          first_name: formData.firstName,
          last_name: formData.lastName,
          medical_interests: formData.medicalInterests
        }),
      });

      const data = await response.json();
      setMessage(data.message);
      
      if (data.success) {
        setFormData({ firstName: '', lastName: '', email: '', medicalInterests: [] });
      }
    } catch (error) {
      setMessage('Error: Please check if API is running');
    }
  };

  const toggleInterest = (interest: string) => {
    setFormData(prev => ({
      ...prev,
      medicalInterests: prev.medicalInterests.includes(interest)
        ? prev.medicalInterests.filter(i => i !== interest)
        : [...prev.medicalInterests, interest]
    }));
  };

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
    if (!isDropdownOpen) {
      setSearchTerm('');
    }
  };

  const updateField = (field: keyof FormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <form onSubmit={handleSubmit} className="max-w-lg mx-auto space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <input
          type="text"
          placeholder="First Name"
          value={formData.firstName}
          onChange={(e) => updateField('firstName', e.target.value)}
          className="p-3 border rounded-lg text-gray-900 bg-white"
          required
        />
        <input
          type="text"
          placeholder="Last Name"
          value={formData.lastName}
          onChange={(e) => updateField('lastName', e.target.value)}
          className="p-3 border rounded-lg text-gray-900 bg-white"
          required
        />
      </div>
      
      <input
        type="email"
        placeholder="Email"
        value={formData.email}
        onChange={(e) => updateField('email', e.target.value)}
        className="w-full p-3 border rounded-lg text-gray-900 bg-white"
        required
      />
      
      <div ref={dropdownRef}>
        <InterestDropdown
          interests={INTERESTS}
          selectedInterests={formData.medicalInterests}
          onToggleInterest={toggleInterest}
          isOpen={isDropdownOpen}
          onToggle={toggleDropdown}
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
        />
        <SelectedInterests 
          interests={formData.medicalInterests} 
          onRemove={toggleInterest} 
        />
      </div>
      
      <button
        type="submit"
        className="w-full bg-blue-600 text-white p-3 rounded-lg font-bold hover:bg-blue-700"
      >
        Sign Up for MedDigest
      </button>
      
      <MessageDisplay message={message} />
    </form>
  );
} 