'use client';

import type { SecurityAnalysis } from '../../lib/types/security';
import { VulnerabilityBadge } from './VulnerabilityBadge';

interface SecurityAnalysisDisplayProps {
  analysis: SecurityAnalysis | null;
}

const OWASP_NAMES: Record<string, string> = {
  LLM01: 'Prompt Injection',
  LLM02: 'Insecure Output Handling',
  LLM03: 'Training Data Poisoning',
  LLM04: 'Model Denial of Service',
  LLM05: 'Supply Chain Vulnerabilities',
  LLM06: 'Sensitive Information Disclosure',
  LLM07: 'Insecure Plugin Design',
  LLM08: 'Excessive Agency',
  LLM09: 'Overreliance',
  LLM10: 'Model Theft',
};

/**
 * Component to display security analysis results
 */
export function SecurityAnalysisDisplay({ analysis }: SecurityAnalysisDisplayProps) {
  if (!analysis) return null;

  const riskClasses: Record<string, string> = {
    low: 'risk-low bg-green-50 border-green-200',
    medium: 'risk-medium bg-yellow-50 border-yellow-200',
    high: 'risk-high bg-orange-50 border-orange-200',
    critical: 'risk-critical bg-red-50 border-red-200',
  };

  const riskTextClasses: Record<string, string> = {
    low: 'text-green-700',
    medium: 'text-yellow-700',
    high: 'text-orange-700',
    critical: 'text-red-700',
  };

  return (
    <div
      className={`rounded-lg border p-3 ${riskClasses[analysis.riskLevel]}`}
      data-expandable
      aria-live="polite"
    >
      <h3 className="text-sm font-semibold text-gray-800 mb-2" role="heading">
        Security Analysis
      </h3>

      {/* Detection Status */}
      <div className="flex items-center gap-2 mb-3">
        {analysis.isInjectionDetected ? (
          <span className="text-sm font-medium text-red-600">
            ⚠️ Injection Detected
          </span>
        ) : (
          <span className="text-sm font-medium text-green-600">
            ✓ No injection detected - Safe
          </span>
        )}
        <span className={`text-xs font-medium uppercase ${riskTextClasses[analysis.riskLevel]}`}>
          {analysis.riskLevel} risk
        </span>
      </div>

      {/* OWASP Category */}
      {analysis.owaspCategory && (
        <div className="mb-3 text-sm">
          <span className="font-medium text-gray-700">OWASP Category: </span>
          <span className="text-blue-600">
            {analysis.owaspCategory} - {OWASP_NAMES[analysis.owaspCategory] || 'Unknown'}
          </span>
        </div>
      )}

      {/* Detected Patterns */}
      {analysis.detectedPatterns.length > 0 && (
        <div className="mb-3">
          <span className="text-xs font-medium text-gray-600 block mb-1">
            Detected Patterns:
          </span>
          <div className="space-y-2">
            {analysis.detectedPatterns.map((pattern, index) => (
              <div key={index} className="bg-white rounded p-2 text-xs">
                <div className="flex items-center gap-2 mb-1">
                  <VulnerabilityBadge
                    category={pattern.category}
                    severity={pattern.severity}
                    compact
                  />
                </div>
                <div className="text-gray-600 italic">
                  "{pattern.matched}"
                </div>
                <div className="text-gray-700 mt-1">
                  {pattern.explanation}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Educational Notes */}
      {analysis.educationalNotes && analysis.educationalNotes.length > 0 && (
        <div className="border-t border-gray-200 pt-2 mt-2">
          <span className="text-xs font-medium text-gray-600 block mb-1">
            📚 Learn More - Educational Notes:
          </span>
          <ul className="text-xs text-gray-700 space-y-1 list-disc list-inside">
            {analysis.educationalNotes.map((note, index) => (
              <li key={index}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
