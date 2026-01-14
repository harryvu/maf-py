'use client';

import { useState, useCallback, useMemo } from 'react';
import type { SecurityAnalysis, RiskLevel } from '../lib/types/security';
import { analyzeForOwasp } from '../lib/security/owasp-analyzer';

/**
 * Custom hook for managing security analysis state
 */
export function useSecurityAnalysis() {
  const [analysis, setAnalysis] = useState<SecurityAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Analyze a message for security threats
   */
  const analyze = useCallback(async (message: string) => {
    if (!message.trim()) {
      setAnalysis(null);
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      // Simulate async operation for UI feedback
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const result = analyzeForOwasp(message);
      setAnalysis(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed';
      setError(errorMessage);
      setAnalysis(null);
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  /**
   * Clear the analysis
   */
  const clear = useCallback(() => {
    setAnalysis(null);
    setError(null);
  }, []);

  /**
   * Check if injection was detected
   */
  const isInjectionDetected = useMemo(() => {
    return analysis?.isInjectionDetected ?? false;
  }, [analysis]);

  /**
   * Get the risk level
   */
  const riskLevel: RiskLevel | undefined = useMemo(() => {
    return analysis?.riskLevel;
  }, [analysis]);

  return {
    analysis,
    isAnalyzing,
    error,
    analyze,
    clear,
    isInjectionDetected,
    riskLevel,
  };
}
