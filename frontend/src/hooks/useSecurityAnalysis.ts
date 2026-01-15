'use client';

import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import type { SecurityAnalysis, RiskLevel } from '../lib/types/security';
import { looksLikePromptInjection } from '../lib/agent/injection-patterns';
import { createSecurityAnalysis } from '../lib/types/security';

const AUTO_DISMISS_MS = 5_000;

/**
 * Custom hook for managing security analysis state
 */
export function useSecurityAnalysis() {
  const [analysis, setAnalysis] = useState<SecurityAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dismissTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clearDismissTimer = useCallback(() => {
    if (dismissTimerRef.current) {
      clearTimeout(dismissTimerRef.current);
      dismissTimerRef.current = null;
    }
  }, []);

  /**
   * Analyze a message for security threats
   */
  const analyze = useCallback(async (message: string) => {
    if (!message.trim()) {
      clearDismissTimer();
      setAnalysis(null);
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      // Simulate async operation for UI feedback
      await new Promise(resolve => setTimeout(resolve, 100));
      
      const detectionResult = looksLikePromptInjection(message);
      clearDismissTimer();
      setAnalysis(createSecurityAnalysis(detectionResult.patterns));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Analysis failed';
      setError(errorMessage);
      clearDismissTimer();
      setAnalysis(null);
    } finally {
      setIsAnalyzing(false);
    }
  }, [clearDismissTimer]);

  /**
   * Clear the analysis
   */
  const clear = useCallback(() => {
    clearDismissTimer();
    setAnalysis(null);
    setError(null);
  }, [clearDismissTimer]);

  useEffect(() => {
    if (!analysis?.isInjectionDetected) {
      clearDismissTimer();
      return;
    }

    clearDismissTimer();
    dismissTimerRef.current = setTimeout(() => {
      setAnalysis(null);
    }, AUTO_DISMISS_MS);

    return () => {
      clearDismissTimer();
    };
  }, [analysis, clearDismissTimer]);

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
