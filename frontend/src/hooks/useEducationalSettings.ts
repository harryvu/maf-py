'use client';

import { useState, useCallback, useEffect } from 'react';
import type { EducationalSettings } from '../lib/types/settings';
import { DEFAULT_SETTINGS, isValidEducationalSettings } from '../lib/types/settings';

const STORAGE_KEY = 'llm-risk-playground-settings';

/**
 * Custom hook for managing educational settings with localStorage persistence
 */
export function useEducationalSettings() {
  const [settings, setSettings] = useState<EducationalSettings>(DEFAULT_SETTINGS);
  const [isLoaded, setIsLoaded] = useState(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return;
    
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (isValidEducationalSettings(parsed)) {
          setSettings(parsed);
        }
      }
    } catch (error) {
      console.error('Failed to load settings from localStorage:', error);
    }
    
    setIsLoaded(true);
  }, []);

  // Save settings to localStorage whenever they change
  useEffect(() => {
    if (!isLoaded || typeof window === 'undefined') return;
    
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (error) {
      console.error('Failed to save settings to localStorage:', error);
    }
  }, [settings, isLoaded]);

  /**
   * Update settings with partial values
   */
  const updateSettings = useCallback((updates: Partial<EducationalSettings>) => {
    setSettings(prev => ({ ...prev, ...updates }));
  }, []);

  /**
   * Reset settings to defaults
   */
  const resetSettings = useCallback(() => {
    setSettings(DEFAULT_SETTINGS);
  }, []);

  /**
   * Toggle guardrails enabled
   */
  const toggleGuardrails = useCallback(() => {
    setSettings(prev => ({ ...prev, guardrailsEnabled: !prev.guardrailsEnabled }));
  }, []);

  /**
   * Toggle simulation mode
   */
  const toggleSimulation = useCallback(() => {
    setSettings(prev => ({ ...prev, simulationMode: !prev.simulationMode }));
  }, []);

  /**
   * Toggle admin bypass
   */
  const toggleAdminBypass = useCallback(() => {
    setSettings(prev => ({ ...prev, adminBypass: !prev.adminBypass }));
  }, []);

  return {
    settings,
    isLoaded,
    updateSettings,
    resetSettings,
    toggleGuardrails,
    toggleSimulation,
    toggleAdminBypass,
  };
}
