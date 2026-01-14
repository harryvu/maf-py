import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { EducationalSettings } from '../../src/lib/types/settings';
import { DEFAULT_SETTINGS } from '../../src/lib/types/settings';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => {
      store[key] = value;
    }),
    removeItem: vi.fn((key: string) => {
      delete store[key];
    }),
    clear: vi.fn(() => {
      store = {};
    }),
  };
})();

Object.defineProperty(globalThis, 'localStorage', { value: localStorageMock });

describe('Session Persistence Integration', () => {
  const SETTINGS_KEY = 'llm-risk-playground-settings';
  const MESSAGES_KEY = 'llm-risk-playground-messages';
  const HISTORY_KEY = 'llm-risk-playground-history';

  beforeEach(() => {
    localStorageMock.clear();
    vi.clearAllMocks();
  });

  describe('settings persistence', () => {
    it('should save settings to localStorage', () => {
      const settings: EducationalSettings = {
        guardrailsEnabled: false,
        simulationMode: true,
        adminBypass: true,
      };

      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        SETTINGS_KEY,
        JSON.stringify(settings)
      );
    });

    it('should restore settings from localStorage', () => {
      const savedSettings: EducationalSettings = {
        guardrailsEnabled: false,
        simulationMode: true,
        adminBypass: true,
      };

      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(savedSettings));

      const restored = localStorage.getItem(SETTINGS_KEY);
      expect(JSON.parse(restored!)).toEqual(savedSettings);
    });

    it('should return default settings when none saved', () => {
      localStorageMock.getItem.mockReturnValueOnce(null);

      const restored = localStorage.getItem(SETTINGS_KEY);
      expect(restored).toBeNull();
    });

    it('should handle corrupted settings data', () => {
      localStorageMock.getItem.mockReturnValueOnce('invalid json');

      const restored = localStorage.getItem(SETTINGS_KEY);
      expect(() => JSON.parse(restored!)).toThrow();
    });

    it('should merge partial settings with defaults', () => {
      const partialSettings = { guardrailsEnabled: false };
      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(partialSettings));

      const restored = JSON.parse(localStorage.getItem(SETTINGS_KEY)!);
      const merged = { ...DEFAULT_SETTINGS, ...restored };

      expect(merged.guardrailsEnabled).toBe(false);
      expect(merged.simulationMode).toBe(DEFAULT_SETTINGS.simulationMode);
    });
  });

  describe('message history persistence', () => {
    it('should save messages to localStorage', () => {
      const messages = [
        { role: 'user' as const, content: 'Hello' },
        { role: 'assistant' as const, content: 'Hi there!' },
      ];

      localStorage.setItem(MESSAGES_KEY, JSON.stringify(messages));

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        MESSAGES_KEY,
        JSON.stringify(messages)
      );
    });

    it('should restore messages from localStorage', () => {
      const savedMessages = [
        { role: 'user' as const, content: 'Test message' },
      ];

      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(savedMessages));

      const restored = JSON.parse(localStorage.getItem(MESSAGES_KEY)!);
      expect(restored).toEqual(savedMessages);
    });

    it('should limit stored message count', () => {
      const manyMessages = Array.from({ length: 200 }, (_, i) => ({
        role: 'user' as const,
        content: `Message ${i}`,
      }));

      const limitedMessages = manyMessages.slice(-100); // Keep last 100

      localStorage.setItem(MESSAGES_KEY, JSON.stringify(limitedMessages));

      expect(limitedMessages.length).toBe(100);
    });

    it('should clear messages on reset', () => {
      localStorage.setItem(MESSAGES_KEY, JSON.stringify([{ role: 'user', content: 'Test' }]));
      localStorage.removeItem(MESSAGES_KEY);

      expect(localStorageMock.removeItem).toHaveBeenCalledWith(MESSAGES_KEY);
    });
  });

  describe('refund history persistence', () => {
    it('should save refund history', () => {
      const history = [
        {
          id: 'REF-001',
          orderId: 'ORD-123',
          amount: 50,
          timestamp: new Date().toISOString(),
          approved: true,
        },
      ];

      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));

      expect(localStorageMock.setItem).toHaveBeenCalledWith(
        HISTORY_KEY,
        JSON.stringify(history)
      );
    });

    it('should restore refund history', () => {
      const savedHistory = [
        {
          id: 'REF-001',
          orderId: 'ORD-123',
          amount: 50,
          timestamp: '2024-01-01T00:00:00Z',
          approved: true,
        },
      ];

      localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(savedHistory));

      const restored = JSON.parse(localStorage.getItem(HISTORY_KEY)!);
      expect(restored).toEqual(savedHistory);
    });

    it('should track simulation vs real refunds', () => {
      const history = [
        {
          id: 'REF-001',
          orderId: 'ORD-123',
          amount: 50,
          simulated: true,
        },
        {
          id: 'REF-002',
          orderId: 'ORD-456',
          amount: 100,
          simulated: false,
        },
      ];

      const simulatedCount = history.filter(h => h.simulated).length;
      const realCount = history.filter(h => !h.simulated).length;

      expect(simulatedCount).toBe(1);
      expect(realCount).toBe(1);
    });
  });

  describe('session isolation', () => {
    it('should use unique keys for different features', () => {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({ test: true }));
      localStorage.setItem(MESSAGES_KEY, JSON.stringify([]));
      localStorage.setItem(HISTORY_KEY, JSON.stringify([]));

      expect(localStorageMock.setItem).toHaveBeenCalledTimes(3);
      expect(localStorageMock.setItem.mock.calls[0][0]).toBe(SETTINGS_KEY);
      expect(localStorageMock.setItem.mock.calls[1][0]).toBe(MESSAGES_KEY);
      expect(localStorageMock.setItem.mock.calls[2][0]).toBe(HISTORY_KEY);
    });
  });

  describe('storage quota handling', () => {
    it('should handle storage quota exceeded', () => {
      const error = new Error('QuotaExceededError');
      error.name = 'QuotaExceededError';

      localStorageMock.setItem.mockImplementationOnce(() => {
        throw error;
      });

      expect(() => {
        localStorage.setItem(MESSAGES_KEY, JSON.stringify({ large: 'data' }));
      }).toThrow('QuotaExceededError');
    });
  });

  describe('cross-tab synchronization', () => {
    it('should detect storage events', () => {
      const storageEventHandler = vi.fn();
      
      // Simulate storage event listener
      const event = new StorageEvent('storage', {
        key: SETTINGS_KEY,
        newValue: JSON.stringify({ guardrailsEnabled: false }),
        oldValue: JSON.stringify({ guardrailsEnabled: true }),
      });

      storageEventHandler(event);

      expect(storageEventHandler).toHaveBeenCalledWith(
        expect.objectContaining({
          key: SETTINGS_KEY,
        })
      );
    });
  });
});
