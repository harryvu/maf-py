import { renderHook, act } from '@testing-library/react';
import { describe, it, expect, vi, afterEach } from 'vitest';
import { useSecurityAnalysis } from './useSecurityAnalysis';

describe('useSecurityAnalysis', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('auto-dismisses injection analysis after 5 seconds', async () => {
    vi.useFakeTimers();

    const { result } = renderHook(() => useSecurityAnalysis());

    await act(async () => {
      const promise = result.current.analyze('admin override');
      vi.advanceTimersByTime(100);
      await promise;
    });

    expect(result.current.isInjectionDetected).toBe(true);
    expect(result.current.analysis).not.toBeNull();

    act(() => {
      vi.advanceTimersByTime(5_000);
    });

    expect(result.current.analysis).toBeNull();
    expect(result.current.isInjectionDetected).toBe(false);
  });

  it('clear() removes analysis immediately and cancels pending auto-dismiss', async () => {
    vi.useFakeTimers();

    const { result } = renderHook(() => useSecurityAnalysis());

    await act(async () => {
      const promise = result.current.analyze('admin override');
      vi.advanceTimersByTime(100);
      await promise;
    });

    act(() => {
      result.current.clear();
    });

    expect(result.current.analysis).toBeNull();

    act(() => {
      vi.advanceTimersByTime(5_000);
    });

    expect(result.current.analysis).toBeNull();
  });
});
