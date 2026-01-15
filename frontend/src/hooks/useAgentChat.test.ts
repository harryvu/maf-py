import { describe, it, expect, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';

import { useAgentChat } from './useAgentChat';

vi.mock('../app/actions/agent', () => {
  return {
    submitRefundRequest: vi.fn(),
  };
});

import { submitRefundRequest } from '../app/actions/agent';

describe('useAgentChat', () => {
  it('shows a refresh hint when Server Action is not found', async () => {
    vi.mocked(submitRefundRequest).mockRejectedValueOnce(
      new Error(
        'Error: Server Action "601a9015c5af6f01ad9efc4125fc510ea2263c4aa4" was not found on the server.'
      )
    );

    const { result } = renderHook(() =>
      useAgentChat({
        simulationMode: true,
        guardrailsEnabled: false,
        adminBypass: false,
      })
    );

    await act(async () => {
      await result.current.sendMessage('hello', 'ORD-12345', 10);
    });

    expect(result.current.error).toContain('Please refresh');
  });
});
