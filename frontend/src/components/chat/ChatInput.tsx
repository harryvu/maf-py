'use client';

import { useRef, useEffect } from 'react';

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled?: boolean;
  isLoading?: boolean;
  autoFocus?: boolean;
  maxLength?: number;
}

/**
 * Chat input component with send button
 */
export function ChatInput({
  value,
  onChange,
  onSend,
  disabled = false,
  isLoading = false,
  autoFocus = false,
  maxLength,
}: ChatInputProps) {
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus();
    }
  }, [autoFocus]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if (disabled || isLoading) return;
    if (!value.trim()) return;
    onSend();
  };

  const isButtonDisabled = disabled || isLoading || !value.trim();

  return (
    <div className="flex gap-2">
      <div className="flex-1 relative">
        <textarea
          ref={inputRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled || isLoading}
          placeholder="Type your message..."
          aria-label="Message input"
          maxLength={maxLength}
          rows={2}
          className={`
            w-full px-3 py-2 border border-gray-300 rounded-md text-sm
            focus:outline-none focus:ring-2 focus:ring-blue-500
            resize-none
            ${disabled ? 'bg-gray-100 cursor-not-allowed' : ''}
          `}
        />
        {maxLength && (
          <span className="absolute bottom-1 right-2 text-xs text-gray-400">
            {value.length}/{maxLength}
          </span>
        )}
      </div>
      
      <button
        onClick={handleSend}
        disabled={isButtonDisabled}
        aria-label="Send message"
        className={`
          px-4 py-2 rounded-md font-medium text-white transition-colors
          ${isButtonDisabled
            ? 'bg-gray-300 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700'
          }
        `}
      >
        {isLoading ? (
          <span className="animate-pulse">Sending...</span>
        ) : (
          'Send'
        )}
      </button>
    </div>
  );
}
