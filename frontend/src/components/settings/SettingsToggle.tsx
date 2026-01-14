'use client';

interface SettingsToggleProps {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  variant?: 'default' | 'warning' | 'danger';
}

/**
 * Toggle switch component for settings
 */
export function SettingsToggle({
  label,
  description,
  checked,
  onChange,
  disabled = false,
  variant = 'default',
}: SettingsToggleProps) {
  const handleClick = () => {
    if (!disabled) {
      onChange(!checked);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (!disabled) {
        onChange(!checked);
      }
    }
  };

  const variantClasses = {
    default: '',
    warning: 'toggle-warning',
    danger: 'toggle-danger',
  };

  const toggleBgClass = checked
    ? variant === 'danger'
      ? 'bg-red-600'
      : variant === 'warning'
      ? 'bg-yellow-500'
      : 'bg-blue-600'
    : 'bg-gray-300';

  return (
    <div className={`flex items-start gap-3 ${variantClasses[variant]}`}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        onClick={handleClick}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        className={`
          relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full
          border-2 border-transparent transition-colors duration-200 ease-in-out
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
          ${toggleBgClass}
          ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <span
          className={`
            pointer-events-none inline-block h-5 w-5 transform rounded-full
            bg-white shadow ring-0 transition duration-200 ease-in-out
            ${checked ? 'translate-x-5' : 'translate-x-0'}
          `}
        />
      </button>
      <div className="flex-1">
        <span className="text-sm font-medium text-gray-900">{label}</span>
        {description && (
          <p className="text-xs text-gray-500 mt-0.5">{description}</p>
        )}
      </div>
    </div>
  );
}
