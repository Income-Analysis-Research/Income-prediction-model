export default function LoadingSpinner({ size = "md" }: { size?: "sm" | "md" | "lg" }) {
  const dim = { sm: "w-4 h-4", md: "w-8 h-8", lg: "w-12 h-12" }[size];
  return (
    <div className="flex items-center justify-center p-8">
      <div
        className={`${dim} rounded-full border-2 border-navy-600 border-t-blue-500 animate-spin`}
      />
    </div>
  );
}

export function InlineSpinner() {
  return (
    <div className="w-4 h-4 rounded-full border-2 border-navy-500 border-t-blue-400 animate-spin flex-shrink-0" />
  );
}

export function PageLoader({ message = "Loading data…" }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[40vh] gap-4">
      <div className="w-10 h-10 rounded-full border-2 border-navy-600 border-t-blue-500 animate-spin" />
      <p className="text-sm text-slate-500">{message}</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center min-h-[30vh] gap-4 p-8">
      <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center">
        <span className="text-red-400 text-xl">!</span>
      </div>
      <p className="text-sm text-slate-400 text-center max-w-sm">{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="btn-secondary text-xs">
          Retry
        </button>
      )}
    </div>
  );
}
