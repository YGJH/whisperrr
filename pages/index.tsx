import React, { useState, useEffect, useCallback } from 'react';

interface Job {
  id: string;
  status: 'running' | 'finished' | 'failed';
  progress?: {
    stage: string;
    percent: number;
    message: string;
  };
  log: string;
  summaryAvailable: boolean;
}

interface FormData {
  url: string;
  model: 'openai' | 'gemini' | 'ollama';
  systemPrompt: string;
  copyToPCloud: boolean;
}

// API base URL - uses Next.js rewrite proxy in development
// In production, update this to your Flask server URL if deployed separately
const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

export default function WhisperTranscription() {
  const [view, setView] = useState<'form' | 'status'>('form');
  const [formData, setFormData] = useState<FormData>({
    url: '',
    model: 'openai',
    systemPrompt: '',
    copyToPCloud: true
  });
  const [currentJob, setCurrentJob] = useState<Job | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Function to refresh job status manually
  const refreshJobStatus = useCallback(async () => {
    if (!currentJob?.id) return;
    
    try {
      const response = await fetch(`${API_BASE}/api/job/${currentJob.id}`);
      if (response.ok) {
        const data = await response.json();
        setCurrentJob({
          id: data.id,
          status: data.status,
          progress: data.progress || currentJob.progress,
          log: data.log || '',
          summaryAvailable: data.summaryAvailable || false
        });
      }
    } catch (err) {
      console.error('Failed to refresh job status:', err);
    }
  }, [currentJob?.id, currentJob?.progress]);

  // Real job submission to Flask backend
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE}/api/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to start job');
      }

      const data = await response.json();
      setCurrentJob({
        id: data.jobId,
        status: 'running',
        progress: {
          stage: 'initializing',
          percent: 0,
          message: 'Starting job...'
        },
        log: '',
        summaryAvailable: false
      });
      setView('status');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start transcription');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Fetch real progress updates from Flask backend
  useEffect(() => {
    if (currentJob?.status === 'running') {
      const interval = setInterval(async () => {
        try {
          const response = await fetch(`${API_BASE}/api/job/${currentJob.id}`);
          if (response.ok) {
            const data = await response.json();
            setCurrentJob({
              id: data.id,
              status: data.status,
              progress: data.progress || currentJob.progress,
              log: data.log || '',
              summaryAvailable: data.summaryAvailable || false
            });
          }
        } catch (err) {
          console.error('Failed to fetch job status:', err);
        }
      }, 2000); // Poll every 2 seconds
      
      return () => clearInterval(interval);
    }
  }, [currentJob?.status, currentJob?.id]);

  const scrollToBottom = useCallback(() => {
    const logContainer = document.getElementById('logContainer');
    if (logContainer) {
      logContainer.scrollTop = logContainer.scrollHeight;
    }
  }, []);

  const scrollToTop = useCallback(() => {
    const logContainer = document.getElementById('logContainer');
    if (logContainer) {
      logContainer.scrollTop = 0;
    }
  }, []);

  const copyLog = useCallback(() => {
    if (currentJob?.log) {
      navigator.clipboard.writeText(currentJob.log);
      alert('Log copied to clipboard!');
    }
  }, [currentJob?.log]);

  if (view === 'status' && currentJob) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-600 via-pink-600 to-purple-700 p-4">
        <div className="max-w-6xl mx-auto">
          <div className="bg-white rounded-2xl shadow-2xl overflow-hidden">
            {/* Header */}
            <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white p-8 text-center">
              <h1 className="text-3xl font-bold mb-3">🎯 Transcription Job Status</h1>
              <div className="inline-block bg-white bg-opacity-20 backdrop-blur-sm px-4 py-2 rounded-full font-mono text-sm">
                Job ID: {currentJob.id}
              </div>
            </div>

            <div className="p-8">
              {/* Status Badge */}
              <div className="mb-6">
                <span className={`inline-flex items-center px-4 py-2 rounded-full text-sm font-semibold uppercase tracking-wide ${
                  currentJob.status === 'running' ? 'bg-yellow-100 text-yellow-800' :
                  currentJob.status === 'finished' ? 'bg-green-100 text-green-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {currentJob.status === 'running' && (
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  )}
                  {currentJob.status}
                </span>
              </div>

              {/* Progress Section */}
              <div className="bg-gray-50 rounded-xl p-6 mb-8">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                  <div className="bg-white p-4 rounded-lg border-l-4 border-purple-500">
                    <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Current Stage</div>
                    <div className="text-lg font-semibold text-gray-900">{currentJob.progress?.stage || 'initializing'}</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg border-l-4 border-purple-500">
                    <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Progress</div>
                    <div className="text-lg font-semibold text-gray-900">{currentJob.progress?.percent || 0}%</div>
                  </div>
                  <div className="bg-white p-4 rounded-lg border-l-4 border-purple-500">
                    <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">Status Message</div>
                    <div className="text-lg font-semibold text-gray-900">{currentJob.progress?.message || 'starting...'}</div>
                  </div>
                </div>

                {/* Progress Bar */}
                <div className="w-full h-10 bg-gray-200 rounded-full overflow-hidden shadow-inner">
                  <div 
                    className="h-full bg-gradient-to-r from-purple-500 to-pink-500 transition-all duration-500 ease-out flex items-center justify-end pr-4"
                    style={{ width: `${currentJob.progress?.percent || 0}%` }}
                  >
                    <span className="text-white font-semibold text-sm">
                      {currentJob.progress?.percent || 0}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Download Button */}
              {currentJob.summaryAvailable && (
                <div className="text-center mb-6">
                  <a 
                    href={`${API_BASE}/api/download/${currentJob.id}`}
                    download
                    className="inline-block bg-green-500 hover:bg-green-600 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transform transition hover:scale-105"
                  >
                    📥 Download Summary
                  </a>
                </div>
              )}

              {/* Log Section */}
              <div className="mt-8">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold text-gray-900">📋 Execution Log</h2>
                  <div className="flex gap-2">
                    <button 
                      onClick={scrollToBottom}
                      className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition"
                    >
                      ⬇️ Bottom
                    </button>
                    <button 
                      onClick={scrollToTop}
                      className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition"
                    >
                      ⬆️ Top
                    </button>
                    <button 
                      onClick={copyLog}
                      className="bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition"
                    >
                      📋 Copy
                    </button>
                  </div>
                </div>
                <div 
                  id="logContainer"
                  className="bg-gray-900 text-green-400 p-6 rounded-lg font-mono text-sm leading-relaxed h-96 overflow-y-auto shadow-inner whitespace-pre-wrap"
                >
                  {currentJob.log}
                </div>
              </div>

              {/* Actions */}
              <div className="mt-8 text-center pt-6 border-t border-gray-200">
                <button 
                  onClick={() => {
                    setView('form');
                    setCurrentJob(null);
                    setFormData({ url: '', model: 'openai', systemPrompt: '', copyToPCloud: true });
                  }}
                  className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transform transition hover:scale-105 mr-4"
                >
                  🏠 Back to Home
                </button>
                <button 
                  onClick={refreshJobStatus}
                  className="bg-gray-600 hover:bg-gray-700 text-white font-semibold py-3 px-6 rounded-lg shadow-lg transform transition hover:scale-105"
                  disabled={!currentJob?.id}
                >
                  🔄 Refresh Status
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-pink-600 to-purple-700 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl p-8 md:p-10 max-w-2xl w-full transform transition-all hover:scale-105 duration-300">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-3">🎙️ Whisper Transcription</h1>
          <p className="text-gray-600">Transform YouTube videos into text summaries</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Error Message */}
          {error && (
            <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded-lg">
              <p className="text-sm text-red-700">
                ❌ {error}
              </p>
            </div>
          )}

          {/* YouTube URL */}
          <div>
            <label htmlFor="url" className="block text-sm font-semibold text-gray-700 mb-2">
              YouTube URL
            </label>
            <input
              type="text"
              id="url"
              value={formData.url}
              onChange={(e) => setFormData({ ...formData, url: e.target.value })}
              placeholder="https://www.youtube.com/watch?v=..."
              required
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-500 focus:ring-4 focus:ring-purple-100 transition"
            />
          </div>

          {/* AI Model Selection */}
          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-3">
              AI Model
            </label>
            <div className="grid grid-cols-3 gap-3">
              {(['openai', 'gemini', 'ollama'] as const).map((model) => (
                <label
                  key={model}
                  className={`relative flex items-center justify-center py-3 px-4 rounded-lg border-2 cursor-pointer transition-all transform hover:scale-105 ${
                    formData.model === model
                      ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white border-transparent shadow-lg'
                      : 'bg-white text-gray-700 border-gray-300 hover:border-purple-400'
                  }`}
                >
                  <input
                    type="radio"
                    name="model"
                    value={model}
                    checked={formData.model === model}
                    onChange={(e) => setFormData({ ...formData, model: e.target.value as any })}
                    className="sr-only"
                  />
                  <span className="font-medium capitalize">{model}</span>
                </label>
              ))}
            </div>
          </div>

          {/* System Prompt */}
          <div>
            <label htmlFor="systemPrompt" className="block text-sm font-semibold text-gray-700 mb-2">
              System Prompt (Optional)
            </label>
            <textarea
              id="systemPrompt"
              value={formData.systemPrompt}
              onChange={(e) => setFormData({ ...formData, systemPrompt: e.target.value })}
              placeholder="你是一個會將轉錄文字詳細總結成中文的助理，請用繁體中文回覆。"
              rows={3}
              className="w-full px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:border-purple-500 focus:ring-4 focus:ring-purple-100 transition resize-none"
            />
          </div>

          {/* Copy to pCloud */}
          <div>
            <label className="flex items-center p-4 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition">
              <input
                type="checkbox"
                checked={formData.copyToPCloud}
                onChange={(e) => setFormData({ ...formData, copyToPCloud: e.target.checked })}
                className="w-5 h-5 text-purple-600 border-2 border-gray-300 rounded focus:ring-purple-500 focus:ring-2 mr-3"
              />
              <span className="font-medium text-gray-700">📁 Copy summary to pCloud</span>
            </label>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-semibold rounded-lg shadow-lg transform transition hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
          >
            {isSubmitting ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </span>
            ) : (
              '🚀 Start Transcription'
            )}
          </button>

          {/* Info Box */}
          <div className="bg-blue-50 border-l-4 border-blue-400 p-4 rounded-lg">
            <p className="text-sm text-blue-700">
              ℹ️ The transcription process may take several minutes depending on video length. You'll be redirected to a status page to monitor progress.
            </p>
          </div>
        </form>
      </div>
    </div>
  );
}
