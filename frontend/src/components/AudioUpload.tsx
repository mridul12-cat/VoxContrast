import React, { useState } from "react";

interface AudioUploadProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (file: File, transcript: string) => Promise<void>;
}

export const AudioUpload: React.FC<AudioUploadProps> = ({ isOpen, onClose, onSubmit }) => {
  const [file, setFile] = useState<File | null>(null);
  const [transcript, setTranscript] = useState("");
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async () => {
    if (!file) {
      alert("Please select an audio file to analyze.");
      return;
    }
    setLoading(true);
    try {
      await onSubmit(file, transcript);
      onClose();
    } catch (e: any) {
      alert("Analysis error: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-white">Upload Audio for Speech Analysis</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition">
            ✕
          </button>
        </div>

        <div className="space-y-3 text-xs">
          <div>
            <label className="block font-medium text-slate-300 mb-1">
              Audio File (WAV, MP3, FLAC, OGG, max 50MB)
            </label>
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="w-full text-slate-400 bg-slate-800/80 border border-slate-700 rounded-xl file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 cursor-pointer"
            />
          </div>
          <div>
            <label className="block font-medium text-slate-300 mb-1">
              Transcript (Optional, enables forced alignment)
            </label>
            <textarea
              rows={3}
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              placeholder="Enter canonical transcript text..."
              className="w-full bg-slate-800 border border-slate-700 rounded-xl p-3 text-slate-200 placeholder-slate-500 focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            />
          </div>
        </div>

        <div className="flex items-center justify-end space-x-3 pt-2">
          <button onClick={onClose} className="px-4 py-2 rounded-xl text-slate-400 hover:text-white text-xs font-medium transition">
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition"
          >
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>
        </div>
      </div>
    </div>
  );
};
