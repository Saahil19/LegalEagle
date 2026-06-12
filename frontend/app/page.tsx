"use client";

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<'' | 'uploading' | 'analyzing' | 'error'>('');
  const [errorMsg, setErrorMsg] = useState('');
  const [isAuthChecking, setIsAuthChecking] = useState(true);
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  // Authentication Check
  useEffect(() => {
    const auth = localStorage.getItem('isAuthenticated');
    if (!auth) {
      router.push('/login');
    } else {
      setIsAuthChecking(false);
    }
  }, [router]);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const onUpload = async () => {
    if (!file) return;
    setStatus('uploading');
    setErrorMsg('');
    
    try {
      const formData = new FormData();
      formData.append("file", file);

      // 1. Upload
      const resUpload = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData
      });
      if (!resUpload.ok) throw new Error("Upload failed");
      const uploadData = await resUpload.json();
      const jobId = uploadData.job_id;

      // 2. Analyze
      setStatus('analyzing');
      const resAnalyze = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: jobId })
      });
      
      if (!resAnalyze.ok) throw new Error("Analysis failed");
      
      // 3. Redirect to report
      router.push(`/report/${jobId}`);

    } catch (err: any) {
      setStatus('error');
      setErrorMsg(err.message || "An error occurred during analysis.");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('isAuthenticated');
    router.push('/login');
  };

  if (isAuthChecking) {
    return <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><span className="spinning" style={{ fontSize: 40 }}>⚙️</span></div>;
  }

  return (
    <div style={{ position: 'relative' }}>
      
      {/* Background decorations */}
      <div style={{ position: 'absolute', top: -150, left: '20%', width: 500, height: 500, background: 'var(--accent-glow)', borderRadius: '50%', filter: 'blur(100px)', zIndex: -1, pointerEvents: 'none' }} />
      <div style={{ position: 'absolute', top: 200, right: '10%', width: 400, height: 400, background: 'rgba(16, 185, 129, 0.1)', borderRadius: '50%', filter: 'blur(100px)', zIndex: -1, pointerEvents: 'none' }} />

      <button 
        onClick={handleLogout} 
        className="btn btn-ghost" 
        style={{ position: 'absolute', top: 20, right: 30, zIndex: 100 }}
      >
        Sign Out
      </button>

      <main style={{ maxWidth: 1000, margin: '80px auto 100px auto', padding: '0 20px' }}>
        
        {/* Header Section */}
        <div style={{ textAlign: 'center', marginBottom: 60 }}>
          <div style={{ display: 'inline-block', padding: '6px 16px', background: 'rgba(99, 102, 241, 0.1)', border: '1px solid var(--accent)', color: 'var(--accent)', borderRadius: '30px', fontSize: 12, fontWeight: 700, letterSpacing: 1, marginBottom: 20, textTransform: 'uppercase' }}>
            Enterprise Grade
          </div>
          <h1 style={{ fontSize: 56, fontWeight: 800, marginBottom: 24, lineHeight: 1.1, letterSpacing: '-1px' }}>
            Automated Legal <br/><span style={{ color: 'var(--accent)' }}>Document Review</span>
          </h1>
          <p style={{ color: 'var(--muted)', fontSize: 20, maxWidth: 650, margin: '0 auto', lineHeight: 1.5 }}>
            Instantly upload contracts to extract key clauses, detect hidden liabilities, and receive actionable insights from our advanced AI auditor.
          </p>
        </div>

        {/* Upload Section */}
        <div style={{ maxWidth: 700, margin: '0 auto 80px auto' }}>
          {status === 'analyzing' || status === 'uploading' ? (
            <div className="glass fade-in" style={{ padding: 60, borderRadius: 24, textAlign: 'center', boxShadow: '0 20px 40px rgba(0,0,0,0.4)' }}>
              <div className="spinning" style={{ fontSize: 56, marginBottom: 24 }}>⚙️</div>
              <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 12 }}>
                {status === 'uploading' ? 'Uploading Document...' : 'Auditing Contract...'}
              </h2>
              <p style={{ color: 'var(--muted)', fontSize: 16 }}>
                Our AI is currently reading the document and identifying critical risk factors.<br/>This process typically takes 1-3 minutes.
              </p>
            </div>
          ) : (
            <div 
              className="glass fade-in"
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              style={{ 
                padding: '60px 40px', 
                borderRadius: 24,
                border: dragActive ? '2px dashed var(--accent)' : '2px dashed rgba(255,255,255,0.15)',
                background: dragActive ? 'rgba(99, 102, 241, 0.05)' : 'rgba(17, 17, 24, 0.6)',
                transition: 'all 0.3s ease',
                cursor: 'pointer',
                textAlign: 'center',
                boxShadow: '0 20px 40px rgba(0,0,0,0.4)'
              }}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".txt,.pdf"
                onChange={handleChange}
                style={{ display: "none" }}
              />
              <div style={{ fontSize: 56, marginBottom: 20, filter: 'drop-shadow(0 4px 6px rgba(0,0,0,0.5))' }}>📄</div>
              <h3 style={{ fontSize: 24, fontWeight: 600, marginBottom: 10, color: file ? 'var(--accent)' : 'var(--text)' }}>
                {file ? file.name : 'Drag & Drop your contract here'}
              </h3>
              <p style={{ color: 'var(--muted)', fontSize: 15, marginBottom: 32 }}>
                {file ? `File size: ${(file.size / 1024).toFixed(1)} KB` : 'Supported formats: .TXT, .PDF (Max 10MB)'}
              </p>
              
              <button 
                className="btn btn-primary" 
                onClick={(e) => { e.stopPropagation(); onUpload(); }}
                disabled={!file}
                style={{ padding: '16px 40px', fontSize: 18, borderRadius: 12, letterSpacing: '0.5px' }}
              >
                {file ? 'Begin AI Audit' : 'Select File'}
              </button>
              
              {status === 'error' && (
                <div style={{ marginTop: 24, padding: 16, background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: 8, color: 'var(--red)' }}>
                  {errorMsg}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Features Section */}
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 60 }}>
          <div style={{ textAlign: 'center', marginBottom: 40 }}>
            <h2 style={{ fontSize: 32, fontWeight: 700 }}>Why trust LegalEagle?</h2>
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 24 }}>
            <div className="glass" style={{ padding: 30, borderRadius: 16 }}>
              <div style={{ fontSize: 32, marginBottom: 16 }}>🔍</div>
              <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>Deep Clause Extraction</h3>
              <p style={{ color: 'var(--muted)', fontSize: 14, lineHeight: 1.6 }}>
                Automatically identifies and extracts governing laws, termination clauses, and confidentiality agreements with high precision.
              </p>
            </div>
            
            <div className="glass" style={{ padding: 30, borderRadius: 16 }}>
              <div style={{ fontSize: 32, marginBottom: 16 }}>🚦</div>
              <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>Automated Risk Scoring</h3>
              <p style={{ color: 'var(--muted)', fontSize: 14, lineHeight: 1.6 }}>
                Every contract is graded on a scale of 1-10. High-risk liabilities are instantly flagged for attorney review.
              </p>
            </div>
            
            <div className="glass" style={{ padding: 30, borderRadius: 16 }}>
              <div style={{ fontSize: 32, marginBottom: 16 }}>💬</div>
              <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>Interactive Chat Agent</h3>
              <p style={{ color: 'var(--muted)', fontSize: 14, lineHeight: 1.6 }}>
                Have questions about the contract? Chat directly with the document to find specific terms and obligations instantly.
              </p>
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}
