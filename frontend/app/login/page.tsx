"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    // Fake login delay
    setTimeout(() => {
      localStorage.setItem('isAuthenticated', 'true');
      router.push('/');
    }, 1000);
  };

  return (
    <main style={{ 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center', 
      minHeight: 'calc(100vh - 60px)',
      background: 'radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.1) 0%, transparent 50%)'
    }}>
      <div className="glass fade-in" style={{ padding: '40px', width: '100%', maxWidth: '400px', borderRadius: '16px' }}>
        <div style={{ textAlign: 'center', marginBottom: '30px' }}>
          <div style={{ fontSize: '48px', marginBottom: '16px' }}>🔐</div>
          <h1 style={{ fontSize: '24px', fontWeight: 700 }}>Welcome Back</h1>
          <p style={{ color: 'var(--muted)', fontSize: '14px', marginTop: '8px' }}>
            Sign in to access your secure legal workspace
          </p>
        </div>

        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--muted)', marginBottom: '8px', textTransform: 'uppercase' }}>
              Work Email
            </label>
            <input 
              type="email" 
              required
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="attorney@firm.com"
              style={{ 
                width: '100%', padding: '12px 16px', borderRadius: '8px', 
                background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border)',
                color: 'var(--text)', outline: 'none', transition: 'border-color 0.2s'
              }}
              onFocus={e => e.target.style.borderColor = 'var(--accent)'}
              onBlur={e => e.target.style.borderColor = 'var(--border)'}
            />
          </div>

          <div>
            <label style={{ display: 'block', fontSize: '12px', fontWeight: 600, color: 'var(--muted)', marginBottom: '8px', textTransform: 'uppercase' }}>
              Password
            </label>
            <input 
              type="password" 
              required
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              style={{ 
                width: '100%', padding: '12px 16px', borderRadius: '8px', 
                background: 'rgba(0,0,0,0.2)', border: '1px solid var(--border)',
                color: 'var(--text)', outline: 'none', transition: 'border-color 0.2s'
              }}
              onFocus={e => e.target.style.borderColor = 'var(--accent)'}
              onBlur={e => e.target.style.borderColor = 'var(--border)'}
            />
          </div>

          <button 
            type="submit" 
            className="btn btn-primary" 
            disabled={loading || !email || !password}
            style={{ width: '100%', padding: '14px', justifyContent: 'center', marginTop: '10px' }}
          >
            {loading ? <span className="spinning">⏳</span> : 'Secure Sign In'}
          </button>
        </form>

        <p style={{ textAlign: 'center', color: 'var(--muted)', fontSize: '12px', marginTop: '24px' }}>
          Protected by enterprise-grade encryption.
        </p>
      </div>
    </main>
  );
}
