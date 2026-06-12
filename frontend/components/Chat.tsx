"use client";

import { useState, useRef, useEffect } from 'react';

export default function Chat({ jobId }: { jobId: string }) {
  const [messages, setMessages] = useState<{role: 'user'|'ai', text: string}[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const onSend = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || loading) return;

    const userQ = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userQ }]);
    setLoading(true);

    try {
      // Create a placeholder for AI response
      setMessages(prev => [...prev, { role: 'ai', text: '' }]);

      const url = `http://localhost:8000/ask/stream?job_id=${encodeURIComponent(jobId)}&question=${encodeURIComponent(userQ)}`;
      const eventSource = new EventSource(url);

      eventSource.onmessage = (event) => {
        if (event.data === '[DONE]') {
          eventSource.close();
          setLoading(false);
          return;
        }

        try {
          const data = JSON.parse(event.data);
          setMessages(prev => {
            const newArr = [...prev];
            newArr[newArr.length - 1].text += data.token;
            return newArr;
          });
        } catch (err) {}
      };

      eventSource.onerror = () => {
        eventSource.close();
        setLoading(false);
      };

    } catch (err) {
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflowY: 'auto', padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', color: 'var(--muted)', marginTop: 40 }}>
            <span style={{ fontSize: 32 }}>💬</span>
            <p style={{ marginTop: 10 }}>Ask anything about this contract.</p>
          </div>
        )}
        
        {messages.map((m, i) => (
          <div key={i} style={{
            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
            background: m.role === 'user' ? 'var(--accent)' : 'var(--surface2)',
            padding: '12px 16px',
            borderRadius: 12,
            borderBottomRightRadius: m.role === 'user' ? 2 : 12,
            borderBottomLeftRadius: m.role === 'ai' ? 2 : 12,
            maxWidth: '85%',
            lineHeight: 1.5,
            border: m.role === 'ai' ? '1px solid var(--border)' : 'none',
            whiteSpace: 'pre-wrap'
          }}>
            {m.text || <span className="pulsing">...</span>}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div style={{ padding: 16, borderTop: '1px solid var(--border)', background: 'rgba(0,0,0,0.2)' }}>
        <form onSubmit={onSend} style={{ display: 'flex', gap: 10 }}>
          <input 
            type="text" 
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="E.g. What is the governing law?"
            style={{ 
              flex: 1, padding: '12px 16px', borderRadius: 8, 
              background: 'var(--surface)', border: '1px solid var(--border)',
              color: 'var(--text)', outline: 'none'
            }}
          />
          <button type="submit" className="btn btn-primary" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
