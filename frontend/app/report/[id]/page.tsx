"use client";

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import Chat from '@/components/Chat';

export default function ReportPage() {
  const params = useParams();
  const id = params.id as string;
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/report/${id}`)
      .then(r => r.json())
      .then(d => {
        setData(d);
        setLoading(false);
      })
      .catch(e => {
        console.error(e);
        setLoading(false);
      });
  }, [id]);

  if (loading) {
    return <div style={{ padding: 100, textAlign: 'center', fontSize: 20 }}>Loading report...</div>;
  }

  if (!data || data.status === 'error') {
    return <div style={{ padding: 100, textAlign: 'center', color: 'var(--red)' }}>Failed to load report.</div>;
  }

  const score = data.overall_score || 0;
  const badgeClass = score <= 3 ? 'badge-low' : (score <= 7 ? 'badge-medium' : 'badge-high');
  const riskLabel = score <= 3 ? 'LOW RISK' : (score <= 7 ? 'MEDIUM RISK' : 'HIGH RISK');

  // Parse risk scores to array for rendering
  const risks = Object.entries(data.risk_scores || {}).map(([clause, info]: any) => ({
    clause, ...info
  })).sort((a, b) => b.score - a.score);

  return (
    <div style={{ display: 'flex', height: 'calc(100vh - 60px)' }}>
      
      {/* ── Left: Report Dashboard ── */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 40, borderRight: '1px solid var(--border)' }}>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 40 }}>
          <div>
            <h1 style={{ fontSize: 32, fontWeight: 700, marginBottom: 8 }}>{data.contract_name}</h1>
            <p style={{ color: 'var(--muted)' }}>Job ID: {id} • Analyzed on {new Date(data.created_at).toLocaleString()}</p>
          </div>
          <div className={`glass ${badgeClass}`} style={{ padding: '16px 24px', textAlign: 'center' }}>
            <div style={{ fontSize: 36, fontWeight: 800, lineHeight: 1 }}>{score}<span style={{ fontSize: 18, color: 'var(--muted)' }}>/10</span></div>
            <div style={{ fontSize: 12, fontWeight: 700, marginTop: 4, letterSpacing: 1 }}>{riskLabel}</div>
          </div>
        </div>

        {data.needs_human_review && (
          <div style={{ background: 'var(--red-glow)', border: '1px solid var(--red)', padding: 16, borderRadius: 8, marginBottom: 40, color: '#fca5a5', display: 'flex', gap: 12, alignItems: 'center' }}>
            <span style={{ fontSize: 24 }}>⚠️</span>
            <div>
              <strong>Attorney Review Required</strong>
              <p style={{ fontSize: 14, marginTop: 4, color: 'rgba(255,255,255,0.7)' }}>This contract contains clauses with a risk score &gt; 7. Do not sign without legal consultation.</p>
            </div>
          </div>
        )}

        <h2 style={{ fontSize: 20, marginBottom: 16, borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>Extracted Entities</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 16, marginBottom: 40 }}>
          {Object.entries(data.entities || {}).map(([entity, values]: any) => (
            <div key={entity} className="glass" style={{ padding: 16 }}>
              <div style={{ fontSize: 12, color: 'var(--muted)', fontWeight: 600, textTransform: 'uppercase', marginBottom: 8 }}>{entity.replace('_', ' ')}</div>
              {values.slice(0, 3).map((v: string, i: number) => (
                <div key={i} style={{ fontSize: 14, marginBottom: 4 }}>• {v}</div>
              ))}
            </div>
          ))}
        </div>

        <h2 style={{ fontSize: 20, marginBottom: 16, borderBottom: '1px solid var(--border)', paddingBottom: 8 }}>Clause-by-Clause Risk Analysis</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
          {risks.map((r: any) => (
            <div key={r.clause} className="glass" style={{ padding: 24, borderLeft: `4px solid ${r.score <= 3 ? 'var(--green)' : r.score <= 7 ? 'var(--yellow)' : 'var(--red)'}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 12 }}>
                <h3 style={{ fontSize: 18, fontWeight: 600 }}>{r.clause.replace('_', ' ')}</h3>
                <span className={`badge badge-${r.score <= 3 ? 'low' : r.score <= 7 ? 'medium' : 'high'}`}>
                  Score: {r.score}/10
                </span>
              </div>
              <p style={{ color: 'var(--muted)', fontSize: 14, fontStyle: 'italic', marginBottom: 12, padding: '8px 12px', background: 'var(--surface2)', borderRadius: 6 }}>
                "{r.text}..."
              </p>
              <p style={{ fontSize: 15, lineHeight: 1.6 }}>
                <strong style={{ color: 'var(--text)' }}>AI Reasoning:</strong> {r.reasoning}
              </p>
            </div>
          ))}
        </div>

      </div>

      {/* ── Right: Chat Interface ── */}
      <div style={{ width: 400, background: 'var(--bg)' }}>
        <Chat jobId={id} />
      </div>

    </div>
  );
}
