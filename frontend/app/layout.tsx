import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'LegalEagle — AI Contract Risk Analyzer',
  description: 'Analyze legal contracts for risk using BERT NER + LLM + Qdrant RAG',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav style={{
          borderBottom: '1px solid var(--border)',
          background: 'rgba(10,10,15,0.95)',
          backdropFilter: 'blur(12px)',
          position: 'sticky', top: 0, zIndex: 100,
          padding: '0 32px',
          display: 'flex', alignItems: 'center', height: 60,
        }}>
          <a href="/" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span style={{ fontSize: 22 }}>⚖️</span>
            <span style={{ fontWeight: 800, fontSize: 18, letterSpacing: '-0.5px' }}>
              Legal<span style={{ color: 'var(--accent)' }}>Eagle</span>
            </span>
          </a>
          <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--muted)' }}>
            AI-Powered Contract Risk Analysis
          </span>
        </nav>
        {children}
      </body>
    </html>
  )
}
