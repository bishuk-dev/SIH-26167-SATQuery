import React, { useState } from 'react';
import { PageShell } from '../components/PageShell';

interface Key { id: string; name: string; key: string; created: string; status: 'active' | 'revoked'; }

const initial: Key[] = [
  { id: '1', name: 'Production Key', key: 'sq_live_•••••••••••••••••••', created: 'Oct 12, 2023', status: 'active' },
  { id: '2', name: 'Testing Key',    key: 'sq_test_•••••••••••••••••••', created: 'Nov 05, 2023', status: 'active' },
];

export const ApiKeys = () => {
  const [keys, setKeys] = useState<Key[]>(initial);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState('');

  const revoke = (id: string) => setKeys(k => k.map(x => x.id === id ? { ...x, status: 'revoked' } : x));

  const generate = () => {
    if (!newName.trim()) return;
    const rand = Math.random().toString(36).slice(2, 24);
    setKeys(k => [...k, {
      id: Date.now().toString(),
      name: newName.trim(),
      key: `sq_live_${rand}`,
      created: new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
      status: 'active',
    }]);
    setNewName('');
    setShowNew(false);
  };

  return (
    <PageShell title="API Keys" subtitle="Programmatic access to SatQuery engines">
      <div className="max-w-4xl mx-auto space-y-6">
        <p className="text-slate-400 text-sm leading-relaxed">
          Manage your API keys for programmatic access to the SatQuery inference and observation engines.
        </p>

        {/* Keys table */}
        <div className="rounded-2xl border border-white/[0.08] overflow-hidden">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-white/[0.03] text-[11px] uppercase font-semibold text-slate-500 border-b border-white/[0.07]">
              <tr>
                <th className="px-6 py-3">Name</th>
                <th className="px-6 py-3">Key</th>
                <th className="px-6 py-3">Created</th>
                <th className="px-6 py-3">Status</th>
                <th className="px-6 py-3 text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {keys.map((k, i) => (
                <tr key={k.id} className={`border-b border-white/[0.05] hover:bg-white/[0.02] transition ${i === keys.length - 1 ? 'border-0' : ''}`}>
                  <td className="px-6 py-4 font-medium text-white">{k.name}</td>
                  <td className="px-6 py-4 font-mono text-cyan-400 text-xs">{k.key}</td>
                  <td className="px-6 py-4 text-slate-500 text-xs">{k.created}</td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex items-center gap-1.5 text-[10px] font-semibold px-2.5 py-1 rounded-full border ${k.status === 'active' ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-red-500/10 border-red-500/20 text-red-400'}`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${k.status === 'active' ? 'bg-emerald-400' : 'bg-red-400'}`} />
                      {k.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-right">
                    {k.status === 'active' && (
                      <button onClick={() => revoke(k.id)} className="text-xs text-slate-500 hover:text-red-400 transition-colors cursor-pointer" type="button">
                        Revoke
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Generate new */}
        {showNew ? (
          <div className="rounded-2xl border border-cyan-400/20 bg-cyan-500/[0.03] p-5 flex items-center gap-3">
            <input
              autoFocus
              value={newName}
              onChange={e => setNewName(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && generate()}
              placeholder="Key name (e.g. Production v2)"
              className="flex-1 bg-black/30 border border-white/10 rounded-lg px-4 py-2 text-sm text-white placeholder-slate-600 focus:outline-none focus:border-cyan-400/40"
            />
            <button onClick={generate} type="button" className="px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 text-xs font-bold transition-colors cursor-pointer">
              Create
            </button>
            <button onClick={() => setShowNew(false)} type="button" className="px-3 py-2 rounded-lg text-slate-500 hover:text-white text-xs transition-colors cursor-pointer">
              Cancel
            </button>
          </div>
        ) : (
          <button
            onClick={() => setShowNew(true)}
            type="button"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-sm transition-all active:scale-95 cursor-pointer shadow-lg shadow-cyan-500/20"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24">
              <line x1="12" x2="12" y1="5" y2="19" /><line x1="5" x2="19" y1="12" y2="12" />
            </svg>
            Generate New Key
          </button>
        )}
      </div>
    </PageShell>
  );
};
