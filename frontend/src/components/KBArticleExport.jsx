import React from 'react';
import { Download } from 'lucide-react';

export default function KBArticleExport({ article }) {
  if (!article?.title) return null;

  const exportMarkdown = () => {
    const lines = [
      `# ${article.article_id || 'KB Article'}: ${article.title}`,
      '',
      `**Tags:** ${(article.tags || []).join(', ')}`,
      `**Fault Classification:** ${article.fault_classification || '—'}`,
      `**Affected Components:** ${(article.affected_components || []).join(', ')}`,
      '',
      '## Symptom Description',
      article.symptom_description || '',
      '',
      '## Root Cause Explanation',
      article.root_cause_explanation || '',
      '',
      '## Diagnostic Steps',
      ...(article.diagnostic_steps || []).map(
        s => `${s.step}. **${s.action}** — Expected: ${s.expected_output}`
      ),
      '',
      '## Resolution Procedure',
      ...(article.resolution_procedure || []).map(
        s => `${s.step}. ${s.action}${s.command ? `\n   \`${s.command}\`` : ''}`
      ),
      '',
      '## Verification',
      article.verification || '',
      '',
      '## Prevention',
      article.prevention || '',
      '',
      `---`,
      `*Created by: ${article.created_by || 'NOIA'} · Review required: ${article.review_required ? 'Yes' : 'No'}*`,
    ];

    const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `${(article.article_id || 'kb-article').replace(/\s+/g, '_')}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <button
      onClick={exportMarkdown}
      className="flex items-center gap-1.5 text-xs text-emerald-400 hover:text-emerald-300
                 transition-colors duration-150 mt-1"
    >
      <Download size={11} />
      Export KB Article (.md)
    </button>
  );
}
