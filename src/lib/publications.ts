import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parse } from 'bibtex-parse';

export interface Publication {
  key: string;
  title: string;
  authors: { name: string; isMe: boolean; url?: string }[];
  year: string;
  venue: string;
  url: string;
}

const COAUTHORS: Record<string, string> = {
  'Raluca Ada Popa': 'https://people.eecs.berkeley.edu/~raluca/',
  'Mayank Rathee': 'https://mayank0403.github.io/',
  'Henry Corrigan-Gibbs': 'https://people.csail.mit.edu/henrycg/',
};

function parseAuthors(raw: string): Publication['authors'] {
  return raw.split(' and ').map(name => {
    name = name.trim();
    return {
      name,
      isMe: name.includes('Yuwen Zhang'),
      url: COAUTHORS[name],
    };
  });
}

export function getPublications(): Publication[] {
  const bib = readFileSync(join(process.cwd(), 'src/content/papers.bib'), 'utf-8');
  const entries = parse(bib);

  return entries
    .filter((e): e is typeof e & { itemtype: 'entry' } => e.itemtype === 'entry')
    .map(entry => {
      const fields: Record<string, string> = {};
      for (const f of entry.fields) {
        fields[f.name.toLowerCase()] = f.value.replace(/[{}]/g, '');
      }
      return {
        key: entry.key,
        title: fields.title ?? '',
        authors: parseAuthors(fields.author ?? ''),
        year: fields.year ?? '',
        venue: fields.howpublished ?? '',
        url: fields.url ?? '',
      };
    })
    .sort((a, b) => Number(b.year) - Number(a.year));
}
