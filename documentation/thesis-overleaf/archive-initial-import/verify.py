import json, re, glob, difflib

d = json.load(open('doc.json'))
def plain(rs): return ''.join(r['t'] for r in rs)

# expected text stream from the docx (non-empty paragraphs + table cells, in order)
exp = []
for b in d:
    if b['k'] == 'p':
        t = plain(b['runs'])
        if t.strip() != '': exp.append(t)
    else:
        for row in b['rows']:
            for cell in row:
                for pp in cell:
                    t = plain(pp)
                    if t.strip() != '': exp.append(t)

# recovered text stream from generated .tex
src = ''
for f in sorted(glob.glob('sections/*.tex')):
    src += open(f).read() + '\n'

UN = [('\\textbackslash{}','\x00'),('\\quad{}','\t'),('\\textasciitilde{}','~'),
      ('\\textasciicircum{}','^'),('\\{','\x01'),('\\}','\x02'),
      ('\\&','&'),('\\%','%'),('\\$','$'),('\\#','#'),('\\_','_')]

def strip_macro(s, name):
    # remove \name{ ... } keeping the argument, brace-balanced
    out, i = '', 0
    tag = '\\' + name + '{'
    while i < len(s):
        if s.startswith(tag, i):
            i += len(tag); depth = 1; buf = ''
            while i < len(s) and depth:
                c = s[i]
                if c == '{': depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0: i += 1; break
                buf += c; i += 1
            out += strip_macro(buf, name)
        else:
            out += s[i]; i += 1
    return out

got = []
for line in src.split('\n'):
    l = line.strip()
    if l.startswith('\\begin{') or l.startswith('\\end{') or l == '': continue
    if l.startswith('\\hline'): continue
    # \addcontentsline just re-declares the TOC entry for an unnumbered
    # \chapter* -- it repeats that chapter's title, not new document text.
    if l.startswith('\\addcontentsline'): continue
    for name in ('chapter*', 'chapter', 'section*', 'section', 'subsection', 'textbf', 'textit'):
        l = strip_macro(l, name)
    l = l.replace('\\par ', '\x03')
    if l.endswith('\\\\ \\hline'): l = l[:-len('\\\\ \\hline')]
    l = l.replace('\\item ', '')
    l = l.replace('\\\\{}', '\n').replace('\\\\', '\n')
    for a, b in UN: l = l.replace(a, b)
    l = l.replace('\x00', '\\').replace('\x01', '{').replace('\x02', '}')
    for piece in l.split(' & '):
        for sub in piece.split('\x03'):
            if sub.strip() != '': got.append(sub)

# compare ignoring only leading/trailing whitespace introduced by line handling
e = [x.strip() for x in exp]
g = [x.strip() for x in got]
print('docx paragraphs:', len(e), ' tex paragraphs:', len(g))
bad = 0
for line in difflib.unified_diff(e, g, 'docx', 'tex', lineterm='', n=0):
    if line[:3] in ('---','+++','@@ '): continue
    bad += 1
    if bad <= 20: print(repr(line[:200]))
print('DIFF LINES:', bad)
print('char total docx', sum(len(x) for x in e), 'tex', sum(len(x) for x in g))
