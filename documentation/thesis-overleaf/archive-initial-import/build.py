import json, os, re, shutil

d = json.load(open('doc.json'))
# This script now lives inside the project it builds (documentation/thesis-overleaf),
# so it writes in place -- only the generated sections/ get wiped, never the whole dir.
ROOT = '.'
if os.path.isdir(ROOT + '/sections'): shutil.rmtree(ROOT + '/sections')
os.makedirs(ROOT + '/sections')

ESC = {'\\':'\\textbackslash{}','{':'\\{','}':'\\}','&':'\\&','%':'\\%','$':'\\$',
       '#':'\\#','_':'\\_','~':'\\textasciitilde{}','^':'\\textasciicircum{}',
       '\t':'\\quad{}','\n':'\\\\{}\n'}

def esc(s):
    return ''.join(ESC.get(c, c) for c in s)

def render_runs(rs, trim=True):
    out = []
    for r in rs:
        t = esc(r['t'])
        if r['b'] and r['i']: t = '\\textbf{\\textit{' + t + '}}'
        elif r['b']: t = '\\textbf{' + t + '}'
        elif r['i']: t = '\\textit{' + t + '}'
        out.append(t)
    s = ''.join(out)
    if trim:
        # Word line-breaks at the very start/end of a paragraph are vertical
        # spacing only; a bare \\ there is illegal in LaTeX.
        s = re.sub(r'^(?:\\\\\{\}\n)+', '', s)
        s = re.sub(r'(?:\\\\\{\}\n)+$', '', s)
    return s

def plain(rs):
    return ''.join(r['t'] for r in rs)

# Chapter boundaries: (paragraph index, file name, numbered?, title_override).
# Five chapters, per the requested structure:
#   1 Introduction
#   2 Literature review (and Project Contribution -- kept verbatim, see below)
#   3 System Modelling and Problem formulation  (Learning-based methods lives
#     inside it as a \section)
#   4 Methodology  (Imitation Learning/Phase One, Generalization and Phase Two
#     Methodology live inside it as \section's)
#   5 References (unnumbered, as is conventional for a bibliography)
#
# Paragraph 488 is where the reference list starts, but the source docx has no
# heading word there at all -- paragraph 488 IS the text of citation [1], not
# a title. So this is the one spot where the chapter title is NOT taken
# verbatim from the source (there is nothing there to take): "References" is
# supplied, and paragraph 488 itself is then rendered normally as the first
# citation, right below the heading.
CHAPTERS = [(0,   '01-introduction',                            True,  None),
            (8,   '02-literature-review',                       True,  None),
            (14,  '03-system-modelling-and-problem-formulation',True,  None),
            (125, '04-methodology',                             True,  None),
            (488, '05-references',                               False, 'References')]
chapter_of = {i: (n, numbered, override) for i, n, numbered, override in CHAPTERS}

# Paragraphs that were already heading-like lines in the source and should
# read as a \section within their chapter (not a new chapter of their own).
# Value is a title override, or None to keep the source heading text verbatim.
# 143 and 299 are explicitly retitled at the user's request ("change the
# subtitle yes") -- everywhere else the exact source wording is kept.
SECTION_AT = {
    18:  None,                                # System Model:              -> 3.1
    59:  None,                                # Classical Control Stack:   -> 3.2
    104: None,                                # Learning-based methods:    -> 3.3
    127: None,                                # General discussion...:     -> 4.1
    141: None,                                # Implementation details:    -> 4.2
    143: 'Phase One : Imitation Learning',     # was "Imitation Learning / Phase One Methodology:" -> 4.3
    299: 'Phase Two : Reinforcement Learning', # was "Phase Two Methodology:"                      -> 4.4
}

# "Generalization:" is a \subsection nested inside 4.3 (Phase One), i.e. 4.3.1.
SUBSECTION_AT = {
    295: None,  # Generalization:
}

def render_table(b):
    ncol = max(len(r) for r in b['rows'])
    w = 0.92 / ncol
    lines = ['\\begin{longtable}{|' + ('p{%.3f\\textwidth}|' % w) * ncol + '}',
             '\\hline']
    for row in b['rows']:
        cells = []
        for cell in row:
            paras = [render_runs(pp) for pp in cell if plain(pp) != '']
            cells.append('\\par '.join(paras) if paras else '')
        cells += [''] * (ncol - len(cells))
        lines.append(' & '.join(cells) + ' \\\\ \\hline')
    lines += ['\\end{longtable}']
    return '\n'.join(lines)

files, cur, curname = [], None, None
i = 0
while i < len(d):
    new_chapter = i in chapter_of
    if new_chapter:
        if cur is not None: files.append((curname, cur))
        curname, numbered, override = chapter_of[i]
        cur = []
        title = override if override is not None else None  # resolved below
    b = d[i]
    if b['k'] == 'tbl':
        if new_chapter:
            raise RuntimeError('chapter boundary landed on a table, not a paragraph')
        cur.append(render_table(b)); i += 1; continue
    txt = plain(b['runs'])
    if new_chapter:
        if title is None:
            title = render_runs(b['runs']).rstrip()
        cur.append(('\\chapter{' if numbered else '\\chapter*{') + title + '}')
        if not numbered:
            cur.append('\\addcontentsline{toc}{chapter}{' + title + '}')
        if override is None:
            # the paragraph WAS the heading text -- it's fully consumed by the title
            i += 1; continue
        # else: this paragraph is real body content (e.g. citation [1]), not a
        # heading, so fall through and let the normal paragraph logic render it
    if i in SECTION_AT:
        sec_title = SECTION_AT[i] if SECTION_AT[i] is not None else render_runs(b['runs']).rstrip()
        cur.append('\\section{' + sec_title + '}')
        i += 1; continue
    if i in SUBSECTION_AT:
        sub_title = SUBSECTION_AT[i] if SUBSECTION_AT[i] is not None else render_runs(b['runs']).rstrip()
        cur.append('\\subsection{' + sub_title + '}')
        i += 1; continue
    if txt.strip() == '':
        i += 1; continue
    if b['num']:
        env = 'enumerate' if b['fmt'] == 'decimal' else 'itemize'
        items = []
        nid = b['num']
        while (i < len(d) and d[i]['k'] == 'p' and d[i]['num'] == nid
               and i not in chapter_of and i not in SECTION_AT and i not in SUBSECTION_AT):
            if plain(d[i]['runs']).strip() != '':
                items.append('  \\item ' + render_runs(d[i]['runs']))
            i += 1
        # Fold a bracketed definition/vector line ("[pL (t), RL (t), ...]")
        # that immediately follows into the last bullet, instead of letting it
        # fall through as its own flush-left paragraph. In the source docx
        # this line is a separate, non-list paragraph, but it is visually the
        # continuation of the preceding "Label ∈ R^n:" bullet -- rendering it
        # as a bare paragraph breaks it out to the left margin and is exactly
        # the "cooked indentation" this fixes (19 occurrences in this doc).
        while items:
            j = i
            while j < len(d) and d[j]['k'] == 'p' and plain(d[j]['runs']).strip() == '':
                j += 1
            if (j >= len(d) or d[j]['k'] != 'p' or j in chapter_of or j in SECTION_AT
                    or j in SUBSECTION_AT or d[j]['num'] is not None):
                break
            cand = plain(d[j]['runs']).strip()
            if not cand.startswith('['):
                break
            items[-1] += '\\par ' + render_runs(d[j]['runs'])
            i = j + 1
        cur.append('\\begin{%s}\n%s\n\\end{%s}' % (env, '\n'.join(items), env))
        continue
    cur.append(render_runs(b['runs']))
    i += 1
files.append((curname, cur))

for name, blocks in files:
    open('%s/sections/%s.tex' % (ROOT, name), 'w').write('\n\n'.join(blocks) + '\n')
    print(name, len(blocks))

main = r"""% Verbatim import of "thesis text.docx" -- the text is unmodified.
% Compile with LuaLaTeX  (Overleaf: Menu -> Compiler -> LuaLaTeX).
\RequirePackage{iftex}
\RequireLuaTeX
% report (not article) so we get real, numbered \chapter's that each start on
% their own fresh page automatically -- that's built into report.cls, no
% \newpage/\clearpage needed at each chapter.
\documentclass[12pt,a4paper,twoside]{report}

\usepackage{fontspec}
% Required layout: top/bottom 2 cm, outer (external) 2 cm, inner (binding) 3 cm.
% twoside + inner/outer (rather than left/right) makes the wide margin switch
% sides on facing pages, which is what an "inner margin" spec means for a book.
\usepackage[a4paper,twoside,top=2cm,bottom=2cm,inner=3cm,outer=2cm]{geometry}
\usepackage{setspace}
\onehalfspacing
\usepackage{parskip}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage{titlesec}

% --- Chapter styling: big, colored, numbered ("Chapter N" over a big title) -
\definecolor{chaptercolor}{RGB}{0,51,102}
\titleformat{\chapter}[display]
  {\normalfont\bfseries\color{chaptercolor}}
  {\Large\MakeUppercase{\chaptertitlename}~\Large\thechapter}
  {14pt}
  {\Huge\bfseries}
  [\vspace{4pt}{\color{chaptercolor}\titlerule[1.2pt]}\vspace{8pt}]
\titlespacing*{\chapter}{0pt}{0pt}{28pt}

% --- Unicode coverage -------------------------------------------------------
% The source text contains Greek letters, math operators and math-alphanumeric
% characters pasted as literal Unicode. This fallback chain lets them render as
% themselves. Two rules learned the hard way:
%   * a fallback font that is NOT installed is a fatal error, so each one is
%     probed with \IfFontExistsTF before being added;
%   * fallbacks must be given as font NAMES, not file names -- a ".ttf"/".otf"
%     here makes luaotfload choke ("invalid option 'ttf;-fallback'").
\directlua{thesisfbfonts = {}}
\newcommand{\thesisaddfb}[1]{%
  \IfFontExistsTF{#1}{\directlua{table.insert(thesisfbfonts, "#1:mode=node;")}}{}}
\thesisaddfb{DejaVuSans}
\thesisaddfb{STIXTwoMath-Regular}
\thesisaddfb{FreeSerif}
\thesisaddfb{NotoSansMath-Regular}
\directlua{pcall(luaotfload.add_fallback, "thesisfb", thesisfbfonts)}

% --- Body font: Times New Roman -----------------------------------------
% Times New Roman itself is a Microsoft-licensed font and is not shipped with
% Overleaf/TeX Live. Priority order:
%   1) the real "Times New Roman" -- used automatically if you upload its
%      .ttf files into this Overleaf project (any subfolder fontspec can see);
%   2) "Liberation Serif" -- Red Hat's metric-compatible clone of Times New
%      Roman (same glyph widths, so line breaks/page count match); ships on
%      many systems and is what LibreOffice substitutes for Times New Roman;
%   3) "TeX Gyre Termes" -- guaranteed present in every TeX Live install
%      (including Overleaf's), also a Times-metric-compatible clone.
% Whichever is found, ligature rewriting stays OFF: +tlig would silently turn
% '  into  ’, "  into  ”, << into « and -- into – in the PDF, which would break
% the character-for-character fidelity this import is meant to preserve.
\IfFontExistsTF{Times New Roman}{%
  \setmainfont{Times New Roman}[Ligatures=NoCommon,RawFeature={-tlig;fallback=thesisfb}]%
}{%
  \IfFontExistsTF{Liberation Serif}{%
    \setmainfont{Liberation Serif}[Ligatures=NoCommon,RawFeature={-tlig;fallback=thesisfb}]%
  }{%
    \setmainfont{TeX Gyre Termes}[Ligatures=NoCommon,RawFeature={-tlig;fallback=thesisfb}]%
  }%
}

% Long unbreakable pasted strings must not abort the build.
\sloppy
\setlength{\emergencystretch}{4em}
\hbadness=10000
\vbadness=10000

\title{Thesis text}
\author{}
\date{}

\begin{document}
\maketitle

\input{sections/01-introduction}
\input{sections/02-literature-review}
\input{sections/03-system-modelling-and-problem-formulation}
\input{sections/04-methodology}
\input{sections/05-references}

\end{document}
"""
open(ROOT + '/main.tex', 'w').write(main)
print('wrote', ROOT)
