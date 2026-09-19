import json, re, unicodedata
d=json.load(open('doc.json'))
def plain(rs): return ''.join(r['t'] for r in rs)
exp=[]
for b in d:
    if b['k']=='p':
        if plain(b['runs']).strip(): exp.append(plain(b['runs']))
    else:
        for row in b['rows']:
            for c in row:
                for pp in c:
                    if plain(pp).strip(): exp.append(plain(pp))

raw=open('pdf.txt').read()
# strip page-footer numbers: a number directly before a page break (form feed),
# either on its own line or glued onto the preceding word by the extractor
raw=re.sub(r'\d+\s*\f\s*', '', raw)
def norm(s):
    s=unicodedata.normalize('NFC', s)
    s=s.replace('​','').replace('⁡','')
    return re.sub(r'\s+',' ',s).strip()
# two readings of the PDF: hyphenated line breaks joined, and left alone
pdf_join=norm(re.sub(r'-\s*\n\s*','',raw))
pdf_keep=norm(raw)

miss=[]
for p in exp:
    n=norm(p)
    if n and n not in pdf_join and n not in pdf_keep:
        miss.append(p)
print('paragraphs checked:', len(exp))
print('NOT reproduced verbatim in the PDF:', len(miss))
for m in miss:
    n=norm(m); lo,hi=0,len(n)
    while lo<hi:
        mid=(lo+hi+1)//2
        if n[:mid] in pdf_join or n[:mid] in pdf_keep: lo=mid
        else: hi=mid-1
    print('  diverges:', repr(n[max(0,lo-45):lo+45]))
