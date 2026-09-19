import zipfile, xml.etree.ElementTree as ET, json
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
p='/media/hisham/New Volume/Masters/Internship and Thesis prep/FixedWingsCoopTransportation/documentation/thesis text.docx'
z=zipfile.ZipFile(p)
doc=ET.fromstring(z.read('word/document.xml'))
st=ET.fromstring(z.read('word/styles.xml'))
names={}
for s in st.iter(W+'style'):
    sid=s.get(W+'styleId'); n=s.find(W+'name')
    names[sid]=n.get(W+'val') if n is not None else sid
num_fmt={}
nx=ET.fromstring(z.read('word/numbering.xml'))
abs_={}
for a in nx.iter(W+'abstractNum'):
    lv=a.find(W+'lvl')
    abs_[a.get(W+'abstractNumId')]=lv.find(W+'numFmt').get(W+'val')
for n_ in nx.iter(W+'num'):
    num_fmt[n_.get(W+'numId')]=abs_.get(n_.find(W+'abstractNumId').get(W+'val'))
body=doc.find(W+'body')

def runtext(r):
    s=''
    for ch in r:
        tag=ch.tag
        if tag==W+'t': s+= ch.text or ''
        elif tag==W+'tab': s+='\t'
        elif tag in (W+'br',W+'cr'): s+='\n'
        elif tag==W+'noBreakHyphen': s+='-'
    return s

def runs(para):
    rs=[]
    for r in para.iter(W+'r'):
        t=runtext(r)
        if t=='': continue
        rpr=r.find(W+'rPr'); b=i=False
        if rpr is not None:
            eb=rpr.find(W+'b'); ei=rpr.find(W+'i')
            b= eb is not None and eb.get(W+'val') not in ('0','false')
            i= ei is not None and ei.get(W+'val') not in ('0','false')
        rs.append({'t':t,'b':b,'i':i})
    return rs

out=[]
for el in body:
    if el.tag==W+'p':
        ppr=el.find(W+'pPr'); style=num=ilvl=None
        if ppr is not None:
            ps=ppr.find(W+'pStyle')
            if ps is not None: style=names.get(ps.get(W+'val'),ps.get(W+'val'))
            np_=ppr.find(W+'numPr')
            if np_ is not None:
                ni=np_.find(W+'numId'); il=np_.find(W+'ilvl')
                num=ni.get(W+'val') if ni is not None else None
                ilvl=il.get(W+'val') if il is not None else None
        out.append({'k':'p','style':style,'num':num,'fmt':num_fmt.get(num),'ilvl':ilvl,'runs':runs(el)})
    elif el.tag==W+'tbl':
        rows=[]
        for tr in el.findall(W+'tr'):
            rows.append([[runs(pp) for pp in tc.findall(W+'p')] for tc in tr.findall(W+'tc')])
        out.append({'k':'tbl','rows':rows})
json.dump(out,open('doc.json','w'),ensure_ascii=False,indent=1)
print('blocks',len(out))
