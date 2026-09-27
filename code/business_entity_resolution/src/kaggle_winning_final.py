# ==========================================================================
# ANSHIKA'S WINNING NOTEBOOK — Two-stage LightGBM+XGBoost + SIBLING features
#   + address-only blocking channel + expected-F0.5 decision.
#   Reproduces the 0.969-class approach (harshgitty58/Amazon_ML_Challenge).
#
# PARALLEL PLAN:
#   * Run Cells 1-4 NOW (setup, load, blocking, recall). Note your recall.
#   * WAIT for friend's blocking result. Set BLK_TOPK + USE_ADDR in Cell 2
#     to her best config, re-run Cell 4 to confirm recall, THEN run 5-10.
#   * Cell 8 prints honest macro-F0.5 — MUST beat 0.9184 before you run the
#     ~3h inference (Cell 9). If it doesn't, tune and re-run 6-8 first.
#
# Settings: Accelerator = GPU T4, Internet = ON. Attach amazondatasetnew.
# ==========================================================================

# ============================== CELL 1 — install + locate data =============
!pip install -q rapidfuzz unidecode lightgbm xgboost

import os, re, glob, gc, time, math, unicodedata, numpy as np, pandas as pd
from collections import defaultdict, Counter
from rapidfuzz import fuzz, distance
from rapidfuzz.process import cpdist
import lightgbm as lgb, xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

cands = glob.glob("/kaggle/input/**/student_resource", recursive=True)
if os.path.exists("/kaggle/working/student_resource"): cands.append("/kaggle/working/student_resource")
if not cands:
    import gdown, zipfile
    gdown.download("https://drive.google.com/uc?id=1ChbbSOdZO8g3AtbTF75P0SfvwdjBow40","/kaggle/working/sr.zip",quiet=False)
    with zipfile.ZipFile("/kaggle/working/sr.zip") as z: z.extractall("/kaggle/working/")
    cands=["/kaggle/working/student_resource"]
BASE=cands[0]; TR,TE=f"{BASE}/dataset/train",f"{BASE}/dataset/test"
OUT="/kaggle/working/output"; os.makedirs(OUT,exist_ok=True)
print("BASE =", BASE)


# ============================== CELL 2 — config + helpers ==================
SAMPLE_S1=50000; NEG_POOL=1_500_000; RS=42
BLK_TOPK=45           # <-- set to friend's best
BLK_CAP=600
USE_ADDR=True         # <-- set True if friend's address channel raised recall
CHUNK=25000
CONF_THR=0.70         # stage-1 prob to treat a match as a confident "sibling"

def _ascii(x): return unicodedata.normalize("NFKD",x).encode("ascii","ignore").decode()
LEGAL=sorted(["private limited","public limited","limited liability company","pvt limited","pvt ltd",
 "private ltd","public ltd","incorporated","corporation","company","limited","inc","llc","ltd","corp",
 "plc","llp","gmbh","sarl","sas","pvt","co"],key=len,reverse=True)
LEGAL_PAT=r"\b("+"|".join(re.escape(x) for x in LEGAL)+r")\b"
DBA_PAT=r"\b(dba|t\s*/?\s*a|trading as)\b"
ADDR_ABBR={r"\bst\b":"street",r"\brd\b":"road",r"\bave?\b":"avenue",r"\bblvd\b":"boulevard",r"\bdr\b":"drive",
 r"\bln\b":"lane",r"\bct\b":"court",r"\bhwy\b":"highway",r"\bpkwy\b":"parkway",r"\bapt\b":"apartment",
 r"\bste\b":"suite",r"\bfl\b":"floor",r"\bopp\b":"opposite"}
def _base(s):
    s=s.fillna("").astype(str).map(_ascii)
    s=s.str.lower().str.replace("&"," and ",regex=False).str.replace("@"," at ",regex=False)
    s=s.str.replace(r"[^a-z0-9\s]"," ",regex=True)
    return s.str.replace(r"\s+"," ",regex=True).str.strip()
def clean_names(s):
    b=_base(s).str.replace(DBA_PAT," ",regex=True).str.replace(LEGAL_PAT," ",regex=True)
    return b.str.replace(r"\s+"," ",regex=True).str.strip()
def clean_addrs(s):
    b=_base(s)
    for p,r in ADDR_ABBR.items(): b=b.str.replace(p,r,regex=True)
    return b.str.replace(r"\s+"," ",regex=True).str.strip()
def _hnum(a):
    m=re.search(r"\d+",a); return m.group(0) if m else ""
def _sdx(name):
    if not name: return ""
    m={"b":"1","f":"1","p":"1","v":"1","c":"2","g":"2","j":"2","k":"2","q":"2","s":"2","x":"2","z":"2",
       "d":"3","t":"3","l":"4","m":"5","n":"5","r":"6"}
    code=[name[0].upper()]
    for ch in name[1:]:
        d=m.get(ch,"")
        if d and d!=code[-1]: code.append(d)
    return "".join(code[:4]).ljust(4,"0")
def clean_df(df):
    df=df.copy()
    df["name_c"]=clean_names(df["business_name"]); df["addr_c"]=clean_addrs(df["business_address"])
    df["compact"]=df["name_c"].str.replace(" ","",regex=False)
    df["acompact"]=df["addr_c"].str.replace(" ","",regex=False)
    df["country_c"]=df["country"].fillna("").astype(str).str.lower().str.strip()
    df["nums"]=df["addr_c"].str.findall(r"\d+").apply(set)
    df["postal"]=df["addr_c"].str.findall(r"\b\d{5,6}\b").apply(set)
    df["hnum"]=df["addr_c"].map(_hnum)
    df["sdx"]=df["name_c"].map(lambda x:_sdx(x.split()[0]) if x.split() else "")
    return df
def arrays(df):
    return dict(id=df["entity_id"].to_numpy(),name=df["name_c"].to_numpy(),addr=df["addr_c"].to_numpy(),
        compact=df["compact"].to_numpy(),acompact=df["acompact"].to_numpy(),country=df["country_c"].to_numpy(),
        hnum=df["hnum"].to_numpy(),sdx=df["sdx"].to_numpy(),postal=df["postal"].tolist(),nums=df["nums"].tolist())

def rec_keys(nm,cmp,nu,po,ad,acmp,sdx):
    nt=nm.split(); at=ad.split(); k=[]
    if cmp:
        k.append(("c",cmp))
        if len(cmp)>=5: k.append(("p",cmp[:12]))
    if nt:
        k.append(("s"," ".join(sorted(nt)))); ft=nt[0]
        for n in nu: k.append(("nt",ft,n))
        for p in po: k.append(("pt",ft,p))
    if sdx: k.append(("sdx",sdx))
    for t in nt:
        if len(t)>=4: k.append(("t",t))
    for i in range(len(nt)-1): k.append(("b",nt[i],nt[i+1]))
    for p in po: k.append(("pa",p))
    if at:
        aft=at[0]
        for n in nu: k.append(("an",aft,n))
        if nt: k.append(("na",nt[0],aft))
    if USE_ADDR:
        if acmp and len(acmp)>=6: k.append(("ac",acmp[:14]))
        for i in range(len(at)-1): k.append(("ab",at[i],at[i+1]))
        for t in at:
            if len(t)>=5: k.append(("atk",t))
    return k
def build_index(A):
    idx=defaultdict(list)
    for i in range(len(A["name"])):
        for k in rec_keys(A["name"][i],A["compact"][i],A["nums"][i],A["postal"][i],A["addr"][i],A["acompact"][i],A["sdx"][i]):
            idx[k].append(i)
    return idx
def block_pairs(names,comps,nums,posts,addrs,acmps,sdxs,idx,topk,cap):
    qrep,cpos,votes=[],[],[]
    for qi in range(len(names)):
        v=Counter()
        for k in rec_keys(names[qi],comps[qi],nums[qi],posts[qi],addrs[qi],acmps[qi],sdxs[qi]):
            lst=idx.get(k)
            if not lst or len(lst)>cap: continue
            for c in lst: v[c]+=1
        if not v: continue
        for c,ct in v.most_common(topk):
            qrep.append(qi); cpos.append(c); votes.append(ct)
    return np.array(qrep,np.int64),np.array(cpos,np.int64),np.array(votes,np.float32)
def build_idf(names):
    df=Counter(); N=max(1,len(names))
    for nm in names:
        for t in set(nm.split()): df[t]+=1
    return {t:math.log(N/(1+c)) for t,c in df.items()}

# 49-feature builder (proven, from the 0.848 base)
FEAT=["vote_count","name_ratio","name_token_sort","name_token_set","name_partial","name_jaro","addr_ratio",
 "addr_token_set","addr_partial","name_tok_jaccard","name_first_eq","name_idf_overlap","addr_tok_jaccard",
 "addr_first_eq","addr_last_eq","house_num_match","number_overlap","postal_match","compact_equal","len_diff",
 "density_gap","density_count","source_flag","name_tfidf_cos","addr_tfidf_cos","soundex_eq","postal_prefix_match",
 "name_len","addr_len","name_token_count","addr_token_count","name_char_jaccard","addr_char_jaccard","hnum_len_diff",
 "name_qgram_score","addr_qgram_score","name_lev","addr_lev","shared_name","shared_addr","country_match",
 "n_ftok_len","a_ftok_len","n_ltok_len","a_ltok_len","dig_ratio_n","dig_ratio_a","compact_len_diff","name_contain"]
def build_feats(qA,cA,qg,cp,votes,flag,idf,qnt=None,cnt=None,qat=None,cat=None):
    n=len(qg); qn=qA["name"][qg]; cn=cA["name"][cp]; qa=qA["addr"][qg]; ca=cA["addr"][cp]
    f_nr=cpdist(qn,cn,scorer=fuzz.ratio,workers=-1,dtype=np.float32)
    f_nts=cpdist(qn,cn,scorer=fuzz.token_sort_ratio,workers=-1,dtype=np.float32)
    f_ntse=cpdist(qn,cn,scorer=fuzz.token_set_ratio,workers=-1,dtype=np.float32)
    f_np=cpdist(qn,cn,scorer=fuzz.partial_ratio,workers=-1,dtype=np.float32)
    f_jw=cpdist(qn,cn,scorer=distance.JaroWinkler.similarity,workers=-1,dtype=np.float32)*100
    f_ar=cpdist(qa,ca,scorer=fuzz.ratio,workers=-1,dtype=np.float32)
    f_atse=cpdist(qa,ca,scorer=fuzz.token_set_ratio,workers=-1,dtype=np.float32)
    f_apar=cpdist(qa,ca,scorer=fuzz.partial_ratio,workers=-1,dtype=np.float32)
    qc=qA["compact"][qg]; cc=cA["compact"][cp]
    f_cmp=np.where((qc!="")&(qc==cc),100.,0.).astype(np.float32)
    qh=qA["hnum"][qg]; ch=cA["hnum"][cp]
    f_hn=np.where((qh!="")&(qh==ch),100.,0.).astype(np.float32)
    qsd=qA["sdx"][qg]; csd=cA["sdx"][cp]
    f_sd=np.where((qsd!="")&(qsd==csd),100.,0.).astype(np.float32)
    qpost,cpost,qnum,cnum=qA["postal"],cA["postal"],qA["nums"],cA["nums"]
    z=lambda:np.empty(n,np.float32)
    f_njac,f_nfe,f_idf=z(),z(),z(); f_ajac,f_afe,f_ale=z(),z(),z(); f_post,f_ppost=z(),z(); f_num,f_len=z(),z()
    f_nl,f_al,f_ntc,f_atc=z(),z(),z(),z(); f_ncj,f_acj,f_hnd=z(),z(),z(); f_nqg,f_aqg,f_nlev,f_alev=z(),z(),z(),z()
    f_nsh,f_ash,f_cm=z(),z(),z(); f_nft,f_aft,f_nlt,f_alt=z(),z(),z(),z(); f_ndr,f_adr,f_cl,f_ct=z(),z(),z(),z()
    for i in range(n):
        a=qn[i].split(); b=cn[i].split(); sa,sb=set(a),set(b); inter=sa&sb; uni=sa|sb
        f_njac[i]=len(inter)/(len(uni) if uni else 1)*100
        f_nfe[i]=100. if(a and b and a[0]==b[0]) else 0.
        f_idf[i]=sum(idf.get(t,0.) for t in inter)
        aa=qa[i].split(); bb=ca[i].split(); saa,sbb=set(aa),set(bb); ai=saa&sbb; au=saa|sbb
        f_ajac[i]=len(ai)/(len(au) if au else 1)*100
        f_afe[i]=100. if(aa and bb and aa[0]==bb[0]) else 0.
        f_ale[i]=100. if(aa and bb and aa[-1]==bb[-1]) else 0.
        gq=int(qg[i]); cci=int(cp[i])
        f_post[i]=100. if(qpost[gq]&cpost[cci]) else 0.
        pm=0.
        for p1 in qpost[gq]:
            for p2 in cpost[cci]:
                if len(p1)>=3 and len(p2)>=3 and p1[:3]==p2[:3]: pm=100.
        f_ppost[i]=pm
        u=qnum[gq]|cnum[cci]; f_num[i]=len(qnum[gq]&cnum[cci])/(len(u) if u else 1)*100
        f_len[i]=len(cn[i])-len(qn[i])
        f_nl[i]=abs(len(qn[i])-len(cn[i])); f_al[i]=abs(len(qa[i])-len(ca[i]))
        f_ntc[i]=abs(len(a)-len(b)); f_atc[i]=abs(len(aa)-len(bb))
        cq,ccx=set(qn[i]),set(cn[i]); f_ncj[i]=len(cq&ccx)/(len(cq|ccx) if(cq|ccx) else 1)*100
        cqa,cca=set(qa[i]),set(ca[i]); f_acj[i]=len(cqa&cca)/(len(cqa|cca) if(cqa|cca) else 1)*100
        try: f_hnd[i]=abs(int(qh[i])-int(ch[i])) if(qh[i] and ch[i]) else 0.
        except: f_hnd[i]=0.
        f_nqg[i]=distance.Indel.similarity(qn[i],cn[i])*100 if(qn[i] and cn[i]) else 0.
        f_aqg[i]=distance.Indel.similarity(qa[i],ca[i])*100 if(qa[i] and ca[i]) else 0.
        f_nlev[i]=distance.Levenshtein.similarity(qn[i],cn[i])/max(1,max(len(qn[i]),len(cn[i])))*100
        f_alev[i]=distance.Levenshtein.similarity(qa[i],ca[i])/max(1,max(len(qa[i]),len(ca[i])))*100
        f_nsh[i]=len(inter); f_ash[i]=len(ai)
        f_cm[i]=100. if(qA["country"][gq]==cA["country"][cci]) else 0.
        f_nft[i]=abs((len(a[0]) if a else 0)-(len(b[0]) if b else 0))
        f_aft[i]=abs((len(aa[0]) if aa else 0)-(len(bb[0]) if bb else 0))
        f_nlt[i]=abs((len(a[-1]) if a else 0)-(len(b[-1]) if b else 0))
        f_alt[i]=abs((len(aa[-1]) if aa else 0)-(len(bb[-1]) if bb else 0))
        f_ndr[i]=abs(sum(c.isdigit() for c in qn[i])-sum(c.isdigit() for c in cn[i]))
        f_adr[i]=abs(sum(c.isdigit() for c in qa[i])-sum(c.isdigit() for c in ca[i]))
        f_cl[i]=abs(len(qc[i])-len(cc[i])); f_ct[i]=100. if(qn[i] in cn[i] or cn[i] in qn[i]) else 0.
    if qnt is not None and cnt is not None:
        f_ntf=(np.asarray(qnt[qg].multiply(cnt[cp]).sum(1)).ravel()*100).astype(np.float32)
        f_atf=(np.asarray(qat[qg].multiply(cat[cp]).sum(1)).ravel()*100).astype(np.float32)
    else: f_ntf=np.zeros(n,np.float32); f_atf=np.zeros(n,np.float32)
    d=pd.DataFrame({"g":qg,"nr":f_nr}); gmax=d.groupby("g")["nr"].transform("max").to_numpy()
    f_gap=(gmax-f_nr).astype(np.float32); d["cl"]=(f_nr>=gmax-5).astype(np.float32)
    f_close=d.groupby("g")["cl"].transform("sum").to_numpy().astype(np.float32)
    return np.column_stack([votes,f_nr,f_nts,f_ntse,f_np,f_jw,f_ar,f_atse,f_apar,f_njac,f_nfe,f_idf,f_ajac,
        f_afe,f_ale,f_hn,f_num,f_post,f_cmp,f_len,f_gap,f_close,np.full(n,flag,np.float32),f_ntf,f_atf,f_sd,
        f_ppost,f_nl,f_al,f_ntc,f_atc,f_ncj,f_acj,f_hnd,f_nqg,f_aqg,f_nlev,f_alev,f_nsh,f_ash,f_cm,f_nft,
        f_aft,f_nlt,f_alt,f_ndr,f_adr,f_cl,f_ct]).astype(np.float32)

# --- sibling + competition features from a stage-1 probability vector ---
def sibling_feats(grp, cnames, p1, thr=CONF_THR):
    by=defaultdict(list)
    for i,g in enumerate(grp): by[int(g)].append(i)
    sib=np.zeros(len(grp),np.float32); prc=np.zeros(len(grp),np.float32)
    for g,idxs in by.items():
        mx=max(p1[i] for i in idxs)
        conf=[(i,cnames[i]) for i in idxs if p1[i]>=thr]
        for i in idxs:
            prc[i]=p1[i]-mx
            oth=[cn for j,cn in conf if j!=i]
            if oth: sib[i]=max(fuzz.ratio(cnames[i],o) for o in oth)
    return sib.astype(np.float32), prc.astype(np.float32)

def f05(pred,true):
    if not true: return 1.0 if not pred else 0.0
    if not pred: return 0.0
    tp=len(pred&true)
    if tp==0: return 0.0
    P,R=tp/len(pred),tp/len(true); return 1.25*P*R/(0.25*P+R)
def best_set(pairs):
    if not pairs: return []
    pairs=sorted(pairs,key=lambda z:z[0],reverse=True); ps=[p for p,_ in pairs]; T=sum(ps)
    best=1.0
    for p in ps: best*=(1.0-p)
    bk,cum=0,0.0
    for k in range(1,len(ps)+1):
        cum+=ps[k-1]; P=cum/k; R=cum/T if T>0 else 0.0
        f=1.25*P*R/(0.25*P+R) if(0.25*P+R)>0 else 0.0
        if f>best: best,bk=f,k
    return [pairs[i] for i in range(bk)]
print("setup ok | USE_ADDR:",USE_ADDR,"| BLK_TOPK:",BLK_TOPK)


# ============================== CELL 3 — load train + pools + tfidf =========
t0=time.time()
gt=pd.read_csv(f"{TR}/train_ground_truth.tsv",sep="\t"); gt["matched_entity_ids"]=gt["matched_entity_ids"].fillna("")
GT={r.source1_entity_id:set(x for x in str(r.matched_entity_ids).split(",") if x) for r in gt.itertuples()}
samp=clean_df(pd.read_csv(f"{TR}/train_source1.tsv",sep="\t").sample(min(SAMPLE_S1,999999),random_state=RS).reset_index(drop=True))
need=set().union(*[GT.get(i,set()) for i in samp["entity_id"]])
def build_pool(path,need_ids,neg):
    src=pd.read_csv(path,sep="\t"); must=src[src["entity_id"].isin(need_ids)]
    rest=src[~src["entity_id"].isin(need_ids)].sample(min(neg,len(src)),random_state=RS)
    return clean_df(pd.concat([must,rest]).drop_duplicates("entity_id").reset_index(drop=True))
poolS2=build_pool(f"{TR}/train_source2.tsv",{x for x in need if x.startswith('S2')},NEG_POOL)
poolS3=build_pool(f"{TR}/train_source3.tsv",{x for x in need if x.startswith('S3')},NEG_POOL)
qA=arrays(samp); s2A=arrays(poolS2); s3A=arrays(poolS3)
idf2=build_idf(s2A["name"]); idf3=build_idf(s3A["name"])
nvec=TfidfVectorizer(analyzer='word',ngram_range=(1,2)); avec=TfidfVectorizer(analyzer='word',ngram_range=(1,2))
nvec.fit(np.concatenate([qA["name"],s2A["name"],s3A["name"]])); avec.fit(np.concatenate([qA["addr"],s2A["addr"],s3A["addr"]]))
qnt=nvec.transform(qA["name"]); s2nt=nvec.transform(s2A["name"]); s3nt=nvec.transform(s3A["name"])
qat=avec.transform(qA["addr"]); s2at=avec.transform(s2A["addr"]); s3at=avec.transform(s3A["addr"])
print(f"sample {len(samp):,} | poolS2 {len(poolS2):,} | poolS3 {len(poolS3):,} | {time.time()-t0:.0f}s")


# ============================== CELL 4 — block + recall ====================
t0=time.time()
def block_sc(qA,cA,topk,cap):
    QR,CP,VT=[],[],[]
    for ctry in np.unique(qA["country"]):
        qi=np.where(qA["country"]==ctry)[0]; ci=np.where(cA["country"]==ctry)[0]
        if len(qi)==0 or len(ci)==0: continue
        sub={k:(v[ci] if isinstance(v,np.ndarray) else [v[j] for j in ci]) for k,v in cA.items()}
        idx=build_index(sub)
        qr,cp,vt=block_pairs(qA["name"][qi],qA["compact"][qi],[qA["nums"][j] for j in qi],
            [qA["postal"][j] for j in qi],qA["addr"][qi],qA["acompact"][qi],qA["sdx"][qi],idx,topk,cap)
        if len(qr): QR.append(qi[qr]); CP.append(ci[cp]); VT.append(vt)
        del idx,sub; gc.collect()
    if not QR: return (np.array([],np.int64),)*2+(np.array([],np.float32),)
    return np.concatenate(QR),np.concatenate(CP),np.concatenate(VT)
q2,c2,v2=block_sc(qA,s2A,BLK_TOPK,BLK_CAP)
q3,c3,v3=block_sc(qA,s3A,BLK_TOPK,BLK_CAP)
got=defaultdict(set)
for qi,ci in zip(q2,c2): got[int(qi)].add(s2A["id"][int(ci)])
for qi,ci in zip(q3,c3): got[int(qi)].add(s3A["id"][int(ci)])
rec=tot=0
for k,i in enumerate(qA["id"]):
    truth=GT.get(i,set())&need
    if not truth: continue
    rec+=len(truth&got[k]); tot+=len(truth)
print(f"blocking recall: {rec/max(1,tot)*100:.2f}% | avg cand {(len(q2)+len(q3))/max(1,len(qA['id'])):.1f} | {time.time()-t0:.0f}s")
print(">>> If friend found higher recall, set BLK_TOPK/USE_ADDR to hers and RE-RUN Cell 4 before continuing.")


# ============================== CELL 5 — stage-1 features ==================
t0=time.time()
X2=build_feats(qA,s2A,q2,c2,v2,0.,idf2,qnt,s2nt,qat,s2at)
X3=build_feats(qA,s3A,q3,c3,v3,1.,idf3,qnt,s3nt,qat,s3at)
y2=np.array([1 if s2A["id"][int(c2[i])] in GT.get(qA["id"][int(q2[i])],set()) else 0 for i in range(len(q2))])
y3=np.array([1 if s3A["id"][int(c3[i])] in GT.get(qA["id"][int(q3[i])],set()) else 0 for i in range(len(q3))])
X=np.vstack([X2,X3]); y=np.concatenate([y2,y3]); grp=np.concatenate([q2,q3]).astype(int)
cand_ids=np.concatenate([s2A["id"][c2],s3A["id"][c3]])
cand_names=np.concatenate([s2A["name"][c2],s3A["name"][c3]])
print(f"pairs {X.shape} | pos {y.mean()*100:.2f}% | {time.time()-t0:.0f}s")


# ============================== CELL 6 — STAGE 1 (OOF probs) ================
def train_oof(Xm,y,grp):
    gkf=GroupKFold(5); ol=np.zeros(len(y)); ox=np.zeros(len(y)); ML=[]; MX=[]
    lp=dict(objective="binary",metric="auc",boosting_type="gbdt",device="gpu",learning_rate=0.05,
            num_leaves=63,feature_fraction=0.8,seed=RS,verbose=-1)
    xp=dict(objective="binary:logistic",eval_metric="auc",tree_method="hist",device="cuda",
            learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,random_state=RS)
    for tr,va in gkf.split(Xm,y,grp):
        m=lgb.train(lp,lgb.Dataset(Xm[tr],label=y[tr]),num_boost_round=900,
                    valid_sets=[lgb.Dataset(Xm[va],label=y[va])],callbacks=[lgb.early_stopping(50,verbose=False)])
        ol[va]=m.predict(Xm[va]); ML.append(m)
        dm=xgb.DMatrix(Xm[tr],label=y[tr]); dv=xgb.DMatrix(Xm[va],label=y[va])
        mx=xgb.train(xp,dm,num_boost_round=900,evals=[(dv,"v")],early_stopping_rounds=50,verbose_eval=False)
        ox[va]=mx.predict(dv); MX.append(mx)
    return 0.5*ol+0.5*ox, ML, MX
t0=time.time()
p1_oof, LGB1, XGB1 = train_oof(X,y,grp)
print(f"stage-1 done | {time.time()-t0:.0f}s")


# ============================== CELL 7 — SIBLING features ==================
t0=time.time()
sib,prc=sibling_feats(grp,cand_names,p1_oof)
X_s2=np.column_stack([X,sib,prc]).astype(np.float32)
print(f"sibling feats built | sib>0 frac {np.mean(sib>0):.3f} | {time.time()-t0:.0f}s")


# ============================== CELL 8 — STAGE 2 + honest F0.5 GATE =========
t0=time.time()
p2_oof, LGB2, XGB2 = train_oof(X_s2,y,grp)
# honest macro-F0.5 with expected-F0.5 decision, on OOF
by=defaultdict(list)
for g,p,c in zip(grp,p2_oof,cand_ids): by[int(g)].append((float(p),c))
ids=np.unique(grp)
score=sum(f05({c for _,c in best_set(by.get(int(k),[]))}, GT.get(qA["id"][int(k)],set())&need) for k in ids)/len(ids)
print(f"\n*** STAGE-2 HONEST macro-F0.5 = {score:.4f}  (friend's base was 0.9184) ***")
print("If this is clearly >0.9184, run Cell 9 (inference). If not, raise BLK_TOPK / CONF_THR and re-run 6-8.")
print(f"stage-2 done | {time.time()-t0:.0f}s")


# ============================== CELL 9 — TEST inference (two-stage+sibling) =
import warnings; warnings.filterwarnings("ignore")
uc=["entity_id","business_name","business_address","country"]
te1=pd.read_csv(f"{TE}/test_source1.tsv",sep="\t",usecols=uc)
te2=pd.read_csv(f"{TE}/test_source2.tsv",sep="\t",usecols=uc)
te3=pd.read_csv(f"{TE}/test_source3.tsv",sep="\t",usecols=uc)
for d in (te1,te2,te3): d["ck"]=d["country"].fillna("").astype(str).str.lower().str.strip()
def pred(Xm,ML,MX):
    pl=np.mean([m.predict(Xm) for m in ML],axis=0)
    dm=xgb.DMatrix(Xm); px=np.mean([m.predict(dm) for m in MX],axis=0)
    return 0.5*pl+0.5*px
fm=open(f"{OUT}/matching_results.tsv","w"); fm.write("source1_entity_id\tmatched_entity_ids\n")
fc=open(f"{OUT}/candidate_pairs.tsv","w"); fc.write("source1_entity_id\tcandidate_entity_ids\n")
t0=time.time(); mcounts=[]
for ctry in sorted(te1["ck"].unique()):
    qdf=clean_df(te1[te1["ck"]==ctry].drop(columns="ck").reset_index(drop=True)); qAc=arrays(qdf); nq=len(qAc["id"])
    qn_t=nvec.transform(qAc["name"]); qa_t=avec.transform(qAc["addr"])
    src=[]
    for raw,flag in [(te2,0.),(te3,1.)]:
        cdf=clean_df(raw[raw["ck"]==ctry].drop(columns="ck").reset_index(drop=True))
        if len(cdf)==0: continue
        cA=arrays(cdf); src.append((cA,flag,build_index(cA),build_idf(cA["name"]),
                                    nvec.transform(cA["name"]),avec.transform(cA["addr"])))
        del cdf; gc.collect()
    assigned=[[] for _ in range(nq)]; cand_all=[[] for _ in range(nq)]
    for s in range(0,nq,CHUNK):
        e=min(s+CHUNK,nq)
        QG=[]; CIDc=[]; CNMc=[]; Frows=[]
        for cA,flag,idx,idf,cnt,cat in src:
            qr,cp,vt=block_pairs(qAc["name"][s:e],qAc["compact"][s:e],[qAc["nums"][j] for j in range(s,e)],
                [qAc["postal"][j] for j in range(s,e)],qAc["addr"][s:e],qAc["acompact"][s:e],qAc["sdx"][s:e],idx,BLK_TOPK,BLK_CAP)
            if not len(qr): continue
            qg=qr+s
            F=build_feats(qAc,cA,qg,cp,vt,flag,idf,qn_t,cnt,qa_t,cat)
            Frows.append(F); QG.append(qg); CIDc.append(cA["id"][cp]); CNMc.append(cA["name"][cp])
        if not Frows: continue
        F1=np.vstack(Frows); QG=np.concatenate(QG); CID=np.concatenate(CIDc); CNM=np.concatenate(CNMc)
        p1=pred(F1,LGB1,XGB1)
        sibc,prcc=sibling_feats(QG,CNM,p1)
        F2=np.column_stack([F1,sibc,prcc]).astype(np.float32)
        p2=pred(F2,LGB2,XGB2)
        byq=defaultdict(list)
        for gg,pp,cc in zip(QG,p2,CID):
            byq[int(gg)].append((float(pp),cc)); cand_all[int(gg)].append(cc)
        for loc,lst in byq.items():
            for pp,cc in best_set(lst): assigned[loc].append((pp,cc))
        print(f"  [{ctry}] {s//CHUNK+1} | {time.time()-t0:.0f}s",flush=True)
    # global 1-to-1 within country + write
    used=set()
    for loc in range(nq):
        keep=[]
        for pp,cc in sorted(assigned[loc],reverse=True):
            if cc in used: continue
            used.add(cc); keep.append(cc)
        fm.write(f"{qAc['id'][loc]}\t{','.join(keep)}\n")
        fc.write(f"{qAc['id'][loc]}\t{','.join(map(str,cand_all[loc]))}\n")
        mcounts.append(len(keep))
    fm.flush(); fc.flush()
    print(f"=== {ctry} DONE | {time.time()-t0:.0f}s ===",flush=True)
    del qAc,qdf,src,assigned,cand_all; gc.collect()
fm.close(); fc.close()
mc=np.array(mcounts)
print(f"ALL DONE {time.time()-t0:.0f}s | singleton {(mc==0).mean():.3f} | avg {mc.mean():.2f}")


# ============================== CELL 10 — validate =========================
import glob as _g, subprocess
v=_g.glob(f"{BASE}/**/validate_submission.py",recursive=True)
if v:
    print(subprocess.run(["python3",v[0],"--matching",f"{OUT}/matching_results.tsv",
        "--candidate",f"{OUT}/candidate_pairs.tsv","--test-dir",TE],capture_output=True,text=True).stdout)
print("files:",os.listdir(OUT))
