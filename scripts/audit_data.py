"""Data audit for the three datasets and tetramer tables. Writes outputs/audit/{audit.json,*.png}."""
import numpy as np, pandas as pd, collections, itertools, json
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import skew, pearsonr
from pathlib import Path
R=str(Path(__file__).resolve().parents[1])+'/'; O=R+'outputs/audit/'; Path(O).mkdir(parents=True, exist_ok=True)
rc=lambda s:''.join({'A':'T','T':'A','C':'G','G':'C'}.get(c,'N') for c in s[::-1])
canon=lambda s:min(s,rc(s))
def seqstats(name, seqs, y):
    seqs=list(seqs); y=np.asarray(y,float)
    L=collections.Counter(map(len,seqs)); bad=sum(bool(set(s)-set('ACGT')) for s in seqs)
    df=pd.DataFrame({'s':seqs,'y':y}); df['c']=df.s.map(canon)
    exact=df.s.duplicated().sum(); g=df.groupby('c').y
    multi=g.size(); m=multi[multi>1].index
    spread=(g.max()-g.min())[m]
    pal=sum(s==rc(s) for s in seqs)
    r=dict(dataset=name,n=len(seqs),lengths=dict(L),invalid=bad,exact_dups=int(exact),
      canon_groups=int(len(multi)),rc_dup_groups=int(len(m)),rc_dup_label_spread_median=float(spread.median()) if len(m) else None,
      rc_dup_spread_gt_0p1=int((spread>0.1).sum()) if len(m) else 0, palindromes=pal,
      y_min=float(y.min()),y_max=float(y.max()),y_median=float(np.median(y)),skew=float(skew(y)),
      frac_y_gt_med_plus_5mad=float(np.mean(y>np.median(y)+5*np.median(abs(y-np.median(y))))), nan=int(np.isnan(y).sum()))
    fig,ax=plt.subplots(1,2,figsize=(9,3));ax[0].hist(y,100);ax[0].set_title(name+' labels');ax[1].hist(np.log1p(y-y.min()),100);ax[1].set_title('log1p(y-min)')
    fig.tight_layout();fig.savefig(O+name+'_labels.png');plt.close()
    return r
res=[]
up=pd.read_csv(R+'data/upbm/Gata4/Gata4_training.txt',sep='\t')
r=seqstats('upbm_training',up.ID_REF,up.VALUE); r['weights']=up.WEIGHT.value_counts().to_dict()
r['mean_y_by_weight']=up.groupby('WEIGHT').VALUE.mean().to_dict(); res.append(r)
al=pd.read_csv(R+'data/upbm/Gata4/Gata4_alignment_weighted.txt',sep='\t'); res.append(seqstats('upbm_alignment_raw',al.ID_REF,al.VALUE))
st=pd.read_csv(R+'data/selex/Gata4/SELEX_test.txt',sep='\t'); r=seqstats('selex_SELEX_test_file',st.ID_REF,st.VALUE)
r['identical_seqs_to_upbm_training']=len(set(st.ID_REF)&set(up.ID_REF)); res.append(r)
d=pd.read_csv(R+'data/selex/Gata4/4.txt',sep='\t'); res.append(seqstats('selex_cycle4_all',d.Kmer,d.Affinity))
b=sorted(d.Probability)[int(len(d)*0.9)]; top=d[d.Probability>b]
r=seqstats('selex_top10pct',top.Kmer,top.Affinity); r['corr_prob_affinity_all']=float(pearsonr(d.Probability,d.Affinity)[0])
r['count_min_top']=int(top.ObservedCount.min()); r['frac_count_le5_all']=float((d.ObservedCount<=5).mean()); r['se_over_aff_median_top']=float((top.SE/top.Affinity).median()); res.append(r)
plt.figure();plt.scatter(np.log10(d.ObservedCount),d.Affinity,s=1);plt.xlabel('log10 count');plt.ylabel('Affinity');plt.savefig(O+'selex_count_vs_aff.png');plt.close()
g=pd.read_csv(R+'data/gcpbm/cbf1/cbf1_100.txt',sep='\t'); g=g.dropna(subset=['Sequence','Intensity'])
r=seqstats('gcpbm_raw60',g.Sequence,g.Intensity); r['id_prefix_counts']=g.ID.str.split('_').str[0].value_counts().head(8).to_dict()
s30=g.Sequence.str[3:33]; r2=seqstats('gcpbm_trimmed30',s30,g.Intensity)
r2['center_6mer_top']=s30.str[12:18].value_counts().head(5).to_dict()
r2['const_tail_check']=g.Sequence.str[33:].value_counts().head(2).to_dict()
fm=np.array(pd.read_csv(R+'data/gcpbm/cbf1/cbf1_freq_matrix_6.txt',sep=r'\s+')); r2['freq_matrix_rows']=fm.tolist()
tr={'A':0,'C':1,'G':2,'T':3}; pos=list(range(0,7))+list(range(18,25))
def flank(s,rcmode=False):
    sc=[sum(fm[j][tr[s[i+j]]] for j in range(6)) for i in pos]
    if rcmode: sc+= [sum(fm[j][tr[rc(s)[i+j]]] for j in range(6)) for i in pos]
    return max(sc)
mx=s30.map(flank); r2['frac_discarded_2.5']=float((mx>2.5).mean())
r2['flank_positions_scanned']=pos; r2['uncovered_positions_note']='windows start 0..6 (cover 0..11) and 18..24 (cover 18..29); RC strand not scanned'
r2['frac_discarded_with_rc']=float((s30.map(lambda s:flank(s,True))>2.5).mean())
r2['max_possible_score']=float(fm.max(1).sum()); r2['random_seq_frac_window_gt2.5']=None
rng=np.random.default_rng(0); rs=[''.join(rng.choice(list('ACGT'),30)) for _ in range(3000)]
r2['random_seq_frac_discard']=float(np.mean([flank(s)>2.5 for s in rs]))
y=(g.Intensity-g.Intensity.min())/(g.Intensity.max()-g.Intensity.min())
r2['mean_y_kept']=float(y[mx<=2.5].mean()); r2['mean_y_discarded']=float(y[mx>2.5].mean()); res.append(r2)
plt.figure();plt.hist(mx,60);plt.axvline(2.5,c='r');plt.title('gcPBM max flank PWM score');plt.savefig(O+'gcpbm_flank_scores.png');plt.close()
# tetramer tables
tabs={}
for f in ['avg_tetramer.dat','fce_tetramer.dat','electrostatic.txt','mgw_rohs.txt']:
    rows=[l.split() for l in open(R+'data/input/'+f) if 'SHIFT' not in l]
    keys=[x[0] for x in rows]; vals=np.array([[float(v) for v in x[1:]] for x in rows])
    all4=[''.join(p) for p in itertools.product('ACGT',repeat=4)]
    dk=collections.Counter(keys); dupk=[k for k,c in dk.items() if c>1]
    dupconf=[k for k in dupk if len({tuple(x[1:]) for x in rows if x[0]==k})>1]
    present=set(keys); d={k:np.array([float(v) for v in x[1:]]) for x,k in zip(rows,keys)}
    rcdiff=[np.abs(d[k]-d[rc(k)]).max() for k in present if rc(k) in present and k<rc(k)]
    tabs[f]=dict(rows=len(rows),unique=len(present),missing=sorted(set(all4)-present)[:20],n_missing=len(set(all4)-present),
       dup_keys=len(dupk),dup_conflicting=len(dupconf),ncols=vals.shape[1],nan=int(np.isnan(vals).sum()),
       col_mean=np.round(vals.mean(0),3).tolist()[:8],col_std=np.round(vals.std(0),3).tolist()[:8],
       rc_maxabsdiff_median=float(np.median(rcdiff)) if rcdiff else None, n_canonical_present=len({canon(k) for k in present}))
json.dump(dict(datasets=res,tables=tabs),open(O+'audit.json','w'),indent=1,default=str)
print(json.dumps(dict(datasets=res,tables=tabs),indent=1,default=str))
