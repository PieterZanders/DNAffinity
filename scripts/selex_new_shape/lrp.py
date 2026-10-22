"""Layer-wise Relevance Propagation (manual, exact implementation) for DFN and CNN1D trained with the final protocol.
Rules: BatchNorm folded into preceding Conv1d (exact in eval mode); Dropout = identity (eval); ReLU = identity for relevance;
input layer: LRP-epsilon (inputs are z-scored, can be negative -> gamma not applicable); hidden Conv1d: LRP-gamma (gamma=0.25);
Linear layers: LRP-epsilon (eps=1e-6*... see EPS); residual addition and global average pooling: epsilon-rule proportional split
(z_j / sum z). Bias relevance is absorbed (reported via conservation ratio sum(R_in)/f(x)).
Sanity check: with all rules = LRP-0 the relevance equals gradient x input (piecewise-linear net), verified numerically.
Usage: lrp.py OUTDIR MODEL SHAPE(0/1)"""
import sys, glob, json, numpy as np, torch, torch.nn as nn
from lib import *
from dnaffinity.models.cnn1d import CNN1D
from dnaffinity.models.dfn import RegressionCV, FeedForward
torch.set_num_threads(4)
od, model, shape = sys.argv[1], sys.argv[2], bool(int(sys.argv[3]))
EPS, GAM = 1e-2, 0.25

def fold_bn(conv, bn):
    w, b = conv.weight.detach(), (conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels))
    s = bn.weight.detach() / torch.sqrt(bn.running_var + bn.eps)
    return w * s[:, None, None], (b - bn.running_mean) * s + bn.bias.detach()

def stab(z, e): return z + e * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))

def lrp_lin(a, R, f, rho_w, rho_b, e):
    """generic: z = f(a, rho(W), rho(b)); R_in = a * grad(z . (R/z))"""
    a = a.detach().requires_grad_(True); z = f(a, rho_w, rho_b); s = (R / stab(z, e)).detach()
    (z * s).sum().backward(); return (a * a.grad).detach()

def gam(w, g): return w + g * w.clamp(min=0)

def cnn_lrp(net, x, rule):  # rule: 'comp' or 'zero'
    e = 1e-9 if rule == 'zero' else EPS; g = 0.0 if rule == 'zero' else GAM
    bb = net.backbone; convs = [fold_bn(bb.stem[0], bb.stem[1])]
    for blk in bb.blocks: convs += [fold_bn(blk.conv1, blk.bn1), fold_bn(blk.conv2, blk.bn2)]
    cv = lambda a, w, b: torch.nn.functional.conv1d(a, w, b, padding=1)
    # forward, storing activations
    with torch.no_grad():
        h0 = x.transpose(1, 2); acts = []
        h = torch.relu(cv(h0, *convs[0])); acts.append(h0)
        H = [h]; inner = []
        for i in range(len(bb.blocks)):
            w1, w2 = convs[1 + 2 * i], convs[2 + 2 * i]
            a1 = torch.relu(cv(h, *w1)); o = cv(a1, *w2); inner.append((h, a1, o)); h = torch.relu(o + h); H.append(h)
        pooled = h.mean(2); L1, L2 = net.head.net[0], net.head.net[3]
        q = torch.relu(L1(pooled)); out = L2(q)
    lin = lambda a, w, b: torch.nn.functional.linear(a, w, b)
    R = out.clone()
    R = lrp_lin(q, R, lin, L2.weight.detach(), L2.bias.detach(), e)
    R = lrp_lin(pooled, R, lin, L1.weight.detach(), L1.bias.detach(), e)
    # avg pool: R_j(l) = (h_l/L) / pooled * R
    R = (h / h.shape[2]) * (R / stab(pooled, e))[:, :, None]
    for i in reversed(range(len(bb.blocks))):
        hin, a1, o = inner[i]; w1, w2 = convs[1 + 2 * i], convs[2 + 2 * i]
        zs = stab(o + hin, e); Ro = o * R / zs; Rh = hin * R / zs   # residual sum: proportional split
        Ra = lrp_lin(a1, Ro, cv, gam(w2[0], g), gam(w2[1], g), e)
        Rh = Rh + lrp_lin(hin, Ra, cv, gam(w1[0], g), gam(w1[1], g), e)
        R = Rh
    R = lrp_lin(acts[0], R, cv, convs[0][0], convs[0][1], e)  # input layer: epsilon
    return R.transpose(1, 2), out.ravel()

def dfn_lrp(net, x, rule):
    e = 1e-9 if rule == 'zero' else EPS
    mods = [m for m in net.nn.network if isinstance(m, nn.Linear)]; A = [x]
    with torch.no_grad():
        a = x
        for i, m in enumerate(mods):
            a = m(a); a = torch.relu(a) if i < len(mods) - 1 else a; A.append(a)
    R = A[-1].clone(); lin = lambda a, w, b: torch.nn.functional.linear(a, w, b)
    for i in reversed(range(len(mods))): R = lrp_lin(A[i], R, lin, mods[i].weight.detach(), mods[i].bias.detach(), e)
    return R, A[-1].ravel()

def gxi(net, x):
    x = x.clone().requires_grad_(True); net(x).sum().backward(); return (x * x.grad).detach()

c = json.load(open(OUT / od / 'config.json')); d = load_labels(10, 'perstrand')
full = pd.read_csv(ROOT / 'data/selex/Gata4/4.txt', sep='\t'); allg = full.Kmer.map(canonical).values; tg = set(allg[split(allg)])
T = np.where(d.group.isin(tg).values)[0]; seqs = list(d.seq.values[T]); y = np.log(d.aff.values[T])
X = (windowed if model == 'cnn1d' else flat_feats)(seqs, tuple(int(k) for k in c['ks']), tuple(c['props'].split('+')), bool(c['onehot']), bool(c['shape']))
RS, RG, RZ, OUTS, CONS = [], [], [], [], []
for f in sorted(glob.glob(str(OUT / od / model / 'fold*_seed*.pt'))):
    ck = torch.load(f, weights_only=False); Z = torch.tensor((X - ck['mu']) / ck['sd'], dtype=torch.float32)
    if model == 'cnn1d': net = CNN1D(Z.shape[2], 64, 2, 3, 'avg', 0.2, normalization=False); fn = cnn_lrp
    else: net = RegressionCV(FeedForward([Z.shape[1], 256, 128, 64, 1], dropout=0.2), normalization=False); fn = dfn_lrp
    net.load_state_dict(ck['state']); net.eval()
    with torch.no_grad(): ref = net(Z).ravel()
    R, o = fn(net, Z, 'comp'); R0, _ = fn(net, Z, 'zero'); G = gxi(net, Z)
    assert torch.allclose(o, ref, atol=1e-4), 'LRP forward mismatch (BN folding)'
    RS.append(R.numpy()); RZ.append(R0.numpy()); RG.append(G.numpy()); OUTS.append(o.numpy())
    print(f, 'conservation sum(R)/f:', np.median(R.reshape(len(R), -1).sum(1).numpy() / o.numpy()), flush=True)
np.savez_compressed(OUT / od / f'lrp_{model}.npz', R=np.array(RS), R0=np.array(RZ), G=np.array(RG), out=np.array(OUTS), X=X, seqs=np.array(seqs), y=y)
