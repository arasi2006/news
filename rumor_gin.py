import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_USER, D_SIM, D_TEXT = 8, 110, 768
SEED = 0


def make_event(text, user, sim, edges, label):
    return dict(text=torch.tensor(text, dtype=torch.float),
                user=torch.tensor(user, dtype=torch.float),
                sim=torch.tensor(sim, dtype=torch.float),
                edges=torch.tensor(edges, dtype=torch.long).t().contiguous(),
                y=label)


def load_real_data(path="events.pkl", max_hours=None):
    import pickle
    with open(path, "rb") as f:
        raw = pickle.load(f)
    events = []
    for r in raw:
        keep = np.arange(len(r["t"])) if max_hours is None else np.where(r["t"] <= max_hours)[0]
        if 0 not in keep:
            keep = np.append(keep, 0)
        remap = {int(o): i for i, o in enumerate(keep)}
        edges = [(remap[a], remap[b]) for a, b in r["edges"] if a in remap and b in remap] or [(0, 0)]
        events.append(make_event(r["text"][keep], r["user"][keep], r["sim"][keep], edges, r["y"]))
    return events


def collate(batch):
    xs_t, xs_u, xs_s, edges, ys, bvec, off = [], [], [], [], [], [], 0
    for g, ev in enumerate(batch):
        n = ev["text"].shape[0]
        xs_t.append(ev["text"]); xs_u.append(ev["user"]); xs_s.append(ev["sim"])
        edges.append(ev["edges"] + off)
        bvec.append(torch.full((n,), g, dtype=torch.long))
        ys.append(ev["y"]); off += n
    return (torch.cat(xs_t).to(DEVICE), torch.cat(xs_u).to(DEVICE), torch.cat(xs_s).to(DEVICE),
            torch.cat(edges, 1).to(DEVICE), torch.cat(bvec).to(DEVICE),
            torch.tensor(ys).to(DEVICE), len(batch))


class FusionBranch(nn.Module):
    def __init__(self, d_in, d_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ReLU(), nn.Linear(d_hid, d_hid))
        self.bn = nn.BatchNorm1d(d_hid)

    def forward(self, x):
        return self.bn(self.net(x))


class ResMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.bn = nn.BatchNorm1d(d)

    def forward(self, x):
        return self.bn(F.relu(x + self.f(x)))


class GINLayer(nn.Module):
    def __init__(self, d, eps=0.0):
        super().__init__()
        self.eps, self.nn = eps, ResMLP(d)

    def forward(self, h, edge_index):
        src = torch.cat([edge_index[0], edge_index[1]])
        dst = torch.cat([edge_index[1], edge_index[0]])
        agg = torch.zeros_like(h).index_add_(0, dst, h[src])
        return self.nn((1 + self.eps) * h + agg)


def graph_max_pool(h, batch, B):
    out = torch.full((B, h.shape[1]), -1e9, device=h.device)
    idx = batch.unsqueeze(1).expand_as(h)
    return out.scatter_reduce(0, idx, h, reduce="amax", include_self=True)


class RumorGIN(nn.Module):
    def __init__(self, hid=(16, 32, 256), layers=4, n_cls=2, drop=0.3):
        super().__init__()
        self.fu = FusionBranch(D_USER, hid[0])
        self.fs = FusionBranch(D_SIM, hid[1])
        self.ft = FusionBranch(D_TEXT, hid[2])
        d = sum(hid)
        self.gins = nn.ModuleList([GINLayer(d) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(d * (layers + 1), 128), nn.ReLU(),
                                  nn.Dropout(drop), nn.Linear(128, n_cls))

    def forward(self, xt, xu, xs, edge_index, batch, B):
        h = torch.cat([self.fu(xu), self.fs(xs), self.ft(xt)], dim=1)
        outs = [h]
        for layer in self.gins:
            h = layer(h, edge_index)
            outs.append(h)
        hg = graph_max_pool(torch.cat(outs, 1), batch, B)
        return self.head(hg)


def run_epoch(model, data, bs, opt=None):
    train = opt is not None
    model.train(train)
    if train:
        random.shuffle(data)
    ys, ps, tot = [], [], 0.0
    for i in range(0, len(data), bs):
        chunk = data[i:i + bs]
        if train and len(chunk) < 2:
            continue
        xt, xu, xs, ei, b, y, B = collate(chunk)
        with torch.set_grad_enabled(train):
            logits = model(xt, xu, xs, ei, b, B)
            loss = F.cross_entropy(logits, y)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * B
        ys += y.tolist(); ps += logits.argmax(1).tolist()
    p, r, f, _ = precision_recall_fscore_support(ys, ps, average="binary", zero_division=0)
    return tot / len(data), accuracy_score(ys, ps), p, r, f


def main(epochs=30, bs=32, lr=1e-3, layers=4, data_path=None, max_hours=None, save_path=None):
    torch.manual_seed(SEED); random.seed(SEED)
    data = load_real_data(data_path, max_hours)
    random.shuffle(data)
    n = len(data)
    tr, va, te = data[:int(.8 * n)], data[int(.8 * n):int(.85 * n)], data[int(.85 * n):]
    model = RumorGIN(layers=layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state = 0, None
    for ep in range(1, epochs + 1):
        l, a, *_ = run_epoch(model, tr, bs, opt)
        _, va_acc, *_ = run_epoch(model, va, bs)
        if va_acc >= best:
            best, best_state = va_acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep{ep:02d} loss {l:.3f} train_acc {a:.3f} val_acc {va_acc:.3f}")
    model.load_state_dict(best_state)
    if save_path:
        torch.save(best_state, save_path)
    _, acc, p, r, f = run_epoch(model, te, bs)
    print(f"TEST  acc {acc:.3f}  P {p:.3f}  R {r:.3f}  F1 {f:.3f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--save", default=None)
    a = ap.parse_args()
    main(epochs=a.epochs, layers=a.layers, data_path=a.data, max_hours=a.hours, save_path=a.save)
