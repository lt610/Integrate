
import torch as th
import dgl.function as fn


def compute_D_and_e(g, lam, propagation):
    if propagation == "lt":
        degs = g.in_degrees().float() - 1.0
        nor_A = th.pow(lam * degs + 1, -0.5).unsqueeze(1)
        nor_X = th.pow(lam * degs + 1, - 1).unsqueeze(1)

        g.ndata["nor_A"] = nor_A
        g.ndata['nor_X'] = nor_X
        # 这里是后续扩展到大图训练的时候用到
        # g.apply_edges(fn.u_mul_v('bef_A', 'aft_A', 'e'))
        # g.update_all(fn.copy_e('e', 't'), fn.sum('t', 'D_tilde'))
        # g.edata['e'] = g.edata['e'].view(g.num_edges(), 1)
    elif propagation == "yyy":
        g.ndata['D_in'] = 1 / g.in_degrees().float().sqrt().unsqueeze(1)
        g.ndata['D_out'] = 1 / g.out_degrees().float().sqrt().unsqueeze(1)
        # 这里是后续扩展到大图训练的时候用到
        # g.apply_edges(fn.u_mul_v('D_in', 'D_out', 'e'))
        # g.update_all(fn.copy_e('e', 't'), fn.sum('t', 'D_tilde'))
        # g.edata['e'] = g.edata['e'].view(g.num_edges(), 1)
    elif propagation == "exact":
        adj = g.adjacency_matrix().to(g.device)
        adj = (- th.eye(adj.shape[0]).to(g.device)).add(adj)
        degs = g.in_degrees().float().to(g.device)
        bef_A = degs.sqrt().diag()
        aft_A = (1 / degs.sqrt()).diag()
        sub_1 = (lam * degs + 1 - lam).diag()
        sub_2 = lam * bef_A.mm(adj.mm(aft_A))
        sub = sub_1 - sub_2
        g.ndata["exact"] = sub.inverse()
