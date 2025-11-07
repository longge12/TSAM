import sys
import os
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from dataset import VTKG
from TSAM import TSAM
from tqdm import tqdm
from utils import calculate_rank, metrics
from losses.alignment_losses_ext import compute_front_loss
from modules.fusion import GatedFusion
from modules.memory_bank import MemoryBank

import numpy as np
import argparse
import torch
import torch.nn as nn
import time
import math
import logging

from merge_tokens import get_entity_visual_tokens, get_entity_textual_tokens


os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

torch.manual_seed(2024)
np.random.seed(2024)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_format)
logger.addHandler(stream_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="MKG-W", type=str)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--dim', default=200, type=int)
    parser.add_argument('--num_epoch', default=120, type=int)
    parser.add_argument('--valid_epoch', default=200, type=int, help='evaluate every N epochs')
    parser.add_argument('--log_epoch_interval', default=1, type=int, help='write train logs every N epochs (console always)')
    parser.add_argument('--exp', default='TSAM')
    parser.add_argument('--no_write', action='store_true')
    parser.add_argument('--num_layer_enc_ent', default=1, type=int)
    parser.add_argument('--num_layer_enc_rel', default=1, type=int)
    parser.add_argument('--num_layer_dec', default=2, type=int)
    parser.add_argument('--num_head', default=2, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--dropout', default=0.01, type=float)
    parser.add_argument('--emb_dropout', default=0.9, type=float)
    parser.add_argument('--vis_dropout', default=0.4, type=float)
    parser.add_argument('--txt_dropout', default=0.1, type=float)
    parser.add_argument('--smoothing', default=0.0, type=float)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--decay', default=0.0, type=float)
    parser.add_argument('--max_img_num', default=3, type=int)
    parser.add_argument('--cont', action='store_true')
    parser.add_argument('--step_size', default=50, type=int)
    parser.add_argument('--max_vis_token', default=8, type=int)
    parser.add_argument('--max_txt_token', default=8, type=int)
    parser.add_argument('--score_function', default="tucker", type=str)
    parser.add_argument('--mu', default=0, type=float)
    # dual-branch
    parser.add_argument('--use_dual_branch', action='store_true', default=True)
    parser.add_argument('--no_use_dual_branch', dest='use_dual_branch', action='store_false')
    parser.add_argument('--use_cross_attn', action='store_true', default=True)
    parser.add_argument('--no_use_cross_attn', dest='use_cross_attn', action='store_false')
    parser.add_argument('--curvature_init', default=0.1, type=float)
    parser.add_argument('--use_diversity_reg', action='store_true')
    parser.add_argument('--diversity_margin', default=0.5, type=float)
    # semantic-only FERF-lite reconstruction weight
    parser.add_argument('--lambda_rec_sem', default=0.05, type=float)
    parser.add_argument('--gate_t_target', default=0.55, type=float)
    parser.add_argument('--gate_v_target', default=0.50, type=float)
    # front loss weights
    parser.add_argument('--lambda_h', default=0.8, type=float)
    parser.add_argument('--lambda_r', default=0.1, type=float)
    parser.add_argument('--lambda_f', default=0.1, type=float)
    parser.add_argument('--tau', default=0.07, type=float)
    parser.add_argument('--angle_margin', default=0.1, type=float)
    parser.add_argument('--r_t_target', default=0.5, type=float)
    parser.add_argument('--r_v_target', default=1.0, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--num_neg_samples', default=16, type=int)

    args = parser.parse_args()

    # default knobs for new features
    if not hasattr(args, 'angle_loss_type'):
        args.angle_loss_type = 'hinge'
    if not hasattr(args, 'neg_hard_mining'):
        args.neg_hard_mining = True
    if not hasattr(args, 'neg_hard_k'):
        args.neg_hard_k = 12
    if not hasattr(args, 'angle_target_anneal'):
        args.angle_target_anneal = True
    if not hasattr(args, 'anneal_epochs'):
        args.anneal_epochs = 40
    if not hasattr(args, 'pos_margin_deg_start'):
        args.pos_margin_deg_start = 28.0
    if not hasattr(args, 'pos_margin_deg_end'):
        args.pos_margin_deg_end = 12.0
    if not hasattr(args, 'neg_margin_deg_start'):
        args.neg_margin_deg_start = 110.0
    if not hasattr(args, 'neg_margin_deg_end'):
        args.neg_margin_deg_end = 80.0
    if not hasattr(args, 'hyp_lr_factor'):
        args.hyp_lr_factor = 3.0
    # memory bank & distance branch
    if not hasattr(args, 'use_memory_bank'):
        args.use_memory_bank = True
    if not hasattr(args, 'memory_bank_size'):
        args.memory_bank_size = 65536
    if not hasattr(args, 'memory_cand_size'):
        args.memory_cand_size = 8192
    if not hasattr(args, 'neg_source'):
        args.neg_source = 'hybrid'
    if not hasattr(args, 'memory_ratio'):
        args.memory_ratio = 0.7
    if not hasattr(args, 'lambda_d'):
        args.lambda_d = 0.3
    if not hasattr(args, 'use_dist_hinge'):
        args.use_dist_hinge = True
    if not hasattr(args, 'dist_pos_margin'):
        args.dist_pos_margin = 0.6
    if not hasattr(args, 'dist_neg_margin'):
        args.dist_neg_margin = 1.8
    # force enable diversity regularization by default
    try:
        if not args.use_diversity_reg:
            args.use_diversity_reg = True
    except Exception:
        pass

    # device & log file
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    if not args.no_write:
        os.makedirs(f"./result/{args.exp}/{args.data}", exist_ok=True)
        os.makedirs(f"./ckpt/{args.exp}/{args.data}", exist_ok=True)
        os.makedirs(f"./logs/{args.exp}/{args.data}", exist_ok=True)
        file_handler = logging.FileHandler(f"./logs/{args.exp}/{args.data}/train.log")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    logger.info(os.getpid())
    logger.info(args)

    KG = VTKG(args.data, logger, max_vis_len=args.max_img_num)
    KG_Loader = torch.utils.data.DataLoader(KG, batch_size=args.batch_size, shuffle=True, num_workers=2)

    visual_token_index, visual_key_mask = get_entity_visual_tokens(dataset=args.data, max_num=args.max_vis_token)
    visual_token_index = visual_token_index.to(device)
    text_token_index, text_key_mask = get_entity_textual_tokens(dataset=args.data, max_num=args.max_txt_token)
    text_token_index = text_token_index.to(device)

    model = TSAM(
        num_ent=KG.num_ent,
        num_rel=KG.num_rel,
        ent_vis_mask=visual_key_mask,
        ent_txt_mask=text_key_mask,
        dim_str=args.dim,
        num_head=args.num_head,
        dim_hid=args.hidden_dim,
        num_layer_enc_ent=args.num_layer_enc_ent,
        num_layer_enc_rel=args.num_layer_enc_rel,
        num_layer_dec=args.num_layer_dec,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        vis_dropout=args.vis_dropout,
        txt_dropout=args.txt_dropout,
        visual_token_index=visual_token_index,
        text_token_index=text_token_index,
        score_function=args.score_function,
        use_dual_branch=args.use_dual_branch,
        use_cross_attn=args.use_cross_attn,
        curvature_init=args.curvature_init,
        use_diversity_reg=args.use_diversity_reg,
        diversity_margin=args.diversity_margin,
        device=device,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    base_params, hyp_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (hyp_params if 'hyperbolic_head' in name else base_params).append(p)
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': args.lr, 'weight_decay': args.decay},
        {'params': hyp_params, 'lr': args.lr * args.hyp_lr_factor, 'weight_decay': args.decay},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.step_size, T_mult=2)

    mem_bank = MemoryBank(dim=args.dim, size=args.memory_bank_size, device=device) if args.use_memory_bank else None

    best_mrr = 0.0
    start = time.time()
    logger.info("EPOCH\tLOSS\tTOTAL TIME")

    for epoch in range(1, args.num_epoch + 1):
        total_loss = 0.0
        total_kgc_loss = 0.0
        total_sacl_loss = 0.0
        total_front_loss = 0.0
        total_sem_loss = 0.0
        total_ang_loss = 0.0
        total_dist_loss = 0.0
        total_rad_loss = 0.0
        total_fuse_reg_loss = 0.0

        alpha = min(1.0, epoch / float(args.anneal_epochs)) if args.angle_target_anneal and args.anneal_epochs > 0 else 1.0
        pos_margin_deg = args.pos_margin_deg_start + (args.pos_margin_deg_end - args.pos_margin_deg_start) * alpha
        neg_margin_deg = args.neg_margin_deg_start + (args.neg_margin_deg_end - args.neg_margin_deg_start) * alpha
        pos_margin_rad = math.radians(pos_margin_deg)
        neg_margin_rad = math.radians(neg_margin_deg)

        gate_t_values, gate_v_values = [], []
        sem_similarities, hyp_angles = [], []
        hyp_radius_t, hyp_radius_v = [], []
        curvature_t_values, curvature_v_values = [], []
        ang_pos_list, ang_neg_list, neg_cos_list = [], [], []

        for batch, label in KG_Loader:
            if args.use_dual_branch:
                ent_embs, rel_embs, closs, intermediates = model(return_intermediates=True)
            else:
                ent_embs, rel_embs, closs = model()
                intermediates = None

            scores = model.score(ent_embs, rel_embs, batch.to(device))
            kgc_loss = loss_fn(scores, label.to(device))
            sacl_loss = closs * 0.01

            front_loss = torch.tensor(0.0, device=ent_embs.device)
            fuse_reg_loss = torch.tensor(0.0, device=ent_embs.device)

            if args.use_dual_branch and intermediates is not None:
                t_sem = intermediates['t_sem']
                v_sem = intermediates['v_sem']
                t_hyp = intermediates['t_hyp']
                v_hyp = intermediates['v_hyp']
                c_t = intermediates['c_t']
                c_v = intermediates['c_v']
                vis_available = intermediates.get('vis_available', None)
                txt_available = intermediates.get('txt_available', None)

                front_loss_dict = compute_front_loss(
                    t_sem=t_sem,
                    v_sem=v_sem,
                    t_hyp=t_hyp,
                    v_hyp=v_hyp,
                    c_t=c_t,
                    c_v=c_v,
                    lambda_h=args.lambda_h,
                    lambda_r=args.lambda_r,
                    tau=args.tau,
                    margin=args.angle_margin,
                    r_t_target=args.r_t_target,
                    r_v_target=args.r_v_target,
                    num_neg_samples=args.num_neg_samples,
                    vis_available=vis_available,
                    txt_available=txt_available,
                    loss_type=args.angle_loss_type,
                    pos_margin_rad=pos_margin_rad,
                    neg_margin_rad=neg_margin_rad,
                    use_hard_neg=args.neg_hard_mining,
                    hard_neg_k=args.neg_hard_k,
                    return_ang_stats=True,
                    memory_bank=mem_bank,
                    neg_source=args.neg_source,
                    memory_cand_size=args.memory_cand_size,
                    memory_ratio=args.memory_ratio,
                    lambda_d=args.lambda_d,
                    use_dist_hinge=args.use_dist_hinge,
                    dist_pos_margin=args.dist_pos_margin,
                    dist_neg_margin=args.dist_neg_margin,
                )

                front_loss = front_loss_dict['L_front']

                gate_info = intermediates['gate_info']
                t_hyp_tangent = model.hyperbolic_head.log_map(t_hyp, c_t)
                v_hyp_tangent = model.hyperbolic_head.log_map(v_hyp, c_v)
                diversity_loss = model.fusion.compute_diversity_loss(t_sem, v_sem, t_hyp_tangent, v_hyp_tangent)
                gate_reg_loss_val = model.fusion.compute_gate_reg_loss(
                    gate_info,
                    target_t=args.gate_t_target,
                    target_v=args.gate_v_target,
                    difficulty_aware=False,
                )
                fuse_reg_loss = diversity_loss + args.lambda_f * gate_reg_loss_val
                # 语义分支FERF-lite重构损失（仅文本/视觉）
                try:
                    recon_sem_loss = model.semantic_recon_loss()
                except Exception:
                    recon_sem_loss = torch.tensor(0.0, device=ent_embs.device)
                fuse_reg_loss = fuse_reg_loss + args.lambda_rec_sem * recon_sem_loss

                total_sem_loss += front_loss_dict['L_sem'].item()
                total_ang_loss += front_loss_dict['L_ang'].item()
                if 'L_dist' in front_loss_dict:
                    total_dist_loss += front_loss_dict['L_dist'].item()
                total_rad_loss += front_loss_dict['L_rad'].item()
                total_fuse_reg_loss += fuse_reg_loss.item()

                with torch.no_grad():
                    if 'ang_stats' in front_loss_dict:
                        ang_stats = front_loss_dict['ang_stats']
                        ang_pos_list.append(float(ang_stats['pos_loss'].cpu()))
                        ang_neg_list.append(float(ang_stats['neg_loss'].cpu()))
                        neg_cos_list.append(float(ang_stats['neg_cos_mean'].cpu()))
                    gate_t = gate_info['gate_t'].detach().cpu()
                    gate_v = gate_info['gate_v'].detach().cpu()
                    gate_t_values.append(gate_t.mean().item())
                    gate_v_values.append(gate_v.mean().item())
                    t_sem_n = torch.nn.functional.normalize(t_sem, p=2, dim=-1)
                    v_sem_n = torch.nn.functional.normalize(v_sem, p=2, dim=-1)
                    sem_similarities.append((t_sem_n * v_sem_n).sum(dim=-1).mean().item())
                    from utils.hyperbolic import lorentz_angle, lorentz_radius, lorentz_inner, lorentz_norm_sq
                    inner = lorentz_inner(t_hyp, v_hyp).detach()
                    t_norm = torch.sqrt(torch.clamp(-lorentz_norm_sq(t_hyp), min=1e-10)).detach()
                    v_norm = torch.sqrt(torch.clamp(-lorentz_norm_sq(v_hyp), min=1e-10)).detach()
                    cos_angle = torch.clamp(-inner / (t_norm * v_norm), min=-1.0 + 1e-6, max=1.0 - 1e-6)
                    angles = torch.acos(cos_angle).cpu()
                    hyp_angles.append(angles.mean().item())
                    cos_angle_mean = cos_angle.mean().cpu().item()
                    if not hasattr(model, '_cos_angle_history'):
                        model._cos_angle_history = []
                    model._cos_angle_history.append(cos_angle_mean)
                    r_t = lorentz_radius(t_hyp, c_t).detach().cpu()
                    r_v = lorentz_radius(v_hyp, c_v).detach().cpu()
                    hyp_radius_t.append(r_t.mean().item())
                    hyp_radius_v.append(r_v.mean().item())
                    curvature_t_values.append(c_t.item() if isinstance(c_t, torch.Tensor) else c_t)
                    curvature_v_values.append(c_v.item() if isinstance(c_v, torch.Tensor) else c_v)

                if mem_bank is not None:
                    with torch.no_grad():
                        mem_bank.enqueue(t_hyp_tangent, v_hyp_tangent)

            loss = kgc_loss + sacl_loss + front_loss + fuse_reg_loss
            total_loss += loss.item()
            total_kgc_loss += kgc_loss.item()
            total_sacl_loss += sacl_loss.item()
            total_front_loss += front_loss.item() if isinstance(front_loss, torch.Tensor) else float(front_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

        scheduler.step()

        elapsed = int(time.time() - start)
        eh, em, es = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60

        if args.use_dual_branch:
            avg_gate_t = np.mean(gate_t_values) if gate_t_values else 0.0
            avg_gate_v = np.mean(gate_v_values) if gate_v_values else 0.0
            avg_sem_sim = np.mean(sem_similarities) if sem_similarities else 0.0
            avg_hyp_angle = np.mean(hyp_angles) if hyp_angles else 0.0
            avg_radius_t = np.mean(hyp_radius_t) if hyp_radius_t else 0.0
            avg_radius_v = np.mean(hyp_radius_v) if hyp_radius_v else 0.0
            avg_c_t = np.mean(curvature_t_values) if curvature_t_values else 0.0
            avg_c_v = np.mean(curvature_v_values) if curvature_v_values else 0.0
            avg_ang_pos = np.mean(ang_pos_list) if ang_pos_list else 0.0
            avg_ang_neg = np.mean(ang_neg_list) if ang_neg_list else 0.0
            avg_neg_cos = np.mean(neg_cos_list) if neg_cos_list else 0.0

            logger.info(
                f"{epoch} \t Total: {total_loss:.6f} \t KGC: {total_kgc_loss:.6f} \t "
                f"SACL: {total_sacl_loss:.6f} \t Front: {total_front_loss:.6f} \t "
                f"Sem: {total_sem_loss:.6f} \t Ang: {total_ang_loss:.6f} \t "
                f"Rad: {total_rad_loss:.6f} \t Dist: {total_dist_loss:.6f} \t FuseReg: {total_fuse_reg_loss:.6f}"
            )
            avg_cos_angle = np.mean(model._cos_angle_history[-10:]) if hasattr(model, '_cos_angle_history') and len(model._cos_angle_history) > 0 else 0.0
            logger.info(
                f"  Metrics: Gate_T: {avg_gate_t:.4f} \t Gate_V: {avg_gate_v:.4f} \t "
                f"SemSim: {avg_sem_sim:.4f} \t HypAngle: {avg_hyp_angle:.4f} \t CosAngle: {avg_cos_angle:.4f} \t "
                f"AngPos: {avg_ang_pos:.4f} \t AngNeg: {avg_ang_neg:.4f} \t NegCos: {avg_neg_cos:.4f} \t "
                f"PosThrDeg: {pos_margin_deg:.1f} \t NegThrDeg: {neg_margin_deg:.1f} \t "
                f"R_T: {avg_radius_t:.4f} \t R_V: {avg_radius_v:.4f} \t "
                f"C_T: {avg_c_t:.4f} \t C_V: {avg_c_v:.4f} \t "
                f"Time: {eh}h-{em}m-{es}s"
            )
        else:
            logger.info(f"{epoch} \t {total_loss:.6f} \t {eh}h-{em}m-{es}s")

        if (epoch) % args.valid_epoch == 0:
            model.eval()
            with torch.no_grad():
                ent_embs, rel_embs, closs = model(return_intermediates=False)
                lp_list_rank = []
                for triplet in tqdm(KG.valid):
                    h, r, t = triplet
                    head_score = model.score(ent_embs, rel_embs, torch.tensor([[KG.num_ent + KG.num_rel, r + KG.num_ent, t + KG.num_rel]]).to(device))[0].detach().cpu().numpy()
                    head_rank = calculate_rank(head_score, h, KG.filter_dict[(-1, r, t)])
                    tail_score = model.score(ent_embs, rel_embs, torch.tensor([[h + KG.num_rel, r + KG.num_ent, KG.num_ent + KG.num_rel]]).to(device))[0].detach().cpu().numpy()
                    tail_rank = calculate_rank(tail_score, t, KG.filter_dict[(h, r, -1)])
                    lp_list_rank.append(head_rank); lp_list_rank.append(tail_rank)
                lp_list_rank = np.array(lp_list_rank)
                mr, mrr, hit10, hit3, hit1 = metrics(lp_list_rank)
                logger.info("Link Prediction on Validation Set")
                logger.info(f"MR: {mr}")
                logger.info(f"MRR: {mrr}")
                logger.info(f"Hit10: {hit10}")
                logger.info(f"Hit3: {hit3}")
                logger.info(f"Hit1: {hit1}")

                lp_list_rank = []
                for triplet in tqdm(KG.test):
                    h, r, t = triplet
                    head_score = model.score(ent_embs, rel_embs, torch.tensor([[KG.num_ent + KG.num_rel, r + KG.num_ent, t + KG.num_rel]]).to(device))[0].detach().cpu().numpy()
                    head_rank = calculate_rank(head_score, h, KG.filter_dict[(-1, r, t)])
                    tail_score = model.score(ent_embs, rel_embs, torch.tensor([[h + KG.num_rel, r + KG.num_ent, KG.num_ent + KG.num_rel]]).to(device))[0].detach().cpu().numpy()
                    tail_rank = calculate_rank(tail_score, t, KG.filter_dict[(h, r, -1)])
                    lp_list_rank.append(head_rank); lp_list_rank.append(tail_rank)
                lp_list_rank = np.array(lp_list_rank)
                mr, mrr, hit10, hit3, hit1 = metrics(lp_list_rank)
                logger.info("Link Prediction on Test Set")
                logger.info(f"MR: {mr}")
                logger.info(f"MRR: {mrr}")
                logger.info(f"Hit10: {hit10}")
                logger.info(f"Hit3: {hit3}")
                logger.info(f"Hit1: {hit1}")

                if best_mrr < mrr:
                    best_mrr = mrr
                    best_result = (mr, mrr, hit10, hit3, hit1)
            model.train()

    logger.info(f"Done! {args.data}. The best results are shown below:")
    if 'best_result' in locals():
        logger.info(f"Best result: {best_result}")
    else:
        logger.info("Best result: N/A (eval disabled)")
