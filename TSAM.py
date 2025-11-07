import torch
import torch.nn as nn
import numpy as np
import sys
import os
# 确保当前目录在Python路径中，优先导入本地模块
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
from Tucker import *
from CLossModel import *
from TransELayer import *
from Rotate import *
from models.heads import SemanticHead, HyperbolicHead
from modules.fusion import GatedFusion

class TSAM(nn.Module):
    def __init__(
            self, 
            num_ent, 
            num_rel, 
            ent_vis_mask,
            ent_txt_mask,
            dim_str,
            num_head,
            dim_hid,
            num_layer_enc_ent,
            num_layer_enc_rel,
            num_layer_dec,
            dropout = 0.1,
            emb_dropout = 0.6, 
            vis_dropout = 0.1, 
            txt_dropout = 0.1,
            visual_token_index = None, 
            text_token_index = None,
            score_function = "tucker",
            # 新增分支参数
            use_dual_branch: bool = True,
            use_cross_attn: bool = True,
            curvature_init: float = 0.1,
            use_diversity_reg: bool = False,
            diversity_margin: float = 0.5,
            device = None,  # 新增：设备参数
        ):
        super(TSAM, self).__init__()
        self.dim_str = dim_str
        self.num_head = num_head
        self.dim_hid = dim_hid
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.data_type=torch.float32
        
        # 确定设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        visual_tokens = torch.load("tokens/visual.pth")
        textual_tokens = torch.load("tokens/textual.pth")
        self.visual_token_index = visual_token_index
        self.visual_token_embedding = nn.Embedding.from_pretrained(visual_tokens).requires_grad_(False)
        self.text_token_index = text_token_index
        self.text_token_embedding = nn.Embedding.from_pretrained(textual_tokens).requires_grad_(False)
        self.score_function = score_function
        self.scale = torch.Tensor([1. / np.sqrt(self.dim_str)]).to(self.device)
        self.visual_token_embedding.requires_grad_(False)
        self.text_token_embedding.requires_grad_(False)

        # 保留原始可用性掩码，便于编码与缺模态处理（True 表示缺失/不可用）
        self.ent_vis_mask = ent_vis_mask.bool().to(self.device)
        self.ent_txt_mask = ent_txt_mask.bool().to(self.device)

        false_ents = torch.full((self.num_ent,1),False, device=self.device)
        self.ent_mask = torch.cat([false_ents, false_ents, self.ent_vis_mask, self.ent_txt_mask], dim = 1)
        
        # print(self.ent_mask.shape)
        false_rels = torch.full((self.num_rel,1),False, device=self.device)
        self.rel_mask = torch.cat([false_rels, false_rels], dim = 1)
        
        self.ent_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.rel_token = nn.Parameter(torch.Tensor(1, 1, dim_str))
        self.ent_embeddings = nn.Parameter(torch.Tensor(num_ent, 1, dim_str))
        self.rel_embeddings = nn.Parameter(torch.Tensor(num_rel, 1 ,dim_str))
        self.lp_token = nn.Parameter(torch.Tensor(1,dim_str))

        self.str_ent_ln = nn.LayerNorm(dim_str)
        self.str_rel_ln = nn.LayerNorm(dim_str)
        self.vis_ln = nn.LayerNorm(dim_str)
        self.txt_ln = nn.LayerNorm(dim_str)

        self.embdr = nn.Dropout(p = emb_dropout)
        self.visdr = nn.Dropout(p = vis_dropout)
        self.txtdr = nn.Dropout(p = txt_dropout)


        self.pos_str_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_str_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_vis_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_txt_rel = nn.Parameter(torch.Tensor(1,1,dim_str))

        self.pos_head = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_rel = nn.Parameter(torch.Tensor(1,1,dim_str))
        self.pos_tail = nn.Parameter(torch.Tensor(1,1,dim_str))
        
        self.proj_ent_vis = nn.Linear(32, dim_str)
        self.proj_ent_txt = nn.Linear(768, dim_str)
        # 轻量文本增强：注意力池化 + Adapter（不改训练超参）
        self.txt_pool_proj = nn.Linear(dim_str, 1)
        self.txt_adapter = nn.Sequential(
            nn.LayerNorm(dim_str),
            nn.Linear(dim_str, max(32, dim_str // 4)),
            nn.GELU(),
            nn.Linear(max(32, dim_str // 4), dim_str),
        )
        # 视觉可学习回退（缺视觉时将结构映射为“视觉风格”）
        self.vis_fallback = nn.Sequential(
            nn.LayerNorm(dim_str),
            nn.Linear(dim_str, max(32, dim_str // 4)),
            nn.GELU(),
            nn.Linear(max(32, dim_str // 4), dim_str),
        )

        # FERF-lite 重构器（仅语义：文本/视觉双向重构）
        self.text_recon_mlp = nn.Sequential(
            nn.LayerNorm(2 * dim_str),
            nn.Linear(2 * dim_str, dim_str),
            nn.GELU(),
            nn.Linear(dim_str, dim_str),
        )
        self.vis_recon_mlp = nn.Sequential(
            nn.LayerNorm(2 * dim_str),
            nn.Linear(2 * dim_str, dim_str),
            nn.GELU(),
            nn.Linear(dim_str, dim_str),
        )
        # 缺文本时的可学习回退：将结构表示映射到“文本风格”后再回退
        self.text_fallback = nn.Sequential(
            nn.LayerNorm(dim_str),
            nn.Linear(dim_str, max(32, dim_str // 4)),
            nn.GELU(),
            nn.Linear(max(32, dim_str // 4), dim_str),
        )

        ######
        self.context_vec = nn.Parameter(torch.randn((1, dim_str)))
        
        self.act = nn.Softmax(dim=1)
        

        ent_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.ent_encoder = nn.TransformerEncoder(ent_encoder_layer, num_layer_enc_ent)
        rel_encoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.rel_encoder = nn.TransformerEncoder(rel_encoder_layer, num_layer_enc_rel)
         
        decoder_layer = nn.TransformerEncoderLayer(dim_str, num_head, dim_hid, dropout, batch_first = True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layer_dec)
        
        

        
        self.num_con = 256
        self.num_vis = ent_vis_mask.shape[1]
        if self.score_function == "tucker":
            self.tucker_decoder = TuckERLayer(dim_str, dim_str)
        elif self.score_function == "transe":
            self.transE_decoder = TransELayer()
        elif self.score_function == "rotate":
            self.rotate_decoder = RotatELayer(dim_str)
        else:
            pass
        
        # 新增：双分支架构
        self.use_dual_branch = use_dual_branch
        if use_dual_branch:
            # 语义分支头（欧式拉近）
            self.semantic_head = SemanticHead(
                d=dim_str,
                use_cross_attn=use_cross_attn,
                num_heads=num_head,
                dropout=dropout
            )
            
            # 几何分支头（双曲拉远）
            self.hyperbolic_head = HyperbolicHead(
                d=dim_str,
                curvature_init=curvature_init,
                learnable_curvature=True,
                dropout=dropout
            )
            
            # 门控融合模块
            self.fusion = GatedFusion(
                d=dim_str,
                use_diversity_reg=use_diversity_reg,
                diversity_margin=diversity_margin,
                dropout=dropout
            )
        
        self.init_weights()
        torch.save(self.visual_token_embedding, open("visual_token.pth", "wb"))
        torch.save(self.text_token_embedding, open("textual_token.pth", "wb"))
        

    def init_weights(self):
        nn.init.xavier_uniform_(self.ent_embeddings)
        nn.init.xavier_uniform_(self.rel_embeddings)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_ent_txt.weight)
        nn.init.xavier_uniform_(self.ent_token)
        nn.init.xavier_uniform_(self.rel_token)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        nn.init.xavier_uniform_(self.pos_str_rel)
        nn.init.xavier_uniform_(self.pos_vis_rel)
        nn.init.xavier_uniform_(self.pos_txt_rel)
        nn.init.xavier_uniform_(self.pos_head)
        nn.init.xavier_uniform_(self.pos_rel)
        nn.init.xavier_uniform_(self.pos_tail)

    def forward(self, return_intermediates: bool = False):
        """
        Args:
            return_intermediates: 是否返回中间结果（用于训练时的损失计算）
        
        Returns:
            如果 return_intermediates=False: (ent_embs, rep_rel_str, closs)
            如果 return_intermediates=True: (ent_embs, rep_rel_str, closs, intermediates_dict)
        """
        ent_tkn = self.ent_token.tile(self.num_ent, 1, 1)
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)) + self.pos_str_ent
        
        entity_visual_tokens = self.visual_token_embedding(self.visual_token_index)
        rep_ent_vis = self.visdr(self.vis_ln(self.proj_ent_vis(entity_visual_tokens))) + self.pos_vis_ent
        entity_text_tokens = self.text_token_embedding(self.text_token_index)
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_ent_txt(entity_text_tokens))) + self.pos_txt_ent
        
        ent_tkn2 = ent_tkn.squeeze(1)
        ent_seq1 = torch.cat([ent_tkn, rep_ent_str, ], dim = 1)
        ent_seq2 = torch.cat([ent_tkn, rep_ent_vis, ], dim = 1)
        ent_seq3 = torch.cat([ent_tkn,  rep_ent_txt], dim = 1)
        str_embdding = self.ent_encoder(ent_seq1)[:,0]
        # 视觉通路：加入 key mask，避免缺失/填充 token 参与注意力
        vis_key_mask = torch.cat([
            torch.zeros((self.num_ent, 1), dtype=torch.bool, device=self.device),
            self.ent_vis_mask
        ], dim=1)
        vis_seq_enc = self.ent_encoder(ent_seq2, src_key_padding_mask=vis_key_mask)
        vis_embdding = vis_seq_enc[:,0]
        vis_m_raw = vis_embdding.clone()
        # 文本通路：加入 key mask，避免缺失/填充 token 参与注意力；随后注意力池化 + 适配器增强
        txt_key_mask = torch.cat([
            torch.zeros((self.num_ent, 1), dtype=torch.bool, device=self.device),
            self.ent_txt_mask
        ], dim=1)  # (num_ent, 1+L_txt)
        txt_seq_enc = self.ent_encoder(ent_seq3, src_key_padding_mask=txt_key_mask)  # (num_ent, L, d)
        txt_scores = self.txt_pool_proj(txt_seq_enc).squeeze(-1)  # (num_ent, L)
        txt_weights = torch.softmax(txt_scores, dim=1)
        txt_embdding = torch.sum(txt_weights.unsqueeze(-1) * txt_seq_enc, dim=1)  # (num_ent, d)
        txt_embdding = txt_embdding + self.txt_adapter(txt_embdding)
        txt_m_raw = txt_embdding.clone()
        
        # 原始对比损失（SACL）
        clmodel = CLoss()
        closs = clmodel(str_embdding, vis_embdding, txt_embdding)
        
        # 新增：双分支架构
        intermediates_dict = {}
        if self.use_dual_branch:
            # e_t = txt_embdding, e_v = vis_embdding (原始编码器输出)
            e_t = txt_embdding  # (num_ent, d)
            e_v = vis_embdding  # (num_ent, d)
            
            # 模态缺失处理：从ent_mask中提取模态可用性
            # ent_mask shape: (num_ent, 4) = [str_mask, str_mask, vis_mask, txt_mask]
            # ent_mask[:, 2] = vis_mask (True表示缺失，False表示存在)
            # ent_mask[:, 3] = txt_mask (True表示缺失，False表示存在)
            vis_available = ~self.ent_vis_mask.any(dim=1)  # 任一视觉token可用即视为存在
            txt_available = ~self.ent_txt_mask.any(dim=1)  # 任一文本token可用即视为存在
            
            # 语义分支（欧式拉近）
            t_sem, v_sem = self.semantic_head(e_t, e_v)
            
            # 几何分支（双曲拉远）
            t_hyp, v_hyp, c_t, c_v = self.hyperbolic_head(e_t, e_v)
            
            # 将双曲空间向量映射回切空间（欧式空间）
            t_hyp_tangent = self.hyperbolic_head.log_map(t_hyp, c_t)
            v_hyp_tangent = self.hyperbolic_head.log_map(v_hyp, c_v)
            
            # 门控融合
            t_fuse, v_fuse, gate_info = self.fusion(
                t_sem, v_sem,
                t_hyp_tangent, v_hyp_tangent
            )
            
            # 模态缺失处理：对于缺失的模态，使用结构表示作为fallback
            # 如果文本缺失，使用结构表示；如果视觉缺失，使用结构表示
            # 这里使用str_embdding作为fallback
            t_fuse = torch.where(txt_available.unsqueeze(-1), t_fuse, self.text_fallback(str_embdding))
            v_fuse = torch.where(vis_available.unsqueeze(-1), v_fuse, self.vis_fallback(str_embdding))
            
            # 使用融合后的表示替换原始表示
            txt_embdding = t_fuse
            vis_embdding = v_fuse
            
            # 保存中间结果（包括模态可用性信息）
            intermediates_dict = {
                't_sem': t_sem,
                'v_sem': v_sem,
                't_hyp': t_hyp,
                'v_hyp': v_hyp,
                't_fuse': t_fuse,
                'v_fuse': v_fuse,
                'c_t': c_t,
                'c_v': c_v,
                'gate_info': gate_info,
                'vis_available': vis_available,  # 模态可用性信息
                'txt_available': txt_available,  # 模态可用性信息
            }
        else:
            # 不使用双分支时，保持原有逻辑
            intermediates_dict = None
        
        # 原有的融合逻辑（使用融合后的 txt_embdding 和 vis_embdding）
        cands = torch.stack([ent_tkn2, str_embdding, vis_embdding, txt_embdding], dim=1)  # (1500, 4, 256)
        context_vec = self.context_vec
        
        att_weights = torch.sum(context_vec * cands* self.scale , dim=-1, keepdim=True)
        att_weights = self.act(att_weights)  
        ent_embs = torch.sum(att_weights * cands, dim=1)  

        rep_rel_str = self.embdr(self.str_rel_ln(self.rel_embeddings))
        
        if return_intermediates:
            return torch.cat([ent_embs, self.lp_token], dim = 0), rep_rel_str.squeeze(dim=1), closs, intermediates_dict
        else:
            return torch.cat([ent_embs, self.lp_token], dim = 0), rep_rel_str.squeeze(dim=1), closs


   

    def score(self, emb_ent, emb_rel, triplets):
        
        h_seq = emb_ent[triplets[:,0] - self.num_rel].unsqueeze(dim = 1) + self.pos_head
        r_seq = emb_rel[triplets[:,1] - self.num_ent].unsqueeze(dim = 1) + self.pos_rel
        t_seq = emb_ent[triplets[:,2] - self.num_rel].unsqueeze(dim = 1) + self.pos_tail
        dec_seq = torch.cat([h_seq, r_seq, t_seq], dim = 1)
        output_dec = self.decoder(dec_seq)
        rel_emb = output_dec[:, 1, :]
        ctx_emb = output_dec[triplets == self.num_ent + self.num_rel]     
        
        if self.score_function == "tucker":
            tucker_emb = self.tucker_decoder(ctx_emb, rel_emb)        
            scores = torch.mm(tucker_emb, emb_ent[:-1].transpose(1, 0))
           
        elif self.score_function == "transe":
            trans_embedding=self.transE_decoder(ctx_emb, rel_emb)
            
            scores = torch.mm(trans_embedding, emb_ent[:-1].transpose(1, 0))
        elif self.score_function == "rotate":
            rotae_emb = self.rotate_decoder(output_dec, rel_emb)
            scores = torch.mm(rotae_emb, emb_ent[:-1].transpose(1, 0))
            

        return scores
