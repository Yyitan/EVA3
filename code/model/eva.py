import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn.modules.loss import CrossEntropyLoss

from clip_modules.clip_model import load_clip
from clip_modules.tokenization_clip import SimpleTokenizer
from model.moe_adapter import MoEAdapter


class EVA(nn.Module):
    """EVA: Mixture-of-Experts Framework for Semantic Variant Alignment (ICLR 2026).

    Built on frozen CLIP ViT-L/14. Two ideas:
      1. Domain-expert adaption: a MoE adapter parallel to the FFN in *every*
         layer of both the image and text encoders.
      2. Semantic variant alignment: text-to-image (partner-max) and
         image-to-text (per-expert CLS variants) primitive alignment.
    """

    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.offset = offset                                 # = len(attributes)
        self.n_states = len(attributes)
        self.n_objs = len(classes)

        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length)
        self.tokenizer = SimpleTokenizer()
        self.dtype = self.clip.dtype                          # fp32 for this ckpt
        self.attr_dropout = nn.Dropout(config.attr_dropout)

        # ---- soft prompts (CSP/Troika style) ----
        self.token_ids, soft_att_obj, comp_ctx, attr_ctx, obj_ctx = self.construct_soft_prompt()

        # freeze the whole CLIP backbone first
        for p in self.parameters():
            p.requires_grad = False

        assert 'ViT' in config.clip_model, "EVA only supports a ViT image encoder."

        # ---- trainable soft prompt parameters ----
        self.soft_att_obj = nn.Parameter(soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx)
        self.attr_ctx_vectors = nn.Parameter(attr_ctx)
        self.obj_ctx_vectors = nn.Parameter(obj_ctx)

        # ---- MoE adapters (one per layer, parallel to FFN) ----
        img_width = self.clip.visual.transformer.width       # 1024
        txt_width = self.clip.transformer.width              # 768
        self.n_img_layers = self.clip.visual.transformer.layers
        self.n_txt_layers = self.clip.transformer.layers
        self.image_moe = nn.ModuleList([
            MoEAdapter(img_width, config.n_routed, config.topk_expert, config.expert_dim, getattr(config, 'expert_scale', 1.0), getattr(config, 'expert_dropout', 0.0))
            for _ in range(self.n_img_layers)
        ])
        self.text_moe = nn.ModuleList([
            MoEAdapter(txt_width, config.n_routed, config.topk_expert, config.expert_dim, getattr(config, 'expert_scale', 1.0), getattr(config, 'expert_dropout', 0.0))
            for _ in range(self.n_txt_layers)
        ])

        # ---- learnable temperatures for text-to-image primitive scoring ----
        self.tau_s = nn.Parameter(torch.tensor(float(config.tau_init)))
        self.tau_o = nn.Parameter(torch.tensor(float(config.tau_init)))

    # ------------------------------------------------------------------ #
    #  soft prompt construction (identical recipe to Troika)             #
    # ------------------------------------------------------------------ #
    def construct_soft_prompt(self):
        token_ids = self.tokenizer(self.config.prompt_template,
                                   context_length=self.config.context_length).cuda()

        tokenized = torch.cat([
            self.tokenizer(tok, context_length=self.config.context_length)
            for tok in self.attributes + self.classes
        ])
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx = embedding[0, 1:1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx = embedding[1, 1:1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx = embedding[2, 1:1 + n_ctx[2], :].to(self.clip.dtype)
        return token_ids, soft_att_obj, comp_ctx, attr_ctx, obj_ctx

    def construct_token_tensors(self, pair_idx):
        """Build the embedded token sequences for the three prompts.

        Returns a list [Pc_tokens, Ps_tokens, Po_tokens]:
            Pc_tokens: [n_pairs, ctx_len, d_txt]
            Ps_tokens: [n_states, ctx_len, d_txt]
            Po_tokens: [n_objs,  ctx_len, d_txt]
        """
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        num_elements = [len(pair_idx), self.offset, len(self.classes)]
        token_tensor = []
        for i in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i].repeat(num_elements[i], 1)
            token_tensor.append(
                self.clip.token_embedding(class_token_ids.cuda()).type(self.clip.dtype)
            )

        eos_idx = [int(self.token_ids[i].argmax()) for i in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj)

        # Pc = [ctx, state, object]
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[attr_idx].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[obj_idx + self.offset].type(self.clip.dtype)
        token_tensor[0][:, 1:len(self.comp_ctx_vectors) + 1, :] = self.comp_ctx_vectors.type(self.clip.dtype)
        # Ps = [ctx, state]
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[:self.offset].type(self.clip.dtype)
        token_tensor[1][:, 1:len(self.attr_ctx_vectors) + 1, :] = self.attr_ctx_vectors.type(self.clip.dtype)
        # Po = [ctx, object]
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[self.offset:].type(self.clip.dtype)
        token_tensor[2][:, 1:len(self.obj_ctx_vectors) + 1, :] = self.obj_ctx_vectors.type(self.clip.dtype)
        return token_tensor

    # ------------------------------------------------------------------ #
    #  encoders with MoE adapters injected in every layer                #
    # ------------------------------------------------------------------ #
    def _use_ckpt(self):
        return getattr(self.config, 'grad_checkpoint', False) and self.training

    def _img_block(self, x, i):
        """One image transformer block with the MoE adapter parallel to FFN."""
        blk = self.clip.visual.transformer.resblocks[i]
        x = x + blk.attention(blk.ln_1(x))
        normed = blk.ln_2(x)
        return x + blk.mlp(normed) + self.image_moe[i](normed)

    def _txt_block(self, x, i):
        """One text transformer block with the MoE adapter parallel to FFN."""
        blk = self.clip.transformer.resblocks[i]
        x = x + blk.attention(blk.ln_1(x))
        normed = blk.ln_2(x)
        out = x + blk.mlp(normed)
        # [diagnostic] optionally drop the text-encoder MoE (Troika-style frozen text)
        if not getattr(self.config, 'disable_text_moe', False):
            out = out + self.text_moe[i](normed)
        return out

    def encode_image(self, x):
        """Returns (f_c [B,d], variants V [B, Ne+1, d]) in the projected space."""
        visual = self.clip.visual
        x = visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        x = x.permute(1, 0, 2)                                # NLD -> LND

        use_ckpt = self._use_ckpt()
        variants_hidden = None
        for i in range(self.n_img_layers):
            if i == self.n_img_layers - 1:
                # last layer is run normally so we can extract the CLS variants
                blk = visual.transformer.resblocks[i]
                x = x + blk.attention(blk.ln_1(x))
                normed = blk.ln_2(x)
                ffn = blk.mlp(normed)
                moe = self.image_moe[i](normed)
                cls_normed = normed[0]                        # [B, d_img]
                base_cls = (x + ffn)[0]                        # [B, d_img]
                expert_cls = self.image_moe[i].expert_outputs(cls_normed)  # [B, Ne+1, d_img]
                variants_hidden = base_cls.unsqueeze(1) + expert_cls       # [B, Ne+1, d_img]
                x = x + ffn + moe
            elif use_ckpt:
                x = cp.checkpoint(self._img_block, x, i, use_reentrant=False)
            else:
                x = self._img_block(x, i)

        x = x.permute(1, 0, 2)                                # LND -> NLD
        f_c = visual.ln_post(x[:, 0, :])
        V = visual.ln_post(variants_hidden)
        if visual.proj is not None:
            f_c = f_c @ visual.proj
            V = V @ visual.proj
        return f_c, V

    def encode_text(self, token_ids, token_tensors):
        """Encode one prompt set with MoE adapters. Returns [n, d] text features."""
        x = token_tensors.type(self.dtype) + self.clip.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)                                # NLD -> LND
        use_ckpt = self._use_ckpt()
        for i in range(self.n_txt_layers):
            if use_ckpt:
                x = cp.checkpoint(self._txt_block, x, i, use_reentrant=False)
            else:
                x = self._txt_block(x, i)
        x = x.permute(1, 0, 2)                                # LND -> NLD
        x = self.clip.ln_final(x).type(self.dtype)
        tf = x[torch.arange(x.shape[0]), token_ids.argmax(dim=-1)] @ self.clip.text_projection
        return tf

    def _encode_all_text(self, pair_idx):
        """Encode Pc/Ps/Po for the given pair set. Returns normalized features."""
        tt = self.construct_token_tensors(pair_idx)
        t_c = self.encode_text(self.token_ids[0], tt[0])
        t_s = self.encode_text(self.token_ids[1], tt[1])
        t_o = self.encode_text(self.token_ids[2], tt[2])
        t_c = t_c / t_c.norm(dim=-1, keepdim=True)
        t_s = t_s / t_s.norm(dim=-1, keepdim=True)
        t_o = t_o / t_o.norm(dim=-1, keepdim=True)
        return t_c, t_s, t_o

    # ------------------------------------------------------------------ #
    #  training forward                                                  #
    # ------------------------------------------------------------------ #
    def forward(self, batch, idx):
        img = batch[0].cuda()
        t_c, t_s, t_o = self._encode_all_text(idx)
        f_c, V = self.encode_image(img.type(self.dtype))
        f_c = f_c / f_c.norm(dim=-1, keepdim=True)
        V = V / V.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits_c = logit_scale * (f_c @ t_c.t())             # [B, n_pairs]

        predict = {
            'logits_c': logits_c,
            'f_c': f_c,
            'V': V,
            't_s': t_s,
            't_o': t_o,
            'pair_attr': idx[:, 0],
            'pair_obj': idx[:, 1],
            'logit_scale': logit_scale,
        }
        return predict

    # ------------------------------------------------------------------ #
    #  losses                                                            #
    # ------------------------------------------------------------------ #
    def _partner_max(self, pc, group_idx, num_groups):
        """pc:[B,P], group_idx:[P] -> [B,num_groups] max over partners per group."""
        B = pc.shape[0]
        out = pc.new_zeros(B, num_groups)
        index = group_idx.unsqueeze(0).expand(B, -1)
        out = out.scatter_reduce(1, index, pc, reduce='amax', include_self=False)
        return out

    def loss_calu(self, predict, target):
        ce = CrossEntropyLoss()
        attr_label = target[1].cuda()
        obj_label = target[2].cuda()
        pair_label = target[3].cuda()

        logits_c = predict['logits_c']
        f_c = predict['f_c']
        V = predict['V']
        t_s = predict['t_s']
        t_o = predict['t_o']
        pair_attr = predict['pair_attr']
        pair_obj = predict['pair_obj']
        logit_scale = predict['logit_scale']

        # --- composition loss ---
        loss_c = ce(logits_c, pair_label)

        # --- text-to-image primitive loss (partner max over current pair set) ---
        # Hypothesis-1(b): build the per-state/object logits from the RAW
        # composition logits (scale ~100) instead of the softmax probabilities,
        # so the cross-entropy has a meaningful gradient. Inference (logit_infer)
        # stays on the probability scale per Eq.16.
        if getattr(self.config, 't2i_use_logits', False):
            ps = self._partner_max(logits_c, pair_attr, self.n_states)
            po = self._partner_max(logits_c, pair_obj, self.n_objs)
        else:
            pc = F.softmax(logits_c, dim=-1)
            ps = self._partner_max(pc, pair_attr, self.n_states) * self.tau_s
            po = self._partner_max(pc, pair_obj, self.n_objs) * self.tau_o
        loss_s = ce(ps, attr_label)
        loss_o = ce(po, obj_label)

        # --- image-to-text variant alignment loss ---
        B = f_c.shape[0]
        t_s_b = t_s[attr_label]                               # [B, d]
        t_o_b = t_o[obj_label]                                # [B, d]
        A_s = torch.einsum('bnd,bd->bn', V, t_s_b)
        A_o = torch.einsum('bnd,bd->bn', V, t_o_b)
        A_v = torch.einsum('bnd,bd->bn', V, f_c)
        AS = A_s + self.config.alpha * A_v
        AO = A_o + self.config.alpha * A_v
        idx_s = AS.argmax(dim=-1)
        idx_o = AO.argmax(dim=-1)
        ar = torch.arange(B, device=V.device)
        f_s = V[ar, idx_s]                                    # [B, d]
        f_o = V[ar, idx_o]
        logits_vs = logit_scale * (f_s @ t_s.t())            # [B, n_states]
        logits_vo = logit_scale * (f_o @ t_o.t())            # [B, n_objs]
        loss_vs = ce(logits_vs, attr_label)
        loss_vo = ce(logits_vo, obj_label)

        loss = loss_c \
            + self.config.lambda_1 * (loss_s + loss_o) \
            + self.config.lambda_2 * (loss_vs + loss_vo)
        return loss

    # ------------------------------------------------------------------ #
    #  inference scoring (Eq. 16)                                        #
    # ------------------------------------------------------------------ #
    def logit_infer(self, predict, pairs):
        logits_c = predict['logits_c']
        pair_attr = predict['pair_attr']
        pair_obj = predict['pair_obj']
        pc = F.softmax(logits_c, dim=-1)
        ps = self._partner_max(pc, pair_attr, self.n_states) * self.tau_s
        po = self._partner_max(pc, pair_obj, self.n_objs) * self.tau_o
        score = pc + self.config.beta * (ps[:, pair_attr] + po[:, pair_obj])
        return score

    # ------------------------------------------------------------------ #
    #  evaluation helpers (text caching, used by test.py)                #
    # ------------------------------------------------------------------ #
    def build_text_cache(self, pairs, text_bs):
        """Encode Pc (chunked), Ps, Po once for evaluation. Returns a dict."""
        t_c_list = []
        n = pairs.shape[0]
        n_chunk = (n + text_bs - 1) // text_bs
        for i in range(n_chunk):
            chunk = pairs[i * text_bs:(i + 1) * text_bs, :]
            tt = self.construct_token_tensors(chunk)
            t_c = self.encode_text(self.token_ids[0], tt[0])
            t_c = t_c / t_c.norm(dim=-1, keepdim=True)
            t_c_list.append(t_c)
        t_c_all = torch.cat(t_c_list, dim=0)
        # Ps / Po: all states / objects (cheap, one shot)
        tt = self.construct_token_tensors(pairs)
        t_s = self.encode_text(self.token_ids[1], tt[1])
        t_o = self.encode_text(self.token_ids[2], tt[2])
        t_s = t_s / t_s.norm(dim=-1, keepdim=True)
        t_o = t_o / t_o.norm(dim=-1, keepdim=True)
        return {
            't_c': t_c_all, 't_s': t_s, 't_o': t_o,
            'pair_attr': pairs[:, 0], 'pair_obj': pairs[:, 1],
            'logit_scale': self.clip.logit_scale.exp(),
        }

    def forward_with_cache(self, batch, cache):
        """Image forward + assemble predict dict using cached text features."""
        img = batch[0].cuda()
        f_c, V = self.encode_image(img.type(self.dtype))
        f_c = f_c / f_c.norm(dim=-1, keepdim=True)
        V = V / V.norm(dim=-1, keepdim=True)
        logits_c = cache['logit_scale'] * (f_c @ cache['t_c'].t())
        return {
            'logits_c': logits_c,
            'f_c': f_c,
            'V': V,
            't_s': cache['t_s'],
            't_o': cache['t_o'],
            'pair_attr': cache['pair_attr'],
            'pair_obj': cache['pair_obj'],
            'logit_scale': cache['logit_scale'],
        }
