import argparse

parser = argparse.ArgumentParser()

# ---- general / training ----
parser.add_argument("--model_name", help="model name", type=str, default='eva')
parser.add_argument("--lr", help="learning rate", type=float, default=1e-04)
parser.add_argument("--dataset", help="name of the dataset", type=str, default='mit-states')
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
parser.add_argument("--epoch_start", help="start epoch", default=0, type=int)
parser.add_argument("--train_batch_size", help="train batch size", default=48, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", default=16, type=int)
parser.add_argument("--num_workers", help="number of workers", default=4, type=int)
parser.add_argument("--context_length", help="clip context length", default=8, type=int)
parser.add_argument("--attr_dropout", help="dropout on the soft primitive embeddings", type=float, default=0.3)
parser.add_argument("--yml_path", help="yml path", type=str)
parser.add_argument("--clip_arch", help="clip checkpoint path", type=str)
parser.add_argument("--dataset_path", help="dataset path", type=str)
parser.add_argument("--save_path", help="save path", type=str)
parser.add_argument("--save_every_n", default=5, type=int, help="save the model every n epochs")
parser.add_argument("--save_final_model", help="save the final model state dict", action="store_true")
parser.add_argument("--load_model", default=None, help="load a trained model")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", help="gradient accumulation steps", default=1, type=int)
parser.add_argument("--same_prim_sample", help="if sample same prim samples", action="store_true")

# ---- evaluation ----
parser.add_argument("--open_world", help="evaluate on open world setup", default=False)
parser.add_argument("--bias", help="eval bias", type=float, default=1e3)
parser.add_argument("--topk", help="eval topk", type=int, default=1)
parser.add_argument("--text_encoder_batch_size", help="batch size of the text encoder", default=1024, type=int)
parser.add_argument('--threshold', type=float, default=None, help="optional feasibility threshold")
parser.add_argument('--threshold_trials', type=int, default=50, help="how many threshold values to try")

# ---- EVA: domain-expert adaption (MoE adapter) ----
parser.add_argument("--grad_checkpoint", help="use gradient checkpointing in encoders (saves memory)", action="store_true")
parser.add_argument("--n_routed", help="number of routed experts N_e", type=int, default=8)
parser.add_argument("--topk_expert", help="number of activated routed experts K", type=int, default=2)
parser.add_argument("--expert_dim", help="LoRA hidden dim r of each expert", type=int, default=64)
parser.add_argument("--expert_scale", help="LoRA-style output scaling for each expert (alpha/r)", type=float, default=1.0)
parser.add_argument("--expert_dropout", help="dropout inside each expert", type=float, default=0.0)
parser.add_argument("--warmup_proportion", help="warmup proportion for *_w_warmup schedulers", type=float, default=0.1)

# ---- EVA: semantic variant alignment ----
parser.add_argument("--t2i_use_logits", help="[exp] build L_s/L_o from raw comp logits instead of softmax probs", action="store_true")
parser.add_argument("--disable_text_moe", help="[diagnostic] turn off the text-encoder MoE adapters", action="store_true")
parser.add_argument("--tau_init", help="initial value of learnable tau_s / tau_o", type=float, default=1.0)
parser.add_argument("--alpha", help="intra/inter affinity balance in SVA", type=float, default=0.5)
parser.add_argument("--beta", help="inference fusion weight for primitives", type=float, default=0.5)
parser.add_argument("--lambda_1", help="weight of (Ls + Lo)", type=float, default=0.5)
parser.add_argument("--lambda_2", help="weight of (Lvs + Lvo)", type=float, default=0.1)
