import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
import json
import argparse
from trl import GRPOConfig, GRPOTrainer
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer
from utils import *

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_math(split="train", chunk_size=1000, root='data/MATH') -> Dataset:
    problems, solutions = [], []
    for folder in os.listdir(os.path.join(root, split)):
        for file in os.listdir(os.path.join(root, split, folder)):
            if file.endswith('.json'):
                with open(os.path.join(root, split, folder, file), 'r') as f:
                    entry = json.load(f)
                problems.append(entry['problem'])
                solutions.append(entry['solution'])
    
    dataset = Dataset.from_dict({
        'problem': problems,
        'solution': solutions,
    })
    return dataset.map(process_math, batched=True, 
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    bias = args.action_bias
    exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-math-tarpo-group{args.group_size}"
            f"-lora{args.lora_rank}-temp{args.temperature}-len{args.max_prompt_length}-{args.max_completion_length}-bias{bias[0]}"
            f"-actemp{args.action_temperature}-topk{args.soft_top_k}-weight{args.action_loss_weight}"
            f"-seed{args.seed}"
            + (f"-alpha{args.action_kl_alpha}" if args.action_kl_alpha != 1.0 else "")
            + (f"-headlr{args.lr_action_head}" if args.lr_action_head != 1e-4 else "")
            + ("-frozenrouter" if args.freeze_router else ""))
    
    # Resume support: the newest checkpoint-* in the experiment directory is
    # picked up automatically, so a job killed by the Slurm wall clock can be
    # resubmitted unchanged. --resume never restores the original behaviour.
    resume_from = get_last_checkpoint(exp_name) if os.path.isdir(exp_name) else None

    if args.resume == "never":
        resume_from = None
    elif args.resume == "must" and resume_from is None:
        print(f"--resume must, but no checkpoint-* found in {exp_name}. Exiting...")
        exit()

    if resume_from is not None:
        print(f"Resuming training from {resume_from}")
    elif os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        exit()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_prompt_length + args.max_completion_length,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = False,
    )

    model.action_head.custom_init(bias=bias)
    print("🔧 Manually triggered action_head.custom_init(), overriding the default initialization!")

    model.answer_start = ANSWER_START

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save = [
            "action_head",
        ], 
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )

    if args.freeze_router:
        frozen = 0
        for pname, param in model.named_parameters():
            if "action_head" in pname:
                param.requires_grad_(False)
                frozen += param.numel()
        print(f"❄️  Router frozen: {frozen} action_head params excluded from the optimizer.")

    training_args = GRPOConfig(
        use_vllm = False,
        learning_rate = args.lr,
        beta = args.beta,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        optim = args.optimizer,
        max_grad_norm = args.max_grad_norm,
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        temperature = args.temperature,
        action_temperature = args.action_temperature,
        soft_top_k = args.soft_top_k,
        action_loss_weight = args.action_loss_weight,
        action_kl_alpha = args.action_kl_alpha,
        num_generations = args.group_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        per_device_train_batch_size = args.per_device_train_batch_size,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = 1,
        save_steps = 250,
        save_total_limit = 3,
        report_to = "wandb",
        output_dir = exp_name,
    )

    dataset = preprocess_math('train', chunk_size=500, root=args.dataset_root)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            get_reward_func(process_math_answer),
        ],
        args = training_args,
        train_dataset = dataset,
    )
    
    patch_trainer_optimizer(
        trainer,
        lr_action_head = args.lr_action_head,
    )
    trainer.train(resume_from_checkpoint=resume_from)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=32)

    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005)
    
    parser.add_argument("--lr_action_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--optimizer", type=str, default="paged_adamw_8bit")
    parser.add_argument("--max_grad_norm", type=float, default=0.1)

    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=1024)

    parser.add_argument("--action_temperature", type=float, default=1.0)
    parser.add_argument("--soft_top_k", type=int, default=10)
    parser.add_argument("--action_loss_weight", type=float, default=0.1)
    parser.add_argument("--action_kl_alpha", type=float, default=1.0,
                        help="alpha on the action-head KL (paper Eq. 6); effective weight is "
                             "beta*alpha. 0.0 = the paper's w/o-Action-KL ablation.")
    parser.add_argument("--freeze_router", action="store_true",
                        help="control: keep the action head at its initialisation by "
                             "disabling its gradient, leaving routing rate fixed.")

    parser.add_argument("--action_bias", type=float, nargs='+', default=None)

    parser.add_argument("--dataset_root", type=str, default="data/MATH")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--resume", type=str, default="auto",
                        choices=["auto", "never", "must"],
                        help="auto: continue from the newest checkpoint in the experiment "
                             "directory if there is one; never: refuse to touch an existing "
                             "experiment directory; must: fail unless a checkpoint is found")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)