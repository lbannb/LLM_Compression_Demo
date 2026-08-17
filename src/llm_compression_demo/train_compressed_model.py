import math, sys, time, torch

from pathlib import Path
from datetime import datetime, timedelta

from bitsandbytes.optim.adamw import AdamW8bit

from torch import nn
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer

from llm_compression_demo import model_loading as m
from llm_compression_demo.build_compressed_model import MPOLinear, build_compressed_model

# The paper heals on "generic chat datasets such as Ultrachat, Alpaca and Open-Hermes".
CHAT_CORPORA = (
    ("tatsu-lab/alpaca", None, "train", lambda r: r["text"]),
    ("HuggingFaceH4/ultrachat_200k", None, "train_sft", lambda r: "\n".join(t["content"] for t in r["messages"])),
    ("teknium/OpenHermes-2.5", None, "train", lambda r: "\n".join(t["value"] for t in r["conversations"])),
)

# wikitext-2 is the evaluation data set
EVAL_CORPUS = ("Salesforce/wikitext", "wikitext-2-raw-v1")
SEQ_LEN = 512


def _blocks(text: str, tok, seq_len: int, limit: int | None = None) -> torch.Tensor:
    """One long token stream chopped into fixed-length blocks."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // seq_len
    blocks = ids[: n * seq_len].view(n, seq_len)
    return blocks[:limit] if limit else blocks


def eval_blocks(tok, seq_len: int = SEQ_LEN, limit: int | None = None) -> torch.Tensor:
    return _blocks("\n\n".join(load_dataset(*EVAL_CORPUS, split="test")["text"]), tok, seq_len, limit)


def chat_blocks(tok, per_corpus: int = 4000, seq_len: int = SEQ_LEN) -> torch.Tensor:
    texts = []
    for repo, cfg, split, extract in CHAT_CORPORA:
        taken = 0
        for row in load_dataset(repo, cfg, split=split, streaming=True):
            texts.append(extract(row))
            taken += 1
            if taken >= per_corpus:
                break
        print(f"  {repo:32} {taken:>6,} samples", flush=True)
    return _blocks("\n\n".join(texts), tok, seq_len)


@torch.no_grad()
def perplexity(model, blocks: torch.Tensor, device: str = "cuda") -> float:
    """ 
        Returns the perplexity of the model on the given blocks of token ids. 
        ( see: https://en.wikipedia.org/wiki/Perplexity)
    """
    model.eval()
    total = 0.0
    for i in range(len(blocks)):
        ids = blocks[i : i + 1].to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            total += model(input_ids=ids, labels=ids).loss.item() # loss is the cross-entropy
    return math.exp(total / len(blocks))


def trainable_cores(model) -> list[nn.Parameter]:
    """
        Freezes everything, then promotes the MPO tensors to fp32 and unfreezes them.

        fp32 for the trainable ones is not optional: the cores sit around 1e-2, where the fp16
        spacing is 7.6e-6, while an update at lr=1e-4 is around 1e-6. In fp16 the addition is a
        no-op and the update is silently discarded. The frozen 1.7 B stay fp16 -- autocast runs the
        matmuls in fp16 either way, so mixing parameter dtypes costs nothing.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    cores = []
    for mod in model.modules():
        if isinstance(mod, MPOLinear):
            for i in range(len(mod.mpo_train)):
                mod.mpo_train[i] = nn.Parameter(mod.mpo_train[i].float(), requires_grad=True)
                cores.append(mod.mpo_train[i])
    return cores


def save_healed(model, out_dir: Path) -> int:
    """
        Writes the MPO tensors back out in the same layout build_compressed_model reads, so a healed
        run can be loaded with the same code path as a fresh compression. Cast back to fp16: the
        fp32 master copies only exist so the updates are not lost to rounding.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, MPOLinear):
            continue
        # model.layers.7.self_attn.q_proj -> layer_7_q_proj.pt
        parts = name.split(".")
        block, proj = parts[parts.index("layers") + 1], parts[-1]
        save_file({f"mpo_{k}": t.detach().half().contiguous() for k, t in enumerate(mod.mpo_train)},
                  out_dir / f"layer_{block}_{proj}.pt")
        written += 1
    return written


def heal(model, blocks: torch.Tensor, steps: int, lr: float = 1e-4, batch: int = 4, accum: int = 1,
         device: str = "cuda", log_every: int = 10,
         ckpt_dir: Path | None = None, ckpt_every: int = 500) -> list[float]:
    """
        Trains the MPO tensors only. Embeddings, lm_head, the norms and the excluded down_proj were
        never compressed, so they stay frozen.
    """
    cores = trainable_cores(model)

    # Without checkpointing every block's activations are retained across all 32 blocks and the
    # card is gone. enable_input_require_grads is needed because the embedding is frozen: otherwise
    # the checkpointed blocks receive no grad at all.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    assert model.is_gradient_checkpointing, "checkpointing did not take effect"

    # 8-bit Adam quantises only the momentum and variance, not weights or gradients. It is what
    # makes the fp32 master weights above affordable: 1.9 bytes/param against 8.
    opt = AdamW8bit(cores, lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    print(f"  trainable {sum(p.numel() for p in cores) / 1e6:.0f} M in {len(cores)} tensors, "
          f"gpu {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    losses, t0 = [], time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(accum):
            ids = blocks[torch.randint(len(blocks), (batch,))].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, labels=ids).loss / accum
            scaler.scale(loss).backward()
            running += loss.item()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(cores, 1.0)
        scaler.step(opt)
        scaler.update()

        losses.append(running)
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:5d}/{steps}  loss {running:7.4f}  ppl {math.exp(running):11.1f}"
                  f"  {(time.perf_counter() - t0) / (step + 1):5.2f}s/step"
                  f"  peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)

        if ckpt_dir is not None and step and step % ckpt_every == 0:
            save_healed(model, ckpt_dir)
            print(f"  checkpointed at step {step}", flush=True)

    return losses


if __name__ == "__main__":
    CHI_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/compressed_weights/schedule"
    STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    tok = AutoTokenizer.from_pretrained(m.REPO)
    print("healing corpora:", flush=True)
    train = chat_blocks(tok, per_corpus=8000)
    test = eval_blocks(tok, limit=20)
    print(f"  -> {len(train):,} train blocks, {len(test)} eval blocks of {SEQ_LEN} tokens\n", flush=True)

    model = build_compressed_model(CHI_DIR, dtype=torch.float16)
    print(f"gpu after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)

    before = perplexity(model, test)
    print(f"perplexity before healing: {before:,.1f}\n", flush=True)

    healed = Path("data/compressed_weights/healed")
    heal(model, train, steps=STEPS, ckpt_dir=healed, log_every=25)
    print(f"\nsaved {save_healed(model, healed)} tensors to {healed}", flush=True)

    after = perplexity(model, test)
    print(f"\nperplexity after healing:  {after:,.1f}   (was {before:,.1f})", flush=True)

    # small test 
    ids = tok("The main capital of germany is", return_tensors="pt").to("cuda")
    model.config.use_cache = True
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        out = model.generate(**ids, max_new_tokens=20, do_sample=False)  # type: ignore[attr-defined]
    print(f"output: {ascii(tok.decode(out[0], skip_special_tokens=True))}", flush=True)
