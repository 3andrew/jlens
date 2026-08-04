# Running estimation on RunPod

## Pod spec

- GPU: A100 80GB (~$1.6-2/hr) is the comfortable choice; L40S 48GB (~$1/hr)
  works with batch_size lowered to 4.
- Template: any official RunPod PyTorch image (CUDA torch preinstalled).
- Storage: container disk 30GB; volume disk 80GB mounted at /workspace
  (default). The volume holds the HF model cache (~19GB for the 9B) plus
  checkpoints (~1.6GB each at d~4096) and survives pod stop/start.

## Setup (once per pod)

```bash
cd /workspace
git clone https://github.com/3andrew/jlens.git
cd jlens
bash scripts/runpod_setup.sh
```

## Run (inside tmux — SSH drops must not kill the run)

```bash
tmux new -s jlens
python scripts/estimate_j.py configs/qwen35-9b-runpod.yaml 2>&1 | tee run.log
# detach: Ctrl-b d   reattach: tmux attach -t jlens
```

Resume after any interruption: same command + `--resume`.

## Validate + first light

```bash
python scripts/validate_j.py results/raw/estimate_j/qwen35-9b
python scripts/first_light.py configs/qwen35-9b-runpod.yaml \
  --prompt "Q: How many legs does the animal that spins webs have?
A: It has"
```

## Getting results off the pod

Raw checkpoints stay on the volume (gitignored anyway). Bring home the small
stuff: paste validate/first-light output into results/notes/, commit, push.
To copy a checkpoint locally:

```bash
rsync -avP root@<pod-ip>:/workspace/jlens/results/raw/estimate_j/qwen35-9b/ckpt_006000.pt \
  results/raw/estimate_j/qwen35-9b/
```

## Sizing notes

d_model ~4096 quadruples the probe demand vs the 0.8B (err ~ sqrt((d+1)/M)):
6000 prompts x 512 positions ~ 3M probes -> predicted source identity error
~ 0.037. Expect 1-2 prompts/s on A100 with fast-path kernels; the run is
roughly an hour, $2-4 of compute. git plumbing: the pod clones over HTTPS
(public repo, read-only); commit notes from your laptop, not the pod.
