# Watermark Forgery Attack — atml_team072

Team: Umair Ayaz Aslam (7075924), Shreya Kolhapure (7082775)

This repository reproduces our best leaderboard submission. Each of the 8 clean
target batches receives the watermark of the corresponding source group:
`WM_1→1-25, WM_2→26-50, WM_3→51-75, WM_4→76-100, WM_5→101-125, WM_6→126-150,
WM_7→151-175, WM_8→176-200`.


> **Note on provided files.** `task_template.py` is the instructors' baseline
> (applies one copy attack to all groups; scores ~0.333) and `submission.py` is the
> provided leaderboard submitter. Our actual method lives in `build_submission.py`
> (scheme-aware native re-embeds + copy attack) and `wmcopier/` (no-box diffusion
> forgery for WM_3). These are what reproduce our best result below.

## Setup
```bash
pip install invisible-watermark trustmark opencv-python numpy scipy pillow
# Place the provided dataset at ./Dataset  (Dataset/clean_targets, Dataset/watermarked_sources)
```

## Reproduce the best result

### Step 1 — base 200 images (WM_1,2,7 native + WM_4,5,6 copy attack)
```bash
python build_submission.py         # writes submission_temp/*.png and submission.zip
```
This performs, per group:
- **WM_1 = DwtDct** — decode the 25 sources with `invisible-watermark` dwtDct at
  message length **48** (length found by aliasing-signature matching), majority-vote
  the message, re-embed into targets 1-25.
- **WM_2 = RivaGAN** — decode sources at rivaGan (32-bit), re-embed into 26-50.
- **WM_4/5/6 = averaging (copy) attack** — extract the shared watermark as the
  per-pixel median of high-pass residuals (WM_4 gaussian-hp σ=0.5, WM_5 median-hp 3,
  WM_6 median-hp 5), add to each target with projection-matched strength ×
  `GLOBAL_SCALE=2.0`, onto 76-150. (WM_5 corresponds to HiDDeN at 128², WM_4/WM_6
  are additive learned marks.)
- **WM_7 = TrustMark (variant Q)** — decode the secret `DLSdKpw|`, re-embed into 151-175.

### Step 2 — WM_3 via WMCopier (no-box diffusion forgery)
Requires a GPU (we used Kaggle/Colab). Clone WMCopier and apply the one-line
trainer fix, then:
```bash
# train an unconditional DDIM model on WM_3's 25 sources (256px)
bash wmcopier/train_cmd.sh          # ~1000 steps is sufficient; checkpoint in experiments/wm3
python wmcopier/forge.py --group WM_3 --res 256 --alpha 0.35 \
       --ckpt experiments/wm3/model-*.pt --targets 51 75
```
Forge = shallow DDIM inversion (end_step 20) + refined denoising, blended with the
clean image as `x + 0.35*(rec - x)`.

### Step 3 — package
```bash
python wmcopier/package.py          # copies the 200 base images, swaps in forged WM_3, writes submission.zip
```

Submit `submission.zip` with the provided `submission.py`:
```bash
# edit submission.py: set API_KEY, FILE_PATH=submission.zip, SUBMIT=True
python submission.py
```

`task_template.py` (provided) documents the official target mapping
(WM_i -> clean image batches) and the baseline averaging attack.

## Best public score: 0.632  (WM_1,2,3,4,5,6,7 forged; WM_8 attempted)

## Repository layout
- `build_submission.py` — Step 1 (native re-embeds + copy attack).
- `wmcopier/` — training command, forge, and packaging for the no-box attack (WM_3, WM_8).
- `identify/` — scheme-identification experiments referenced in the report
  (each decodes the 25 sources and checks message consistency vs a clean baseline):
  `try_rivagan.py`, `try_trustmark.py`, `try_vine.py`, `try_stablesig.py`,
  `try_wam.py`, and `crack_dwtdct_length.py` (the dwtDct length crack).
