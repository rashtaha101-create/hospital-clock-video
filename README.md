# Hospital Clock - animated explainer short

2D cartoon characters in a 3D-looking (2.5D) diorama world, with a constantly moving camera, word-by-word captions
and whip/push/dissolve transitions. Rendered in Google Colab; the code lives here.

## Layout
- `src/characters.py` - procedural 2D characters (expressions, poses)
- `src/engine.py` - camera, depth-of-field, VFX, transitions, captions, SFX, export
- `stories/hospital_clock.py` - script, background prompts and the 17-shot list (anchored to word indices)
- `notebooks/colab_render.ipynb` - the Colab runner (clones this repo)

## Run
1. Push this folder to your GitHub repo (see below).
2. Open `notebooks/colab_render.ipynb` in Colab:
   `https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/colab_render.ipynb`
3. Runtime -> T4 GPU. Set `REPO_URL` in the settings cell, run all cells, upload the voice-over.

## Render frame by frame - results are committed to the repo
1. Commit your voice-over as `audio/voiceover.mp3`.
2. GitHub -> **Actions** -> **Render video (frame by frame, committed to the repo)** -> **Run workflow**
   (`scale` 0.5 preview, 1.0 final). The repo needs *Settings -> Actions -> General -> Workflow permissions -> Read and write*.
3. Three stages:
   - **prep** - Whisper word timing, backgrounds + depth maps, `plan.json`; resets the branch `renders-<story>` to an empty commit
   - **frames** - one runner per shot (17 in parallel); each commits its numbered frames to `renders/<story>/frames/` on that branch
     (parallel pushes retry automatically)
   - **assemble** - reads every frame from the repo, burns captions, mixes voice + SFX, commits `renders/<story>/<story>_<scale>.mp4`
4. Browse/download from the `renders-<story>` branch (frames + video). The video is also attached as a workflow artifact.

Size: about 0.1 GB of frames at scale 0.5 and about 0.35 GB at 1.0 (JPEG q95). Each run **replaces** the render branch, so old frames
don't pile up in its history. GitHub recommends repos stay under 1 GB, so keep the render branch out of your main branch
(it is separate) and use `--frame-ext jpg` (default).

Same thing locally: `python src/render.py --stage prep|frames|assemble ...` (`--shot 5,11` renders only those shots).

Notes
- Everything is CPU. Backgrounds on CPU use SDXL-Turbo (softer). For the best look generate them once in Colab (GPU) and commit
  `assets/<story>/backgrounds/*.png` + `depth/*.npy`; the workflow then skips generation.
- Use a **public repo** (4 vCPU / 16 GB runners, free parallel jobs).
- `regen` re-rolls named backgrounds.

## New story
Copy `stories/hospital_clock.py` to `stories/my_story.py`, edit `SCRIPT`, `BACKGROUNDS`, `SHOTS`, push, and set `STORY = 'my_story'` in the notebook.

## Push
```
git init && git add . && git commit -m "initial"
git branch -M main
git remote add origin https://github.com/<USER>/<REPO>.git
git push -u origin main
```
Note: GitHub Actions has no GPU, so image generation can't run there; rendering stays in Colab.
