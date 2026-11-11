# Immich Cull

Did you ever need to quickly needed to sort through about 3000 pictures from your last vacation? Yes? Then you are like me and this tool is probably for you.

Especially if you take 10 pictures that look all almost identical.

No internet connection needed. 100% local evaluation. This was important to me, because I don't want to send all my pictures to OpenAI or Anthropic.

Immich is my photo db of choice, so it uses their API to get the pictures/view them and delete them. It can however also be used without Immich.

## Setup

1. In Immich, open Account Settings → API Keys. Create a key with the permissions `asset.read`, `asset.view` and `asset.delete`. Add `library.read` (admin account only) if you want to choose between your external libraries in the app; without it, only the `path_prefix` from the config is offered.
2. Move `config.example.toml` to `config.toml` and paste api key into there.
3. Start it:

   ```bash
   uv run python -m cull
   ```

4. Open http://127.0.0.1:8765 and you are good to go.

The first start downloads the model weights (about 7 GB in total, most of it the vision model, which only loads when you first press "Ask vision model").

## How the decision process looks like:

- **Grouping:** photos taken within 2 minutes of each other whose DINOv2 embeddings are at least 0.85 alike (`MAX_GAP`, `MIN_SIM` in `cull/grouping.py`).
- **Sharpness:** Tenengrad gradient energy on a lightly smoothed image, taken from the sharpest tiles. Unlike Laplacian variance it is barely affected by sensor noise. When every photo in a group has a face, sharpness is measured on the largest face.
  - Why Sharpness? If you take a lot of images in a row or a burst, some of them are usually blurry. You typically don't want them in your image database. Chang et al [^chang2016] also comes to the conclusion that sharpness matters most for people if they evaluate near identical shots.
- **Aesthics:** LAION aesthetic score is used for this
  - In the beginning I used the CLIP+MLP Aesthetic Score Predictor from Schuhmann [^schuhmann2022], but that score is not very good for my use case. It's based on CLIP and thus far better on evaluating between different pictures not on a set of very similar pictures.
- **Score (0–100, relative within the group):** TODO: 85% sharpness, 15% LAION aesthetic score, minus penalties for closed eyes and blown highlights.
- **Vision model (optional):** Qwen3-VL 8B via MLX compares the photos and names its own pick with reasons; its pick gets a second, blue frame. It does not change the score.

**End decision:** Entirely done by you. You have the final say, nothing gets deleted without you.

Analysis results are cached in `data/cache.sqlite`, so re-scanning a range is fast.

## Known limitations

I currently disagree with the scoring of the vision and the aesthetics model a lot, but they help me make a decision. I would however not currently rely on their decision-making process. I will try to improve that in the future.

## Use without Immich

```bash
uv run python -m cull --folder testdata/photos
```

In this mode "trash" moves files into a `_trash` subfolder.

## Use of AI

This was a quick side project for me, and it was heavily vibecoded by me with Claude. This README is 100% human and all code was reviewed by me.


## Citations

[^chang2016]: H. Chang, F. Yu, J. Wang, D. Ashley, and A. Finkelstein, “Automatic triage for a photo series,” ACM Trans. Graph., vol. 35, no. 4, p. 148:1-148:10, Jul. 2016, doi: 10.1145/2897824.2925908.
[^schuhmann2022]: C. Schuhmann, “christophschuhmann/improved-aesthetic-predictor: CLIP+MLP Aesthetic Score Predictor.” Accessed: Nov. 11, 2026. [Online]. Available: https://github.com/christophschuhmann/improved-aesthetic-predictor

