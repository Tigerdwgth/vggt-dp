# third_party patches

The MetaWorld checkout used by this project differs from the stock
[DP3 MetaWorld fork](https://github.com/YanjieZe/Metaworld) in three files. Copy
them over your checkout after installing MetaWorld:

```bash
cp -r third_party_patches/Metaworld/. third_party/Metaworld/
```

| File | Why |
|---|---|
| `metaworld/envs/assets_v2/objects/assets/xyz_base_vis.xml` | Robot arm and table props rendered opaque (alpha=1). Selected when `eval_robovis=true`. |
| `metaworld/envs/assets_v2/objects/assets/xyz_base_invis.xml` | Same geometry with alpha=0, so the arm is invisible to the cameras while collision stays identical. Selected when `eval_robovis=false`. |
| `gen_demonstration_expert.py` | Adds `--robovis` so demonstrations can be rendered with or without the visible arm. |

`train.py` copies the vis or invis variant over
`metaworld/envs/assets_v2/objects/assets/xyz_base.xml` at startup according to
`eval_robovis`, so that file is overwritten on every run and does not need to be
patched by hand.
