# Environment integrations

TraceRigor isolates optional environment imports so one missing dependency does
not prevent the registry from importing. Inspect the current installation:

```bash
tracerigor envs --verbose
```

The `envs` extra installs the common Gym/Gymnasium, Pillow, and Sokoban
dependencies. Larger integrations remain opt-in:

- SVG may require `beautifulsoup4`, `svgpathtools`, CairoSVG, and system Cairo;
- Navigation may require AI2-THOR and a display/Vulkan-capable runtime;
- PrimitiveSkill may require ManiSkill and separately downloaded assets;
- ALFWorld requires its package and data download;
- ScienceWorld requires its package and Java environment;
- BabyAI Text requires its upstream wrapper stack.

Follow each upstream project's installation and licence instructions. Avoid
pinning an unrelated environment to an incompatible global dependency set; use
separate virtual environments when integrations require conflicting versions.

For the implementation contract and registration process, see
`docs/envs/create-env.md` and `docs/envs/create-service.md`.
