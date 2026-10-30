# Steam Deck

This branch carries the SDK changes for NSS on the Steam Deck (RDNA2 / RADV): the black-line fix, dynamic resolution
and the GPU tuning documented in `docs/nss_gpu_tuning.md`.

The Steam Deck plugin, NGX proxy, loader shims and Vulkan layer that use it live in
[doodlum/DeckyUpscaler](https://github.com/doodlum/DeckyUpscaler), which includes this repository as its `nss` submodule.
