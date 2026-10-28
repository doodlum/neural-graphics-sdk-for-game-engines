Unmodified public headers of AMD's FidelityFX SDK 2.4.0 (MIT licence, see each file's header), except for the
relative include paths in `ffx_upscale.h`, `ffx_api_dx12.h` and `ffx_api_vk.h`. The FSR 4 path (`../fsr4_upscaler.cpp`) talks to AMD's `amdvkffx64.dll`
with these official type definitions; the NSS path keeps using the Arm fork in `ffx-api/`, whose surface format and
resource state values differ.
