"""Decky backend of "AISS - DLSS->NSS". The logic lives in py_modules/aiss_core.py; this file only adapts it to Decky."""

import asyncio
import os
import sys
from pathlib import Path

import decky

_PLUGIN_DIR = Path(getattr(decky, "DECKY_PLUGIN_DIR", os.path.dirname(os.path.abspath(__file__))))
if str(_PLUGIN_DIR / "py_modules") not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR / "py_modules"))

from aiss_core import AissCore  # noqa: E402

VERIFY_INTERVAL_S = 120


class Plugin:
    core: AissCore = None
    _task = None
    _lock = None

    def _core(self) -> AissCore:
        if Plugin.core is None:
            Plugin.core = AissCore(
                home=Path(decky.DECKY_USER_HOME),
                bin_dir=_PLUGIN_DIR / "bin",
                state_file=Path(decky.DECKY_PLUGIN_SETTINGS_DIR) / "state.json",
                logger=decky.logger,
            )
        return Plugin.core

    async def _run(self, fn, *args):
        """Serialise all filesystem work and keep it off the event loop."""
        if Plugin._lock is None:
            Plugin._lock = asyncio.Lock()
        async with Plugin._lock:
            return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    # ---- lifecycle -----------------------------------------------------------------------------------------------

    async def _main(self):
        decky.logger.info("AISS DLSS->NSS loaded")
        core = self._core()
        missing = core.bundle_missing()
        if missing:
            decky.logger.error("bundle incomplete, missing: " + ", ".join(missing))
        Plugin._task = asyncio.get_running_loop().create_task(self._verify_loop())

    async def _verify_loop(self):
        while True:
            try:
                await self._run(self._core().verify_all)
            except Exception as e:
                decky.logger.error(f"verify failed: {e}")
            await asyncio.sleep(VERIFY_INTERVAL_S)

    async def _unload(self):
        if Plugin._task:
            Plugin._task.cancel()
            Plugin._task = None
        decky.logger.info("AISS DLSS->NSS unloaded")

    async def _uninstall(self):
        # Files are removed; launch options can only be changed by the frontend, so disable games before uninstalling.
        core = self._core()
        for appid, gs in list(core.state["games"].items()):
            if gs.get("enabled"):
                core.uninstall_game(appid)
        core.remove_layer()
        core.save_state()

    # ---- frontend API --------------------------------------------------------------------------------------------

    async def get_state(self, rescan: bool = False) -> dict:
        core = self._core()
        games = await self._run(core.scan, bool(rescan))
        return {
            "games": games,
            "settings": core.state["settings"],
            "bundle": core.bundle_info(),
            "layer": core.state.get("layer", {}).get("files", {}),
        }

    async def enable_game(self, appid: str) -> dict:
        return await self._run(self._core().install_game, str(appid))

    async def disable_game(self, appid: str) -> dict:
        return await self._run(self._core().uninstall_game, str(appid))

    async def verify_now(self) -> list:
        return await self._run(self._core().verify_all)

    async def compute_launch_options(self, appid: str, current: str, enable: bool) -> dict:
        return await self._run(self._core().launch_options, str(appid), current or "", bool(enable))

    async def set_settings(self, settings: dict) -> dict:
        try:
            return {"ok": True, "settings": await self._run(self._core().set_settings, settings or {})}
        except (ValueError, TypeError) as e:
            return {"ok": False, "message": str(e), "settings": self._core().state["settings"]}

    async def request_capture(self) -> dict:
        return await self._run(self._core().request_capture)

    async def latest_capture(self, since: float = 0.0) -> dict | None:
        return await self._run(self._core().latest_capture, float(since or 0.0))

    async def game_stopped(self, appid: str) -> list:
        """Called by the frontend when a game exits: the prefix may now exist, deferred registry edits can run."""
        return await self._run(self._core().verify_all)
