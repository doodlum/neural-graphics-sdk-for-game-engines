import {
  ButtonItem,
  DropdownItem,
  Field,
  PanelSection,
  PanelSectionRow,
  ToggleField,
  staticClasses,
} from "@decky/ui";
import { callable, definePlugin, toaster } from "@decky/api";
import { useCallback, useEffect, useState } from "react";
import { FaMagic } from "react-icons/fa";

// ─── Backend API ───────────────────────────────────────────────────────────────

type Status = "installed" | "not_installed" | "conflict" | "prefix_missing" | "needs_repair";

interface Game {
  appid: string;
  name: string;
  install_path: string;
  shim: "dbghelp" | "dxgi" | null;
  exe: string | null;
  targets: string[];
  streamline: boolean;
  prefix: string | null;
  enabled: boolean;
  status: Status;
  message: string;
  launch_applied: boolean;
  reg_pending: boolean;
  conflicts: string[];
  last_run: { state: "active" | "loaded" | "error"; text: string; when: number } | null;
}

interface Settings {
  upscaler: number;    // 0 NSS, 3 XeSS, 4 FSR 4, 9 FSR 3.1
  nss_quality: number; // 0 Quality, 1 Balanced, 2 Performance
  fsr4_model: number;  // 0 Auto, 1 Lite, 2 INT8, 3 INT8 RDNA 2
  watermark: number;
  xess_quality: number;
  dilate_mv: number;
  gpu_timing: number;
  log: number;
}

interface State {
  games: Game[];
  settings: Settings;
  bundle: { missing: string[]; version: string; fsr4?: boolean; fsr3?: boolean };
}

interface OpResult {
  ok: boolean;
  status: string;
  message: string;
}

const getState = callable<[rescan: boolean], State>("get_state");
const enableGame = callable<[appid: string], OpResult>("enable_game");
const disableGame = callable<[appid: string], OpResult>("disable_game");
const verifyNow = callable<[], OpResult[]>("verify_now");
const computeLaunchOptions = callable<
  [appid: string, current: string, enable: boolean],
  { ok: boolean; options: string; changed: boolean; message?: string }
>("compute_launch_options");
const setSettings = callable<[settings: Partial<Settings>], { ok: boolean; message?: string; settings: Settings }>(
  "set_settings",
);
const gameStopped = callable<[appid: string], OpResult[]>("game_stopped");
const requestCapture = callable<[], { seq: number; dir: string }>("request_capture");
const latestCapture = callable<[since: number], { name: string; path: string; frames: number } | null>(
  "latest_capture",
);

// ─── SteamClient launch options ────────────────────────────────────────────────

declare const SteamClient: any;

const getLaunchOptions = (appid: string): Promise<string> =>
  new Promise((resolve, reject) => {
    if (typeof SteamClient === "undefined" || !SteamClient?.Apps?.RegisterForAppDetails) {
      reject(new Error("SteamClient unavailable"));
      return;
    }
    let done = false;
    let reg: { unregister: () => void } | null = null;
    const timer = window.setTimeout(() => {
      if (done) return;
      done = true;
      reg?.unregister();
      reject(new Error("timed out reading launch options"));
    }, 5000);
    reg = SteamClient.Apps.RegisterForAppDetails(parseInt(appid, 10), (details: any) => {
      if (done) return;
      done = true;
      window.clearTimeout(timer);
      // unregister after the callback returns
      window.setTimeout(() => reg?.unregister(), 0);
      resolve(details?.strLaunchOptions ?? "");
    });
  });

/** Merge (enable) or strip (disable) the plugin's launch options for one game. */
async function applyLaunchOptions(appid: string, enable: boolean): Promise<void> {
  const current = await getLaunchOptions(appid);
  const res = await computeLaunchOptions(appid, current, enable);
  if (!res.ok) throw new Error(res.message ?? "cannot compute launch options");
  if (res.options !== current) SteamClient.Apps.SetAppLaunchOptions(parseInt(appid, 10), res.options);
}

/** Bring every game's launch options in line with the backend state (on load and after settings change). */
async function syncAllLaunchOptions(games?: Game[]): Promise<void> {
  const list = games ?? (await getState(false)).games;
  for (const g of list) {
    try {
      if (g.enabled && g.shim && g.status !== "conflict") await applyLaunchOptions(g.appid, true);
      else if (!g.enabled && g.launch_applied) await applyLaunchOptions(g.appid, false);
    } catch (e) {
      console.error(`[AISS] launch options for ${g.appid}:`, e);
    }
  }
}

// ─── UI ────────────────────────────────────────────────────────────────────────

function lastRunLine(g: Game): string {
  if (!g.last_run) return g.enabled ? "Last run: not launched since setup (or the proxy did not load)" : "";
  const mins = Math.max(0, Math.round((Date.now() / 1000 - g.last_run.when) / 60));
  const ago = mins < 60 ? `${mins} min ago` : mins < 2880 ? `${Math.round(mins / 60)} h ago` : `${Math.round(mins / 1440)} d ago`;
  return `Last run (${ago}): ${g.last_run.text}`;
}

const STATUS_LABEL: Record<Status, string> = {
  installed: "Installed",
  not_installed: "Not installed",
  conflict: "Conflict",
  prefix_missing: "Prefix missing (launch once)",
  needs_repair: "Needs repair",
};

const UPSCALER_OPTIONS = [
  { data: 0, label: "NSS" },
  { data: 3, label: "XeSS" },
];
const FSR4_UPSCALER_OPTION = { data: 4, label: "FSR 4" };
const FSR3_UPSCALER_OPTION = { data: 9, label: "FSR 3.1" };
// NSS shader quality names from Arm's FfxApiNssShaderQualityMode (0 Quality, 1 Balanced, 2 Performance).
const NSS_QUALITY_OPTIONS = [
  { data: 0, label: "Quality" },
  { data: 1, label: "Balanced" },
  { data: 2, label: "Performance" },
];
const FSR4_MODEL_OPTIONS = [
  { data: 0, label: "Auto (AMD's choice)" },
  { data: 1, label: "Lite" },
  { data: 2, label: "INT8" },
  { data: 3, label: "INT8 RDNA 2" },
];
const XESS_QUALITY_OPTIONS = [
  { data: 0, label: "Auto (from render scale)" },
  { data: 100, label: "Ultra Performance" },
  { data: 101, label: "Performance" },
  { data: 102, label: "Balanced" },
  { data: 103, label: "Quality" },
  { data: 104, label: "Ultra Quality" },
  { data: 105, label: "Ultra Quality Plus" },
];
const LOG_OPTIONS = [
  { data: 0, label: "Off" },
  { data: 1, label: "Normal" },
  { data: 2, label: "Verbose" },
  { data: 3, label: "Trace" },
];

function Content() {
  const [state, setState] = useState<State | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(async (rescan = false) => {
    try {
      setState(await getState(rescan));
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh(false);
  }, []);

  const toggleGame = async (g: Game, enable: boolean) => {
    setBusy(g.appid);
    try {
      if (enable) {
        const r = await enableGame(g.appid);
        if (r.ok) {
          await applyLaunchOptions(g.appid, true);
          toaster.toast({ title: "AISS", body: `${g.name}: ${r.message || "enabled"}` });
        } else {
          toaster.toast({ title: "AISS - not enabled", body: `${g.name}: ${r.message}` });
        }
      } else {
        await applyLaunchOptions(g.appid, false);
        const r = await disableGame(g.appid);
        toaster.toast({ title: "AISS", body: `${g.name}: ${r.message || "disabled"}` });
      }
    } catch (e) {
      toaster.toast({ title: "AISS error", body: String(e) });
    }
    setBusy(null);
    await refresh(false);
  };

  const enableAll = async () => {
    if (!state) return;
    setBusy("all");
    let ok = 0;
    let failed = 0;
    for (const g of state.games.filter((x) => !x.enabled && x.status !== "conflict")) {
      try {
        const r = await enableGame(g.appid);
        if (r.ok) {
          await applyLaunchOptions(g.appid, true);
          ok++;
        } else failed++;
      } catch {
        failed++;
      }
    }
    toaster.toast({ title: "AISS", body: `Enabled ${ok} game(s)${failed ? `, ${failed} failed` : ""}` });
    setBusy(null);
    await refresh(false);
  };

  const disableAll = async () => {
    if (!state) return;
    setBusy("all");
    for (const g of state.games.filter((x) => x.enabled || x.launch_applied)) {
      try {
        await applyLaunchOptions(g.appid, false);
        await disableGame(g.appid);
      } catch (e) {
        console.error(e);
      }
    }
    setBusy(null);
    await refresh(false);
  };

  const changeSetting = async (patch: Partial<Settings>) => {
    setBusy("settings");
    const r = await setSettings(patch);
    if (!r.ok) toaster.toast({ title: "AISS", body: r.message ?? "invalid setting" });
    const s = await getState(false);
    setState(s);
    await syncAllLaunchOptions(s.games);
    setBusy(null);
  };

  const capture = async () => {
    setBusy("capture");
    const since = Date.now() / 1000 - 1;
    const r = await requestCapture();
    toaster.toast({ title: "AISS", body: "Capturing the next 8 frames of the running game..." });
    // The game picks the request up within half a second and writes the files a few frames later.
    await new Promise((res) => setTimeout(res, 4000));
    const c = await latestCapture(since);
    toaster.toast({
      title: "AISS",
      body: c && c.frames > 0 ? `Saved ${c.frames} frames: ${c.name}` : `No capture yet. Is a DLSS game running? (${r.dir})`,
    });
    setBusy(null);
  };

  const repair = async () => {
    setBusy("all");
    const results = await verifyNow();
    toaster.toast({ title: "AISS", body: results.length ? `Repaired ${results.length} game(s)` : "Everything in place" });
    await syncAllLaunchOptions();
    setBusy(null);
    await refresh(true);
  };

  if (!state) {
    return (
      <PanelSection>
        <PanelSectionRow>
          <Field label={error ? `Error: ${error}` : "Scanning Steam library..."} />
        </PanelSectionRow>
      </PanelSection>
    );
  }

  const s = state.settings;
  return (
    <>
      {state.bundle.missing.length > 0 && (
        <PanelSection title="Plugin bundle incomplete">
          <PanelSectionRow>
            <Field label="Missing" description={state.bundle.missing.join(", ")} />
          </PanelSectionRow>
        </PanelSection>
      )}
      <PanelSection title="Global options">
        <PanelSectionRow>
          <DropdownItem
            label="Upscaler"
            description={
              "Switches live in-game. NSS: Arm's neural upscaler. XeSS: Intel XeSS (dynamic resolution)." +
              (state.bundle.fsr4 ? " FSR 4: AMD's neural upscaler (D3D12, D3D11 and Vulkan games)." : "") +
              (state.bundle.fsr3 ? " FSR 3.1: AMD's non-neural upscaler, tuned to look as close to FSR 4 INT8 as it can." : "")
            }
            rgOptions={[
              ...UPSCALER_OPTIONS,
              ...(state.bundle.fsr4 ? [FSR4_UPSCALER_OPTION] : []),
              ...(state.bundle.fsr3 ? [FSR3_UPSCALER_OPTION] : []),
            ]}
            selectedOption={
              s.upscaler === 3 || (s.upscaler === 4 && state.bundle.fsr4) || (s.upscaler === 9 && state.bundle.fsr3)
                ? s.upscaler
                : 0
            }
            disabled={!!busy}
            onChange={(o) => changeSetting({ upscaler: o.data })}
          />
        </PanelSectionRow>
        {s.upscaler !== 3 && !(s.upscaler === 4 && state.bundle.fsr4) && !(s.upscaler === 9 && state.bundle.fsr3) && (
          <PanelSectionRow>
            <DropdownItem
              label="NSS quality"
              description="Quality: the full network (default). Balanced / Performance: smaller networks, faster, more ghosting."
              rgOptions={NSS_QUALITY_OPTIONS}
              selectedOption={s.nss_quality >= 0 && s.nss_quality <= 2 ? s.nss_quality : 0}
              disabled={!!busy}
              onChange={(o) => changeSetting({ nss_quality: o.data })}
            />
          </PanelSectionRow>
        )}
        {state.bundle.fsr4 && s.upscaler === 4 && (
          <>
            <PanelSectionRow>
              <DropdownItem
                label="FSR 4 model"
                description="Auto lets AMD choose (Lite on a Steam Deck). INT8 and INT8 RDNA 2 are the desktop networks: sharper, but much slower here."
                rgOptions={FSR4_MODEL_OPTIONS}
                selectedOption={s.fsr4_model}
                disabled={!!busy}
                onChange={(o) => changeSetting({ fsr4_model: o.data })}
              />
            </PanelSectionRow>
            <PanelSectionRow>
              <ToggleField
                label="FSR 4 watermark"
                description="Draw AMD's version/network banner on the output (proof of which model runs). Applies when the upscaler is next (re)built."
                checked={!!s.watermark}
                disabled={!!busy}
                onChange={(v) => changeSetting({ watermark: v ? 1 : 0 })}
              />
            </PanelSectionRow>
          </>
        )}
        {s.upscaler === 3 && (
          <PanelSectionRow>
            <DropdownItem
              label="XeSS preset"
              description="Input-range preset. Auto follows the render scale; the others fix the range."
              rgOptions={XESS_QUALITY_OPTIONS}
              selectedOption={s.xess_quality}
              disabled={!!busy}
              onChange={(o) => changeSetting({ xess_quality: o.data })}
            />
          </PanelSectionRow>
        )}
        {((s.upscaler === 4 && state.bundle.fsr4) || (s.upscaler === 9 && state.bundle.fsr3)) && (
          <PanelSectionRow>
            <ToggleField
              label="Dilate motion vectors"
              description="Before FSR runs, each pixel takes the motion of the nearest surface around it, so thin foreground detail (foliage, wires, edges) moves with its own motion. Can reduce shimmer and smearing on thin detail. Switches live."
              checked={!!s.dilate_mv}
              disabled={!!busy}
              onChange={(v) => changeSetting({ dilate_mv: v ? 1 : 0 })}
            />
          </PanelSectionRow>
        )}
        <PanelSectionRow>
          <ToggleField
            label="Log GPU time"
            description="Log the GPU milliseconds per upscale to aiss_nvngx.log (AISS_GPU_TIMING). Switches live."
            checked={!!s.gpu_timing}
            disabled={!!busy}
            onChange={(v) => changeSetting({ gpu_timing: v ? 1 : 0 })}
          />
        </PanelSectionRow>
        <PanelSectionRow>
          <DropdownItem
            label="Logging (AISS_LOG)"
            rgOptions={LOG_OPTIONS}
            selectedOption={s.log}
            disabled={!!busy}
            onChange={(o) => changeSetting({ log: o.data })}
          />
        </PanelSectionRow>
        <PanelSectionRow>
          <ButtonItem
            layout="below"
            disabled={!!busy}
            onClick={capture}
            description="Saves the running game's next 8 frames (colour, depth, motion vectors, output) for debugging"
          >
            {busy === "capture" ? "Capturing..." : "Capture frames"}
          </ButtonItem>
        </PanelSectionRow>
      </PanelSection>
      <PanelSection title={`DLSS games (${state.games.length})`}>
        <PanelSectionRow>
          <ButtonItem layout="below" disabled={!!busy} onClick={enableAll}>
            {busy === "all" ? "Working..." : "Enable all"}
          </ButtonItem>
        </PanelSectionRow>
        {state.games.length === 0 && (
          <PanelSectionRow>
            <Field label="No installed game ships nvngx_dlss.dll" />
          </PanelSectionRow>
        )}
        {state.games.map((g) => (
          <PanelSectionRow key={g.appid}>
            <ToggleField
              label={g.name}
              description={
                `${STATUS_LABEL[g.status] ?? g.status}` +
                (g.shim ? ` · ${g.shim}.dll` : "") +
                (g.message ? ` · ${g.message}` : "") +
                (lastRunLine(g) ? `\n${lastRunLine(g)}` : "")
              }
              checked={g.enabled}
              disabled={!!busy || (!g.enabled && g.status === "conflict")}
              onChange={(v) => toggleGame(g, v)}
            />
          </PanelSectionRow>
        ))}
      </PanelSection>
      <PanelSection title="Maintenance">
        <PanelSectionRow>
          <ButtonItem layout="below" disabled={!!busy} onClick={() => refresh(true)}>
            Rescan library
          </ButtonItem>
        </PanelSectionRow>
        <PanelSectionRow>
          <ButtonItem layout="below" disabled={!!busy} onClick={repair}>
            Verify / repair now
          </ButtonItem>
        </PanelSectionRow>
        <PanelSectionRow>
          <ButtonItem layout="below" disabled={!!busy} onClick={disableAll}>
            Disable all
          </ButtonItem>
        </PanelSectionRow>
        {state.bundle.version && (
          <PanelSectionRow>
            <Field label="Bundle" description={state.bundle.version} />
          </PanelSectionRow>
        )}
      </PanelSection>
    </>
  );
}

export default definePlugin(() => {
  // Steam may have reset launch options (or the plugin was updated): reconcile them once Steam is up.
  const startup = window.setTimeout(() => {
    syncAllLaunchOptions().catch((e) => console.error("[AISS] startup sync failed", e));
  }, 15000);

  // When a game exits its prefix exists and the registry can be edited safely: let the backend finish setup.
  let lifetime: { unregister: () => void } | null = null;
  try {
    lifetime = SteamClient?.GameSessions?.RegisterForAppLifetimeNotifications?.((n: any) => {
      if (n && !n.bRunning && n.unAppID) gameStopped(String(n.unAppID)).catch(() => {});
    });
  } catch (e) {
    console.error("[AISS] lifetime hook failed", e);
  }

  return {
    name: "AISS DLSS-NSS",
    titleView: <div className={staticClasses.Title}>AISS – DLSS→NSS</div>,
    content: <Content />,
    icon: <FaMagic />,
    onDismount() {
      window.clearTimeout(startup);
      lifetime?.unregister();
    },
  };
});
