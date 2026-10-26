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
  quality: number;
  log: number;
}

interface State {
  games: Game[];
  settings: Settings;
  bundle: { missing: string[]; version: string };
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

const QUALITY_OPTIONS = [
  { data: -1, label: "Follow game" },
  { data: 0, label: "0 - large network (slow)" },
  { data: 1, label: "1" },
  { data: 2, label: "2" },
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
            label="NSS quality (AISS_NSS_QUALITY)"
            rgOptions={QUALITY_OPTIONS}
            selectedOption={s.quality}
            disabled={!!busy}
            onChange={(o) => changeSetting({ quality: o.data })}
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
