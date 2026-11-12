// Throwaway UI probe, not the product UI.
import { Probe } from "./rpc.ts";
import { appendFileSync } from "node:fs";
const log = (event, data) =>
  appendFileSync(
    process.env.VSR_PROBE_LOG,
    JSON.stringify({ event, data }) + "\n",
  );
export default {
  id: "vsr.gate0.tui",
  setup(ctx) {
    log("tui.loaded", { version: ctx.app.version });
    const [rpcState, update] = ctx.storage.memory("probe", {
      initial: { enabled: true, decisions: 0 },
    });
    const remote = ctx.client.rpc(Probe);
    const off = remote.events.on("updated", (e) => {
      log("tui.rpc.event", e.data);
      update((d) => Object.assign(d, e.data));
    });
    setTimeout(() => {
      void (async () => {
        const before = await remote.snapshot({});
        log("tui.rpc.before", before);
        log("tui.rpc.toggle", await remote.setEnabled({ enabled: false }));
        const after = await remote.snapshot({});
        log("tui.rpc.after", after);
        update((d) => Object.assign(d, after));
      })().catch((e) => log("tui.rpc.error", e));
    }, 2000);
    ctx.ui.slot({
      append: "session.composer.top",
      render: () => {
        log("tui.composer.render", {});
        return <text>VSR_COMPOSER_PROBE</text>;
      },
    });
    ctx.ui.slot({
      append: "home.footer",
      render: () => <text>VSR_FOOTER_PROBE</text>,
    });
    ctx.ui.slot({
      append: "session.panel",
      render: (panel) => {
        if (panel.name !== "vsr.probe") return null;
        log("tui.panel.render", { sessionID: panel.sessionID });
        return (
          <text>
            VSR_PANEL_PROBE {rpcState.enabled ? "VSR_RPC_ON" : "VSR_RPC_OFF"}{" "}
            {rpcState.decisions}
          </text>
        );
      },
    });
    setTimeout(() => {
      try {
        ctx.keymap.dispatch("vsr.probe");
      } catch (e) {
        log("tui.error", String(e));
      }
    }, 3000);
    ctx.keymap.layer(() => ({
      mode: "global",
      commands: [
        {
          id: "vsr.probe",
          title: "vSR probe",
          bind: "ctrl+g",
          slash: { name: "vsrprobe" },
          palette: true,
          run() {
            log("tui.command", { opened: ctx.ui.panel.open("vsr.probe") });
          },
        },
      ],
      bindings: ["vsr.probe"],
    }));
  },
};
