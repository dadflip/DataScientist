"""Étape 0 — Installation des packages depuis la config."""
from __future__ import annotations
import importlib
import ipywidgets as widgets
from IPython.display import display, clear_output


def _pip_install(packages: list[str], output_widget) -> tuple[bool, list[str]]:
    failed = []
    try:
        from IPython import get_ipython
        _ipy = get_ipython()
    except Exception:
        _ipy = None
    for pkg in packages:
        with output_widget:
            print(f"   Installing {pkg}...")
        try:
            if _ipy:
                _ipy.run_line_magic("pip", f"install {pkg} -q")
            else:
                import subprocess, sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        except Exception as exc:
            failed.append(pkg)
            with output_widget:
                print(f"   [FAILED] {pkg} — {exc}")
    return len(failed) == 0, failed


def _check_imports(modules: list[str]) -> dict[str, bool]:
    result = {}
    for mod in modules:
        try:
            importlib.import_module(mod)
            result[mod] = True
        except ImportError:
            result[mod] = False
    return result


class InstallerUI:
    """Interface d'installation des packages depuis config[environment][packages]."""

    def __init__(self, config: dict):
        self.config = config
        self._groups = config.get("environment", {}).get("packages", {}).get("groups", [])
        self._checkboxes: dict = {}
        self._build_ui()

    def _build_ui(self) -> None:
        header = widgets.HTML(
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px;'>"
            "<span style='font-size:1.3em;font-weight:700;color:#6d28d9;'>Installer</span>"
            "<span style='color:#9ca3af;font-size:0.9em;'>Package Manager</span></div>"
            "<div style='font-size:0.85em;color:#64748b;margin-bottom:12px;'>"
            "Sélectionnez les groupes à installer puis cliquez sur <b>Install selected</b>.</div>"
        )

        table_header = widgets.HBox([
            widgets.HTML("<div style='width:30px;font-weight:bold;color:#475569;'>Sel</div>"),
            widgets.HTML("<div style='width:450px;font-weight:bold;color:#475569;'>Groupe / Description</div>"),
            widgets.HTML("<div style='width:90px;font-weight:bold;color:#475569;'>Action</div>"),
            widgets.HTML("<div style='width:160px;font-weight:bold;color:#475569;'>Statut</div>"),
        ], layout=widgets.Layout(border_bottom="2px solid #cbd5e1", padding="0 8px 5px 8px",
                                  margin="0 0 5px 0", align_items="center"))

        rows = []
        for i, grp in enumerate(self._groups):
            grp_id = grp.get("id", str(i))
            required = grp.get("required", False)
            default  = grp.get("default", False)
            cb = widgets.Checkbox(value=default or required, disabled=required,
                                   indent=False, layout=widgets.Layout(width="30px"))
            self._checkboxes[grp_id] = cb
            lbl = widgets.HTML(
                f"<div style='width:450px;'><span style='font-family:monospace;font-size:0.92em;'>"
                f"<b>{grp.get('label','')}</b></span></div>"
            )
            status_html = widgets.HTML(
                "<div style='width:160px;'><span style='color:#aaa;font-size:0.8em;'>not checked</span></div>"
            )
            check_btn = widgets.Button(description="Check", layout=widgets.Layout(width="80px", height="28px"))
            check_btn.on_click(self._make_checker(grp, status_html))
            bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
            rows.append(widgets.HBox(
                [cb, lbl, check_btn, status_html],
                layout=widgets.Layout(align_items="center", padding="6px 8px",
                                       border_bottom="1px solid #f1f5f9", background_color=bg)
            ))

        groups_box = widgets.VBox([table_header] + rows,
                                   layout=widgets.Layout(width="800px", border="1px solid #e2e8f0",
                                                          border_radius="6px"))

        sel_all = widgets.Button(description="Select all",   layout=widgets.Layout(width="max-content", padding="0 10px"))
        sel_def = widgets.Button(description="Defaults",     layout=widgets.Layout(width="max-content", padding="0 10px"))
        sel_non = widgets.Button(description="Deselect all", layout=widgets.Layout(width="max-content", padding="0 10px"))
        chk_all = widgets.Button(description="Check all", button_style="info",
                                  layout=widgets.Layout(width="max-content", padding="0 10px"))
        sel_all.on_click(lambda _: self._quick_select("all"))
        sel_def.on_click(lambda _: self._quick_select("default"))
        sel_non.on_click(lambda _: self._quick_select("none"))
        chk_all.on_click(self._check_all)
        quick_row = widgets.HBox([sel_all, sel_def, sel_non, chk_all],
                                  layout=widgets.Layout(margin="8px 0 4px 0", gap="6px"))

        self._install_btn = widgets.Button(description="Install selected", button_style="primary",
                                            layout=widgets.Layout(width="max-content", height="38px",
                                                                   margin="0 10px 0 0", padding="0 20px"))
        self._install_btn.on_click(self._on_install)
        self._progress = widgets.IntProgress(min=0, max=1, value=0, bar_style="info",
                                              layout=widgets.Layout(width="600px", visibility="hidden"))
        self._log = widgets.Output(layout=widgets.Layout(border="1px solid #ddd", max_height="280px",
                                                          overflow_y="auto", padding="6px", width="800px"))

        self.ui = widgets.VBox([
            header, groups_box, quick_row,
            widgets.HBox([self._install_btn, self._progress],
                          layout=widgets.Layout(align_items="center", gap="12px")),
            self._log,
        ], layout=widgets.Layout(width="100%", max_width="1000px", border="1px solid #e5e7eb",
                                  padding="18px", border_radius="10px", background_color="#ffffff"))

    def _make_checker(self, grp: dict, status_html: widgets.HTML):
        def _check(_):
            result = _check_imports(grp.get("check", []))
            ok_all = all(result.values())
            parts  = [f"[OK] {m}" if ok else f"[FAIL] {m}" for m, ok in result.items()]
            color  = "#2e7d32" if ok_all else "#c62828"
            status_html.value = (
                f"<span style='font-size:0.78em;color:{color};'>"
                f"{'&nbsp;&nbsp;'.join(parts)}</span>"
            )
        return _check

    def _check_all(self, _) -> None:
        with self._log:
            clear_output()
            print("Checking all groups...")
            for grp in self._groups:
                result = _check_imports(grp.get("check", []))
                ok_all = all(result.values())
                icon   = "[OK]" if ok_all else "[FAIL]"
                status = " | ".join(f"{'OK' if v else 'FAIL'} {k}" for k, v in result.items())
                print(f"  {icon} {grp.get('label',''):50s}  {status}")

    def _quick_select(self, mode: str) -> None:
        for grp in self._groups:
            grp_id   = grp.get("id", "")
            required = grp.get("required", False)
            cb = self._checkboxes.get(grp_id)
            if cb is None or required:
                continue
            if mode == "all":     cb.value = True
            elif mode == "none":  cb.value = False
            elif mode == "default": cb.value = grp.get("default", False)

    def _on_install(self, _) -> None:
        selected = [grp for grp in self._groups
                    if self._checkboxes.get(grp.get("id",""), widgets.Checkbox(value=False)).value
                    or grp.get("required", False)]
        total_pkgs = sum(len(g.get("packages", [])) for g in selected)
        self._progress.max   = max(total_pkgs, 1)
        self._progress.value = 0
        self._progress.layout.visibility = "visible"
        self._install_btn.disabled = True

        with self._log:
            clear_output()
            print(f"Installing {len(selected)} group(s) — {total_pkgs} package(s)...")
            print("=" * 60)
            overall_ok = True
            all_failed: list[str] = []

            for grp in selected:
                with self._log:
                    print(f"\n[GROUP] {grp.get('label','')}")
                ok, failed = _pip_install(grp.get("packages", []), self._log)
                self._progress.value += len(grp.get("packages", []))
                all_failed.extend(failed)
                if not ok:
                    overall_ok = False

            print("\n" + "=" * 60)
            print("Verifying imports...")
            any_fail = False
            for grp in selected:
                result = _check_imports(grp.get("check", []))
                ok_all = all(result.values())
                icon   = "[OK]" if ok_all else "[WARN]"
                mods   = " | ".join(f"{'OK' if v else 'FAIL'} {k}" for k, v in result.items())
                print(f"  {icon} {grp.get('id',''):20s}  {mods}")
                if not ok_all:
                    any_fail = True

            print("\n" + "=" * 60)
            if not any_fail and overall_ok:
                print("[SUCCESS] All packages installed and importable.")
            else:
                print("[WARNING] Some packages could not be verified.")
                if all_failed:
                    print(f"   Failed: {all_failed}")

        self._progress.bar_style = "success" if (overall_ok and not any_fail) else "danger"
        self._install_btn.disabled = False


def runner(config: dict) -> InstallerUI:
    ui = InstallerUI(config)
    display(ui.ui)
    return ui
