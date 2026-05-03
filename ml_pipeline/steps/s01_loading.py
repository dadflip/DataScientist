"""Étape 1 — Chargement des données (DataLoaderUI)."""
from __future__ import annotations
import io, os, traceback, sqlite3, tempfile, zipfile, urllib.request
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore
try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore
try:
    import rdflib
except ImportError:
    rdflib = None  # type: ignore

from ml_pipeline.styles import styles


def _load_sklearn_dataset(name: str) -> pd.DataFrame:
    from sklearn import datasets as _ds
    loader = getattr(_ds, f"load_{name}", None) or getattr(_ds, f"fetch_{name}", None)
    if not loader:
        raise ValueError(f"Dataset '{name}' not found in sklearn.datasets")
    bunch = loader()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    if hasattr(bunch, "target"):
        df["target"] = bunch.target
    return df


class DataLoaderUI:
    """Interface de chargement multi-sources depuis la config."""

    def __init__(self, state):
        self.state  = state
        self.config = state.config.get("loading", {})
        self.supported_types = self.config.get("supported_types", {})
        self.modes           = self.config.get("modes", [])
        self._adv_widgets: dict = {}
        self._build_ui()

    # ── construction UI ───────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        modes_opts = [m["label"] for m in self.modes] if self.modes else ["Single file"]
        self.mode_dd   = widgets.Dropdown(options=modes_opts, value=modes_opts[0],
                                           description="Mode:", layout=styles.LAYOUT_DD_LONG)
        sources_opts   = list(self.supported_types.keys())
        self.source_dd = widgets.Dropdown(options=sources_opts, description="Source type:",
                                           layout=styles.LAYOUT_DD_LONG)
        self.slots_box = widgets.VBox([])
        self.adv_config_box = widgets.Accordion(children=[widgets.VBox([])])
        self.adv_config_box.set_title(0, "Advanced Configurations")
        self.adv_config_box.selected_index = None

        self.btn_load    = widgets.Button(description="Load data",  button_style=styles.BTN_PRIMARY,
                                           layout=styles.LAYOUT_BTN_STD)
        self.btn_preview = widgets.Button(description="Preview",    button_style=styles.BTN_INFO,
                                           layout=styles.LAYOUT_BTN_STD)
        self.clear_cb    = widgets.Checkbox(value=True, description="Clear existing data",
                                             layout=styles.LAYOUT_BTN_LARGE)
        self.out = widgets.Output()

        self.mode_dd.observe(self._on_mode_change,   names="value")
        self.source_dd.observe(self._on_source_change, names="value")
        self.btn_load.on_click(self._on_load)
        self.btn_preview.on_click(self._on_preview)

        self._build_slots()
        self._build_adv_config()

        header_html = styles.card_html("Data Loader", "Import vos données depuis plusieurs sources", "")
        self.ui = widgets.VBox([
            widgets.HTML(value=header_html),
            widgets.HBox([self.mode_dd, self.source_dd]),
            self.slots_box,
            self.adv_config_box,
            widgets.HBox([self.btn_load, self.btn_preview, self.clear_cb]),
            self.out,
        ], layout=styles.LAYOUT_BOX)

    def _get_current_mode_id(self) -> str:
        for m in self.modes:
            if m.get("label") == self.mode_dd.value:
                return m.get("value", "single")
        return "single"

    def _build_slots(self) -> None:
        mode_id  = self._get_current_mode_id()
        slots    = ["Data"]
        for m in self.modes:
            if m.get("value") == mode_id:
                slots = m.get("slots", ["Data"])
                break
        src_type    = self.supported_types.get(self.source_dd.value, "")
        is_sklearn  = src_type == "sklearn"
        is_clipboard = src_type == "clipboard"
        options     = self.config.get("sklearn_datasets", []) if is_sklearn else []
        accept_exts = self.config.get("file_accepts", {}).get(src_type, "")
        children = []
        for s in slots:
            lbl = widgets.Label(f"{s}:", layout=widgets.Layout(width="100px"))
            if is_sklearn:
                inp = widgets.Dropdown(options=options, layout=styles.LAYOUT_DD)
                children.append(widgets.HBox([lbl, inp]))
            elif is_clipboard:
                inp = widgets.Textarea(placeholder="Paste CSV text here...",
                                        layout=widgets.Layout(width="400px", height="100px"))
                children.append(widgets.HBox([lbl, inp]))
            else:
                inp_txt  = widgets.Text(placeholder="URL or Local File Path", layout=styles.LAYOUT_TEXT)
                inp_file = widgets.FileUpload(accept=accept_exts, multiple=False,
                                               layout=widgets.Layout(width="max-content"))
                children.append(widgets.HBox([lbl, inp_txt, widgets.Label(" OR "), inp_file],
                                              layout=widgets.Layout(align_items="center", gap="10px",
                                                                     margin="4px 0")))
        self.slots_box.children = children

    def _build_adv_config(self) -> None:
        src_type  = self.supported_types.get(self.source_dd.value, "")
        mode_id   = self._get_current_mode_id()
        self._adv_widgets.clear()
        adv_list  = self.config.get("adv_configs", {}).get(src_type, [])
        mode_list = self.config.get("mode_configs", {}).get(mode_id, [])
        combined  = adv_list + mode_list
        children  = []
        for conf in combined:
            w_type = conf.get("type")
            w_id   = conf["id"]
            help_t = conf.get("help", "")
            if w_type == "text":
                w = widgets.Text(value=str(conf.get("value", "")),
                                  placeholder=str(conf.get("placeholder", "")),
                                  description=conf.get("description", ""),
                                  layout=styles.LAYOUT_TEXT,
                                  style={"description_width": "initial"})
            elif w_type == "dropdown":
                opts = conf.get("options", [])
                if "options_key" in conf:
                    opts = self.config.get(conf["options_key"], [])
                w = widgets.Dropdown(options=opts, value=conf.get("value", ""),
                                      description=conf.get("description", ""),
                                      layout=styles.LAYOUT_TEXT,
                                      style={"description_width": "initial"})
            elif w_type == "checkbox":
                w = widgets.Checkbox(value=conf.get("value", False),
                                      description=conf.get("description", ""),
                                      layout=widgets.Layout(width="auto"),
                                      style={"description_width": "initial"}, indent=False)
            elif w_type == "floatslider":
                w = widgets.FloatSlider(value=conf.get("value", 1.0),
                                         min=conf.get("min", 0.0), max=conf.get("max", 1.0),
                                         step=conf.get("step", 0.1),
                                         description=conf.get("description", ""),
                                         layout=widgets.Layout(width="350px"),
                                         style={"description_width": "initial"},
                                         readout_format=".2f")
            else:
                continue
            self._adv_widgets[w_id] = w
            if help_t:
                icon = widgets.HTML(
                    f"<span title='{help_t}' style='cursor:help;color:#3b82f6;margin-left:4px;'>&#9432;</span>"
                )
                children.append(widgets.HBox([w, icon], layout=widgets.Layout(align_items="center")))
            else:
                children.append(w)
        if children:
            grid = widgets.GridBox(children, layout=widgets.Layout(
                grid_template_columns="repeat(auto-fit,minmax(320px,1fr))",
                gap="10px", align_items="center"))
            self.adv_config_box.children = [grid]
            self.adv_config_box.layout.display = "block"
        else:
            self.adv_config_box.children = [
                widgets.HTML("<i>No advanced configuration needed for this type/mode.</i>")]
            self.adv_config_box.layout.display = "none"

    def _on_mode_change(self, _) -> None:
        self._build_slots(); self._build_adv_config()

    def _on_source_change(self, _) -> None:
        self._build_slots(); self._build_adv_config()

    # ── chargement ────────────────────────────────────────────────────────────
    def _apply_common_post_load(self, df):
        if not isinstance(df, pd.DataFrame):
            return df
        usecols_w = self._adv_widgets.get("usecols")
        if usecols_w and usecols_w.value.strip():
            cols = [x.strip() for x in usecols_w.value.split(",")]
            selected = []
            for c in cols:
                try:
                    idx = int(c)
                    if idx < len(df.columns):
                        selected.append(df.columns[idx])
                except ValueError:
                    if c in df.columns:
                        selected.append(c)
            if selected:
                df = df[selected]
        idx_w = self._adv_widgets.get("index_col")
        if idx_w and idx_w.value.strip():
            val = idx_w.value.strip()
            try:
                vi = int(val)
                if vi < len(df.columns):
                    df = df.set_index(df.columns[vi])
            except ValueError:
                if val in df.columns:
                    df = df.set_index(val)
        pdates_w = self._adv_widgets.get("parse_dates")
        if pdates_w and pdates_w.value:
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except Exception:
                        pass
        sample_w = self._adv_widgets.get("sample_frac")
        if sample_w:
            try:
                frac = float(sample_w.value)
                if 0.0 < frac < 1.0:
                    df = df.sample(frac=frac, random_state=42)
            except Exception:
                pass
        return df

    def _load_csv(self, path: str, src_type: str = "csv") -> pd.DataFrame:
        header_w = self._adv_widgets.get("header")
        sep_w    = self._adv_widgets.get("sep")
        enc_w    = self._adv_widgets.get("enc")
        skiprows_w = self._adv_widgets.get("skiprows")
        nrows_w    = self._adv_widgets.get("nrows")
        header_val = None if (header_w and header_w.value == "None") else (
            0 if (header_w and header_w.value == "0") else "infer")
        sep_val = sep_w.value if sep_w else ("," if src_type == "csv" else "\t")
        enc_val = enc_w.value if enc_w else "utf-8"
        kwargs: dict = {"sep": sep_val, "encoding": enc_val, "header": header_val}
        if skiprows_w and skiprows_w.value.strip():
            try: kwargs["skiprows"] = int(skiprows_w.value.strip())
            except Exception: pass
        if nrows_w and nrows_w.value.strip():
            try: kwargs["nrows"] = int(nrows_w.value.strip())
            except Exception: pass
        if src_type == "clipboard":
            return pd.read_csv(io.StringIO(path), **kwargs)
        if str(path).startswith("http"):
            req = urllib.request.Request(path, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                return pd.read_csv(resp, **kwargs)
        return pd.read_csv(path, **kwargs)

    def _on_load(self, _) -> None:
        with self.out:
            clear_output()
            if self.clear_cb.value:
                self.state.data_raw.clear()
                self.state.data_types.clear()
            src_type = self.supported_types.get(self.source_dd.value, "")
            loaded: list[str] = []
            for child in self.slots_box.children:
                try:
                    slot_name = child.children[0].value.strip(":")
                    path = ""
                    uploaded_file = None
                    if len(child.children) == 2:
                        path = child.children[1].value
                    else:
                        inp_txt  = child.children[1]
                        inp_file = child.children[3]
                        if inp_file.value:
                            uploaded_file = (
                                inp_file.value[0] if isinstance(inp_file.value, (list, tuple))
                                else list(inp_file.value.values())[0]
                                if isinstance(inp_file.value, dict) else None
                            )
                        if not uploaded_file:
                            path = inp_txt.value
                    if isinstance(path, str):
                        path = path.strip()
                    if not path and not uploaded_file:
                        continue
                    print(f"Loading {slot_name}...")
                    if uploaded_file:
                        ext = os.path.splitext(uploaded_file["name"])[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(uploaded_file["content"])
                            path = tmp.name
                    # ── dispatch par type ─────────────────────────────────────
                    if src_type == "sklearn":
                        df = _load_sklearn_dataset(path)
                    elif src_type == "clipboard":
                        df = self._load_csv(path, src_type)
                    elif src_type == "sqlite":
                        tbl_w = self._adv_widgets.get("table")
                        tbl   = tbl_w.value.strip() if tbl_w else ""
                        with sqlite3.connect(path) as conn:
                            if tbl:
                                df = pd.read_sql(f"SELECT * FROM [{tbl}]", conn)
                            else:
                                tables = pd.read_sql(
                                    "SELECT name FROM sqlite_master WHERE type='table'", conn
                                )["name"].tolist()
                                if not tables:
                                    raise ValueError("No tables found in SQLite database.")
                                df = pd.read_sql(f"SELECT * FROM [{tables[0]}]", conn)
                    elif src_type == "json":
                        orient_w = self._adv_widgets.get("orient")
                        lines_w  = self._adv_widgets.get("lines")
                        df = pd.read_json(path,
                                           orient=orient_w.value if orient_w else "columns",
                                           lines=lines_w.value if lines_w else False)
                    elif src_type in ("excel", "xls", "xlsx"):
                        sheet_w = self._adv_widgets.get("sheet")
                        sheet   = sheet_w.value if sheet_w else 0
                        try: sheet = int(sheet)
                        except ValueError: pass
                        header_w = self._adv_widgets.get("header")
                        hdr = None if (header_w and header_w.value == "None") else 0
                        kw: dict = {"sheet_name": sheet, "header": hdr}
                        if str(path).startswith("http"):
                            req = urllib.request.Request(path, headers={"User-Agent": "Mozilla/5.0"})
                            with urllib.request.urlopen(req) as resp:
                                df = pd.read_excel(resp.read(), **kw)
                        else:
                            df = pd.read_excel(path, **kw)
                    elif src_type == "parquet":
                        df = pd.read_parquet(path)
                    elif src_type == "feather":
                        df = pd.read_feather(path)
                    elif src_type == "hdf5":
                        df = pd.read_hdf(path)
                    elif src_type == "orc":
                        df = pd.read_orc(path)
                    elif src_type == "text":
                        enc_w = self._adv_widgets.get("enc")
                        with open(path, "r", encoding=enc_w.value if enc_w else "utf-8") as f:
                            df = f.read()
                    elif src_type == "image":
                        if Image is None:
                            raise ImportError("Pillow not installed")
                        mode_w   = self._adv_widgets.get("mode")
                        resize_w = self._adv_widgets.get("resize")
                        df = Image.open(path)
                        if mode_w and mode_w.value:
                            df = df.convert(mode_w.value)
                        if resize_w and resize_w.value.strip():
                            try:
                                w, h = map(int, resize_w.value.lower().split("x"))
                                df = df.resize((w, h))
                            except Exception:
                                pass
                    elif src_type == "graph":
                        if nx is None:
                            raise ImportError("networkx not installed")
                        fmt_w = self._adv_widgets.get("format")
                        fmt   = fmt_w.value if fmt_w else "auto"
                        if fmt == "graphml" or (fmt == "auto" and path.endswith(".graphml")):
                            df = nx.read_graphml(path)
                        elif fmt == "gml" or (fmt == "auto" and path.endswith(".gml")):
                            df = nx.read_gml(path)
                        else:
                            df = nx.read_edgelist(path)
                    elif src_type == "ontology":
                        if rdflib is None:
                            raise ImportError("rdflib not installed")
                        fmt_w = self._adv_widgets.get("format")
                        fmt   = fmt_w.value if fmt_w else "auto"
                        df = rdflib.Graph()
                        df.parse(path) if fmt == "auto" else df.parse(path, format=fmt)
                    elif src_type == "zip":
                        ctype_w = self._adv_widgets.get("content_type")
                        ctype   = ctype_w.value if ctype_w else "csv"
                        df = []
                        with zipfile.ZipFile(path, "r") as z:
                            for name in z.namelist():
                                if name.endswith("/") or "__MACOSX" in name:
                                    continue
                                with z.open(name) as f:
                                    if ctype == "csv":
                                        df.append(pd.read_csv(f))
                                    elif ctype == "json":
                                        df.append(pd.read_json(f))
                                    elif ctype == "text":
                                        df.append(f.read().decode("utf-8", errors="ignore"))
                                    elif ctype == "image" and Image:
                                        df.append(Image.open(f).copy())
                    else:
                        df = self._load_csv(path, src_type)

                    df = self._apply_common_post_load(df)

                    # ── auto-split ────────────────────────────────────────────
                    auto_split = "None"
                    if self._get_current_mode_id() == "single":
                        asw = self._adv_widgets.get("auto_split")
                        if asw and asw.value != "None" and isinstance(df, pd.DataFrame):
                            auto_split = asw.value

                    if auto_split != "None":
                        from sklearn.model_selection import train_test_split
                        tsw   = self._adv_widgets.get("test_size")
                        vsw   = self._adv_widgets.get("val_size")
                        stratw = self._adv_widgets.get("stratify")
                        test_size = tsw.value if tsw else 0.2
                        val_size  = vsw.value if vsw else 0.1
                        strat     = stratw.value.strip() if stratw and stratw.value.strip() else None
                        strat_data = df[strat] if strat and strat in df.columns else None
                        if auto_split == "Train/Val/Test":
                            tr_val, te = train_test_split(df, test_size=test_size,
                                                           stratify=strat_data, random_state=42)
                            val_ratio = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.1
                            val_ratio = max(min(val_ratio, 0.99), 0.01)
                            strat_val = tr_val[strat] if strat and strat in tr_val.columns else None
                            tr, va = train_test_split(tr_val, test_size=val_ratio,
                                                       stratify=strat_val, random_state=42)
                            for k, v in [("Train", tr), ("Validation", va), ("Test", te)]:
                                self.state.data_raw[k]   = v
                                self.state.data_types[k] = "tabular"
                                loaded.append(k)
                            print(f" -> Auto-split: Train={tr.shape}, Val={va.shape}, Test={te.shape}")
                        else:
                            tr, te = train_test_split(df, test_size=test_size,
                                                       stratify=strat_data, random_state=42)
                            for k, v in [("Train", tr), ("Test", te)]:
                                self.state.data_raw[k]   = v
                                self.state.data_types[k] = "tabular"
                                loaded.append(k)
                            print(f" -> Auto-split: Train={tr.shape}, Test={te.shape}")
                    else:
                        self.state.data_raw[slot_name]   = df
                        self.state.data_types[slot_name] = (
                            "tabular" if src_type not in ("image","text","graph","ontology","zip")
                            else src_type
                        )
                        loaded.append(slot_name)
                        size_str = str(df.shape) if hasattr(df, "shape") else str(len(df))
                        print(f" -> Success! Size: {size_str}")

                except Exception as e:
                    print(f" -> Error loading {slot_name}: {e}")
                    traceback.print_exc()

            if loaded:
                self.state.log_step("loader", "load_data",
                                     {"loaded_keys": loaded, "source": src_type})

    def _on_preview(self, _) -> None:
        with self.out:
            clear_output()
            if not self.state.data_raw:
                print("No data loaded yet.")
                return
            for key, data in self.state.data_raw.items():
                if isinstance(data, pd.DataFrame):
                    display(HTML(
                        f"<h4>{key} <span style='font-weight:normal;color:#666;'>"
                        f"({data.shape[0]} rows, {data.shape[1]} cols)</span></h4>"
                    ))
                    display(data.head())
                else:
                    size_str = str(getattr(data, "shape", getattr(data, "__len__", lambda: "?")())) 
                    display(HTML(f"<h4>{key} (Type: {type(data).__name__}, Size: {size_str})</h4>"))
                    if isinstance(data, str):
                        print(data[:500] + ("..." if len(data) > 500 else ""))


def runner(state) -> DataLoaderUI:
    loader = DataLoaderUI(state)
    display(loader.ui)
    return loader
