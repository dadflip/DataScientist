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
        self.slots_box      = widgets.VBox([])
        self.mode_config_box = widgets.VBox([])   # configs liées au mode (hors accordion)
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
            self.mode_config_box,
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
        src_type    = self.supported_types.get(self.source_dd.value, "")
        is_sklearn  = src_type == "sklearn"
        is_clipboard = src_type == "clipboard"
        options     = self.config.get("sklearn_datasets", []) if is_sklearn else []
        accept_exts = self.config.get("file_accepts", {}).get(src_type, "")

        if mode_id == "multi_source":
            self._build_multi_source_slots(src_type, is_sklearn, is_clipboard, options, accept_exts)
            return

        slots = ["Data"]
        for m in self.modes:
            if m.get("value") == mode_id:
                slots = m.get("slots", ["Data"])
                break
        children = []
        for s in slots:
            children.append(self._make_slot_row(s, src_type, is_sklearn, is_clipboard, options, accept_exts))
        self.slots_box.children = children

    def _make_slot_row(self, slot_name: str, src_type: str, is_sklearn: bool,
                       is_clipboard: bool, options: list, accept_exts: str) -> widgets.HBox:
        lbl = widgets.Label(f"{slot_name}:", layout=widgets.Layout(width="100px"))
        if is_sklearn:
            inp = widgets.Dropdown(options=options, layout=styles.LAYOUT_DD)
            return widgets.HBox([lbl, inp])
        elif is_clipboard:
            inp = widgets.Textarea(placeholder="Paste CSV text here...",
                                    layout=widgets.Layout(width="400px", height="100px"))
            return widgets.HBox([lbl, inp])
        else:
            inp_txt  = widgets.Text(placeholder="URL or Local File Path", layout=styles.LAYOUT_TEXT)
            inp_file = widgets.FileUpload(accept=accept_exts, multiple=False,
                                           layout=widgets.Layout(width="max-content"))
            return widgets.HBox([lbl, inp_txt, widgets.Label(" OR "), inp_file],
                                  layout=widgets.Layout(align_items="center", gap="10px", margin="4px 0"))

    def _build_multi_source_slots(self, src_type: str, is_sklearn: bool,
                                   is_clipboard: bool, options: list, accept_exts: str) -> None:
        """Mode multi-source : slots dynamiques avec boutons + / - sur le dernier slot."""
        self._multi_slot_counter: list[int] = [3]  # prochain index

        btn_add = widgets.Button(description="+", button_style=styles.BTN_SUCCESS,
                                  tooltip="Ajouter une source",
                                  layout=widgets.Layout(width="32px", height="28px"))
        btn_rem = widgets.Button(description="−", button_style=styles.BTN_DANGER,
                                  tooltip="Retirer la dernière source",
                                  layout=widgets.Layout(width="32px", height="28px"))

        rows_box = widgets.VBox([])

        def _make_row(slot_name: str) -> widgets.HBox:
            base = self._make_slot_row(slot_name, src_type, is_sklearn, is_clipboard, options, accept_exts)
            return base

        def _refresh_buttons() -> None:
            """Déplace les boutons +/- à la fin du dernier slot."""
            rows = list(rows_box.children)
            if not rows:
                return
            # Retirer les boutons de tous les slots
            clean = []
            for row in rows:
                kids = [c for c in row.children if c not in (btn_add, btn_rem)]
                row.children = kids
                clean.append(row)
            # Ajouter les boutons sur le dernier
            last = clean[-1]
            last.children = list(last.children) + [btn_add, btn_rem]
            rows_box.children = clean

        def _add_slot(_=None) -> None:
            idx = self._multi_slot_counter[0]
            self._multi_slot_counter[0] += 1
            row = _make_row(f"Source {idx}")
            rows_box.children = list(rows_box.children) + [row]
            _refresh_buttons()

        def _remove_slot(_=None) -> None:
            if len(rows_box.children) > 1:
                rows_box.children = list(rows_box.children)[:-1]
                _refresh_buttons()

        btn_add.on_click(_add_slot)
        btn_rem.on_click(_remove_slot)

        # Initialiser avec 2 slots
        for i in range(1, 3):
            rows_box.children = list(rows_box.children) + [_make_row(f"Source {i}")]
        _refresh_buttons()

        self.slots_box.children = [rows_box]
        self._multi_rows_box = rows_box

    def _build_adv_config(self) -> None:
        src_type  = self.supported_types.get(self.source_dd.value, "")
        mode_id   = self._get_current_mode_id()
        self._adv_widgets.clear()

        adv_list  = self.config.get("adv_configs", {}).get(src_type, [])
        mode_list = self.config.get("mode_configs", {}).get(mode_id, [])

        # ── Mode configs : affichées directement sous les slots ───────────────
        mode_children = []
        for conf in mode_list:
            w = self._make_config_widget(conf)
            if w is not None:
                mode_children.append(w)
        if mode_children:
            sep = widgets.HTML(
                "<div style='height:1px;background:#e5e7eb;margin:6px 0 4px 0;'></div>")
            self.mode_config_box.children = [sep] + mode_children
            self.mode_config_box.layout.display = "block"
        else:
            self.mode_config_box.children = []
            self.mode_config_box.layout.display = "none"

        # ── Adv configs : dans l'accordion ────────────────────────────────────
        adv_children = []
        for conf in adv_list:
            w = self._make_config_widget(conf)
            if w is not None:
                adv_children.append(w)
        if adv_children:
            grid = widgets.GridBox(adv_children, layout=widgets.Layout(
                grid_template_columns="repeat(auto-fit,minmax(320px,1fr))",
                gap="10px", align_items="center"))
            self.adv_config_box.children = [grid]
            self.adv_config_box.layout.display = "block"
        else:
            self.adv_config_box.children = [
                widgets.HTML("<i>No advanced configuration needed for this type/mode.</i>")]
            self.adv_config_box.layout.display = "none"

    def _make_config_widget(self, conf: dict):
        """Crée un widget ipywidgets depuis un dict de config, l'enregistre dans _adv_widgets."""
        w_type = conf.get("type")
        w_id   = conf.get("id")
        if not w_id:
            return None
        help_t = conf.get("help", "")
        label  = conf.get("description") or conf.get("label", "")

        if w_type == "text":
            w = widgets.Text(value=str(conf.get("value", "")),
                              placeholder=str(conf.get("placeholder", "")),
                              description=label,
                              layout=styles.LAYOUT_TEXT,
                              style={"description_width": "initial"})
        elif w_type == "dropdown":
            opts = conf.get("options", [])
            if "options_key" in conf:
                opts = self.config.get(conf["options_key"], [])
            val = conf.get("value", "")
            if opts and val not in opts:
                val = opts[0]
            w = widgets.Dropdown(options=opts, value=val,
                                  description=label,
                                  layout=styles.LAYOUT_TEXT,
                                  style={"description_width": "initial"})
        elif w_type == "checkbox":
            w = widgets.Checkbox(value=bool(conf.get("value", False)),
                                  description=label,
                                  layout=widgets.Layout(width="auto"),
                                  style={"description_width": "initial"}, indent=False)
        elif w_type == "floatslider":
            slider = widgets.FloatSlider(
                value=float(conf.get("value", 1.0)),
                min=float(conf.get("min", 0.0)),
                max=float(conf.get("max", 1.0)),
                step=float(conf.get("step", 0.1)),
                description="",
                layout=widgets.Layout(width="200px"),
                style={"description_width": "0px"},
                readout_format=".2f")
            lbl_w = widgets.Label(label, layout=widgets.Layout(width="110px", min_width="110px"))
            self._adv_widgets[w_id] = slider
            row_kids = [lbl_w, slider]
            if help_t:
                row_kids.append(widgets.HTML(
                    f"<span title='{help_t}' style='cursor:help;color:#3b82f6;margin-left:4px;'>&#9432;</span>"))
            return widgets.HBox(row_kids, layout=widgets.Layout(align_items="center"))
        else:
            return None

        self._adv_widgets[w_id] = w
        if help_t:
            icon = widgets.HTML(
                f"<span title='{help_t}' style='cursor:help;color:#3b82f6;margin-left:4px;'>&#9432;</span>")
            return widgets.HBox([w, icon], layout=widgets.Layout(align_items="center"))
        return w

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
        # Vider et préparer le Output widget
        self.out.clear_output(wait=True)
        src_type = self.supported_types.get(self.source_dd.value, "")
        loaded: list[str] = []

        if self.clear_cb.value:
            self.state.data_raw.clear()
            self.state.data_types.clear()

        # En mode multi-source les slots sont dans un VBox imbriqué
        mode_id = self._get_current_mode_id()
        if mode_id == "multi_source" and hasattr(self, "_multi_rows_box"):
            slot_rows = list(self._multi_rows_box.children)
        else:
            slot_rows = list(self.slots_box.children)

        order_w = self._adv_widgets.get("load_order")
        if src_type == "ontology" and order_w and not order_w.value:
            slot_rows = list(reversed(slot_rows))
        if src_type == "ontology":
            self._onto_already_loaded = {}
            self._pending_unresolved  = {}

        # Charger les slots en capturant les prints dans self.out
        with self.out:
            self._load_slots(slot_rows, src_type, loaded)
            if loaded:
                self.state.log_step("loader", "load_data",
                                     {"loaded_keys": loaded, "source": src_type})

        # Post-UI ontologie affiché HORS du Output widget
        if src_type == "ontology" and loaded:
            self._show_ontology_post_ui(loaded)

    def _load_slots(self, slot_rows, src_type, loaded):
        """Boucle de chargement des slots — séparée pour pouvoir sortir le post-UI du Output."""
        for child in slot_rows:
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
                    orig_filename = slot_name  # nom affiché par défaut
                    if uploaded_file:
                        orig_filename = uploaded_file["name"]
                        ext = os.path.splitext(uploaded_file["name"])[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(uploaded_file["content"])
                            path = tmp.name
                    elif path:
                        orig_filename = os.path.basename(path)
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
                        import io as _io, re as _re
                        try:
                            import owlready2 as _owlready2
                        except ImportError:
                            _owlready2 = None

                        fmt_w    = self._adv_widgets.get("format")
                        store_w  = self._adv_widgets.get("store")
                        fmt      = fmt_w.value if fmt_w else "auto"
                        use_conj = store_w.value == "ConjunctiveGraph" if store_w else False

                        # Dossiers locaux pour résolution des imports
                        local_dirs = [os.path.dirname(os.path.abspath(path))]
                        extra_dirs_w = self._adv_widgets.get("local_dirs")
                        if extra_dirs_w and extra_dirs_w.value.strip():
                            for d in extra_dirs_w.value.split(";"):
                                d = d.strip()
                                if os.path.isdir(d):
                                    local_dirs.append(d)

                        # Cache inter-slots pour résolution croisée
                        _already = getattr(self, "_onto_already_loaded", {})

                        # ── helpers de détection ──────────────────────────────
                        def _sniff(content: str) -> str:
                            """Retourne le format détecté : owlxml | rdfxml | turtle | n3 | nt | jsonld | unknown."""
                            h = content[:4096]
                            if ("<Ontology" in h or "ontologyIRI=" in h) and "<rdf:RDF" not in h:
                                return "owlxml"
                            if "<rdf:RDF" in h or "<?xml" in h:
                                return "rdfxml"
                            if h.lstrip().startswith("{") or '"@context"' in h:
                                return "jsonld"
                            if _re.search(r"^\s*@prefix\s", h, _re.M) or _re.search(r"^\s*@base\s", h, _re.M):
                                return "turtle"
                            if _re.search(r"^\s*#", h, _re.M) and _re.search(r"<http", h):
                                return "n3"
                            # NT: lignes de la forme <uri> <uri> <uri|"lit"> .
                            if _re.search(r"^<[^>]+>\s+<[^>]+>\s+", h, _re.M):
                                return "nt"
                            return "unknown"

                        def _extract_base_iri(content: str, fname: str) -> str:
                            m = (_re.search(r'ontologyIRI="([^"]+)"', content)
                                 or _re.search(r'xml:base="([^"]+)"', content)
                                 or _re.search(r'rdf:about="([^"]*)"', content))
                            return m.group(1) if (m and m.group(1)) else f"http://temp/{fname}"

                        # ── parsers ───────────────────────────────────────────
                        def _owlxml_via_owlready2(fpath: str, content: str) -> tuple:
                            """OWL/XML → owlready2 avec fileobj (cross-platform, pas de path URI)."""
                            base_iri = _extract_base_iri(content, os.path.basename(fpath))
                            declared = _re.findall(r"<Import>\s*([^<]+?)\s*</Import>", content)
                            # Retirer les <Import> : owlready2 les résoudrait en réseau
                            patched  = _re.sub(r"\s*<Import>[^<]+</Import>", "", content)
                            world = _owlready2.World()
                            onto  = world.get_ontology(base_iri)
                            # fileobj= évite toute logique de path URI → cross-platform
                            with _io.BytesIO(patched.encode("utf-8")) as buf:
                                onto.load(fileobj=buf)
                            g = rdflib.ConjunctiveGraph() if use_conj else rdflib.Graph()
                            for triple in world.as_rdflib_graph():
                                g.add(triple)
                            return g, declared

                        def _owlxml_via_etree(fpath: str, content: str) -> tuple:
                            """Fallback OWL/XML sans owlready2 : parse XML → triples rdflib."""
                            import xml.etree.ElementTree as _ET
                            from rdflib import URIRef, Literal, BNode
                            from rdflib.namespace import RDF, RDFS, OWL as _OWL, XSD

                            NS = {
                                "owl":  "http://www.w3.org/2002/07/owl#",
                                "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                                "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                                "xsd":  "http://www.w3.org/2001/XMLSchema#",
                            }
                            base_iri = _extract_base_iri(content, os.path.basename(fpath))
                            declared = _re.findall(r"<Import>\s*([^<]+?)\s*</Import>", content)
                            g = rdflib.ConjunctiveGraph() if use_conj else rdflib.Graph()
                            g.bind("owl", _OWL); g.bind("rdf", RDF); g.bind("rdfs", RDFS)

                            def _iri(iri_str: str) -> URIRef:
                                if iri_str.startswith("#"):
                                    return URIRef(base_iri + iri_str)
                                if iri_str.startswith("http"):
                                    return URIRef(iri_str)
                                return URIRef(base_iri + "#" + iri_str.lstrip("#"))

                            try:
                                root = _ET.fromstring(content.encode("utf-8"))
                            except _ET.ParseError as e:
                                raise ValueError(f"OWL/XML parse error: {e}")

                            # Lire les prefixes déclarés
                            prefixes = {v: k for k, v in NS.items()}
                            for child in root:
                                tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                                if tag == "Prefix":
                                    name = child.get("name", "")
                                    iri  = child.get("IRI", "")
                                    if iri:
                                        prefixes[iri] = name

                            def _abbrev(iri_str: str) -> URIRef:
                                for prefix_iri, _ in prefixes.items():
                                    if iri_str.startswith(prefix_iri):
                                        return URIRef(iri_str)
                                return _iri(iri_str)

                            # Mapper les constructions OWL/XML → triples RDF
                            type_map = {
                                "Class":             RDF.type,
                                "ObjectProperty":    RDF.type,
                                "DataProperty":      RDF.type,
                                "AnnotationProperty":RDF.type,
                                "NamedIndividual":   RDF.type,
                                "Datatype":          RDF.type,
                            }
                            type_obj_map = {
                                "Class":             _OWL.Class,
                                "ObjectProperty":    _OWL.ObjectProperty,
                                "DataProperty":      _OWL.DatatypeProperty,
                                "AnnotationProperty":_OWL.AnnotationProperty,
                                "NamedIndividual":   _OWL.NamedIndividual,
                                "Datatype":          RDFS.Datatype,
                            }

                            onto_uri = URIRef(base_iri)
                            g.add((onto_uri, RDF.type, _OWL.Ontology))

                            for elem in root:
                                tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                                if tag == "Declaration":
                                    for child in elem:
                                        ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                                        iri_val = child.get("IRI") or child.get("abbreviatedIRI", "")
                                        if iri_val and ctag in type_map:
                                            subj = _abbrev(iri_val)
                                            g.add((subj, RDF.type, type_obj_map[ctag]))
                                elif tag == "SubClassOf":
                                    children = list(elem)
                                    if len(children) >= 2:
                                        def _get_iri(el):
                                            t = el.tag.split("}")[-1] if "}" in el.tag else el.tag
                                            if t in ("Class", "ObjectProperty", "DataProperty"):
                                                return _abbrev(el.get("IRI") or el.get("abbreviatedIRI", ""))
                                            return None
                                        s = _get_iri(children[0])
                                        o = _get_iri(children[1])
                                        if s and o:
                                            g.add((s, RDFS.subClassOf, o))
                                elif tag in ("ObjectPropertyDomain", "DataPropertyDomain"):
                                    children = list(elem)
                                    if len(children) >= 2:
                                        prop_iri = children[0].get("IRI") or children[0].get("abbreviatedIRI", "")
                                        cls_iri  = children[1].get("IRI") or children[1].get("abbreviatedIRI", "")
                                        if prop_iri and cls_iri:
                                            g.add((_abbrev(prop_iri), RDFS.domain, _abbrev(cls_iri)))
                                elif tag in ("ObjectPropertyRange", "DataPropertyRange"):
                                    children = list(elem)
                                    if len(children) >= 2:
                                        prop_iri = children[0].get("IRI") or children[0].get("abbreviatedIRI", "")
                                        rng_iri  = children[1].get("IRI") or children[1].get("abbreviatedIRI", "")
                                        if prop_iri and rng_iri:
                                            g.add((_abbrev(prop_iri), RDFS.range, _abbrev(rng_iri)))
                                elif tag == "AnnotationAssertion":
                                    children = list(elem)
                                    if len(children) >= 3:
                                        prop_tag = children[0].tag.split("}")[-1] if "}" in children[0].tag else children[0].tag
                                        prop_iri = children[0].get("IRI") or children[0].get("abbreviatedIRI", "")
                                        subj_iri = children[1].get("IRI") or children[1].get("abbreviatedIRI", "")
                                        val_el   = children[2]
                                        if prop_iri and subj_iri:
                                            pred = _abbrev(prop_iri)
                                            subj = _abbrev(subj_iri)
                                            val_iri = val_el.get("IRI") or val_el.get("abbreviatedIRI", "")
                                            if val_iri:
                                                g.add((subj, pred, _abbrev(val_iri)))
                                            elif val_el.text:
                                                lang = val_el.get("{http://www.w3.org/XML/1998/namespace}lang")
                                                dtype = val_el.get("datatypeIRI")
                                                if lang:
                                                    g.add((subj, pred, Literal(val_el.text, lang=lang)))
                                                elif dtype:
                                                    g.add((subj, pred, Literal(val_el.text, datatype=URIRef(dtype))))
                                                else:
                                                    g.add((subj, pred, Literal(val_el.text)))
                                elif tag == "ObjectPropertyAssertion":
                                    children = list(elem)
                                    if len(children) >= 3:
                                        prop_iri = children[0].get("IRI") or children[0].get("abbreviatedIRI", "")
                                        s_iri    = children[1].get("IRI") or children[1].get("abbreviatedIRI", "")
                                        o_iri    = children[2].get("IRI") or children[2].get("abbreviatedIRI", "")
                                        if prop_iri and s_iri and o_iri:
                                            g.add((_abbrev(s_iri), _abbrev(prop_iri), _abbrev(o_iri)))

                            return g, declared

                        def _rdf_cascade(fpath: str, forced_fmt: str = "auto", display_name: str = "") -> tuple:
                            """RDF/XML, Turtle, N3, NT, JSON-LD via rdflib avec cascade."""
                            from rdflib.namespace import OWL as _OWL
                            ext = os.path.splitext(fpath)[-1].lower()
                            display_name = display_name or os.path.basename(fpath)
                            sniffed = _sniff(open(fpath, "r", encoding="utf-8", errors="ignore").read(4096))
                            if forced_fmt != "auto":
                                fmts = [forced_fmt]
                            else:
                                # Ordre prioritaire selon sniff + extension
                                sniff_map = {
                                    "rdfxml":  ["xml", "turtle", "n3"],
                                    "turtle":  ["turtle", "n3", "xml"],
                                    "n3":      ["n3", "turtle", "xml"],
                                    "nt":      ["nt", "xml"],
                                    "jsonld":  ["json-ld"],
                                    "unknown": ["xml", "turtle", "n3", "nt", "json-ld"],
                                }
                                ext_map = {
                                    ".ttl":    ["turtle", "n3"],
                                    ".n3":     ["n3", "turtle"],
                                    ".nt":     ["nt"],
                                    ".jsonld": ["json-ld"],
                                    ".json":   ["json-ld"],
                                    ".owl":    ["xml", "turtle", "n3"],
                                    ".rdf":    ["xml", "turtle", "n3"],
                                }
                                fmts = ext_map.get(ext) or sniff_map.get(sniffed, ["xml", "turtle", "n3"])

                            last_err = None
                            for fmt_try in fmts:
                                try:
                                    g2 = rdflib.ConjunctiveGraph() if use_conj else rdflib.Graph()
                                    g2.parse(fpath, format=fmt_try)
                                    declared = [str(o) for o in g2.objects(None, _OWL.imports)]
                                    print(f"   [ontology] {display_name} → rdflib '{fmt_try}'")
                                    return g2, declared
                                except Exception as e:
                                    last_err = e
                            raise last_err or RuntimeError(f"Aucun format rdflib n'a fonctionné pour {fpath}")

                        # ── chargeur principal ────────────────────────────────
                        def _load_one(fpath: str, forced_fmt: str = "auto") -> tuple:
                            """Charge une ontologie, retourne (graph, declared_imports)."""
                            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                                content = fh.read()
                            sniffed = _sniff(content)
                            display_name = orig_filename if fpath == path else os.path.basename(fpath)

                            if forced_fmt == "owl" or (forced_fmt == "auto" and sniffed == "owlxml"):
                                if _owlready2 is not None:
                                    try:
                                        print(f"   [ontology] {display_name} → OWL/XML (owlready2)")
                                        return _owlxml_via_owlready2(fpath, content)
                                    except Exception as e1:
                                        print(f"   [ontology] owlready2 échoué ({e1}), fallback etree")
                                        return _owlxml_via_etree(fpath, content)
                                else:
                                    print(f"   [ontology] {display_name} → OWL/XML (etree)")
                                    return _owlxml_via_etree(fpath, content)
                            else:
                                return _rdf_cascade(fpath, forced_fmt, display_name)

                        # ── résolution récursive des imports ──────────────────
                        def _resolve_imports(g, declared, depth=0):
                            if depth > 6:
                                return list(declared)
                            unresolved_imp = []
                            for imp_iri in declared:
                                imp_name = imp_iri.rstrip("/").rsplit("/", 1)[-1]
                                if imp_iri in _already:
                                    for t in _already[imp_iri]:
                                        g.add(t)
                                    print(f"   {'  '*depth}[import ✓] {imp_name} ← cache")
                                    continue
                                found = None
                                for d in local_dirs:
                                    for cand in [imp_name, imp_name + ".owl", imp_name + ".rdf",
                                                 imp_name + ".ttl", imp_name + ".n3"]:
                                        p = os.path.join(d, cand)
                                        if os.path.exists(p):
                                            found = p
                                            break
                                    if found:
                                        break
                                if found:
                                    try:
                                        sub_g, sub_decl = _load_one(found)
                                        sub_unres = _resolve_imports(sub_g, sub_decl, depth + 1)
                                        for t in sub_g:
                                            g.add(t)
                                        unresolved_imp.extend(sub_unres)
                                        print(f"   {'  '*depth}[import ✓] {imp_name} ← local ({len(sub_g)} triples)")
                                    except Exception as e:
                                        unresolved_imp.append(imp_iri)
                                        print(f"   {'  '*depth}[import ✗] {imp_name} ← erreur: {e}")
                                else:
                                    unresolved_imp.append(imp_iri)
                                    print(f"   {'  '*depth}[import ✗] {imp_name} ← non résolu")
                            return unresolved_imp

                        # ── chargement principal ──────────────────────────────
                        g_onto, declared_imports = _load_one(path, fmt)
                        unresolved_imports = _resolve_imports(g_onto, declared_imports)

                        # Stocker dans le cache inter-slots
                        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                            _head = fh.read(512)
                        _base_iri = _extract_base_iri(_head, orig_filename)
                        if not hasattr(self, "_onto_already_loaded"):
                            self._onto_already_loaded = {}
                        self._onto_already_loaded[_base_iri] = g_onto
                        # Stocker le nom original sur le graphe pour la preview
                        try:
                            g_onto._orig_filename = orig_filename
                        except Exception:
                            pass

                        df = g_onto
                        if unresolved_imports:
                            self._pending_unresolved = getattr(self, "_pending_unresolved", {})
                            self._pending_unresolved[slot_name] = {
                                "iris": unresolved_imports,
                                "filename": orig_filename,
                            }
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

    def _show_ontology_post_ui(self, loaded: list[str]) -> None:
        """Panneau post-chargement : ordre, imports non résolus, preview graphe + triplets."""
        import io as _io
        import matplotlib
        import matplotlib.pyplot as plt
        from IPython.display import display as ipy_display

        onto_keys = [k for k in loaded if k in self.state.data_raw
                     and hasattr(self.state.data_raw[k], "objects")]
        pending   = getattr(self, "_pending_unresolved", {})

        if not onto_keys and not any(s in loaded for s in pending):
            return

        panels = []

        # ── 1. Ordre de chargement ────────────────────────────────────────────
        if len(onto_keys) > 1:
            order_list = list(onto_keys)
            panels.append(widgets.HTML(
                "<div style='font-weight:600;color:#374151;margin:8px 0 4px 0;'>"
                "📋 Ordre des ontologies chargées</div>"
            ))
            rows_vbox = widgets.VBox([])

            def _rebuild_rows():
                new_rows = []
                for i, key in enumerate(order_list):
                    g   = self.state.data_raw.get(key)
                    n   = len(g) if g and hasattr(g, "__len__") else "?"
                    fn  = getattr(g, "_orig_filename", key)
                    lbl = widgets.HTML(
                        f"<span style='padding:3px 8px;background:#f1f5f9;"
                        f"border:1px solid #cbd5e1;border-radius:4px;font-size:0.83em;'>"
                        f"<b>{key}</b> <span style='color:#6b7280;'>— {fn} ({n} triples)</span></span>"
                    )
                    btn_up = widgets.Button(description="↑", tooltip="Monter",
                                            layout=widgets.Layout(width="28px", height="24px"))
                    btn_dn = widgets.Button(description="↓", tooltip="Descendre",
                                            layout=widgets.Layout(width="28px", height="24px"))
                    def _make_mover(idx, delta):
                        def _move(_):
                            ni = idx + delta
                            if 0 <= ni < len(order_list):
                                order_list[idx], order_list[ni] = order_list[ni], order_list[idx]
                                reordered = {k: self.state.data_raw[k] for k in order_list
                                             if k in self.state.data_raw}
                                for k, v in self.state.data_raw.items():
                                    if k not in reordered:
                                        reordered[k] = v
                                self.state.data_raw = reordered
                                _rebuild_rows()
                        return _move
                    btn_up.on_click(_make_mover(i, -1))
                    btn_dn.on_click(_make_mover(i, +1))
                    new_rows.append(widgets.HBox(
                        [lbl, btn_up, btn_dn],
                        layout=widgets.Layout(gap="4px", align_items="center", margin="2px 0")))
                rows_vbox.children = new_rows

            _rebuild_rows()
            panels.append(rows_vbox)

        # ── 2. Imports non résolus ────────────────────────────────────────────
        unresolved_all = {s: info for s, info in pending.items() if s in loaded}
        if unresolved_all:
            loaded_onto_opts = [(f"{k}  ({len(v) if hasattr(v,'__len__') else '?'} triples)", k)
                                for k, v in self.state.data_raw.items() if hasattr(v, "objects")]
            panels.append(widgets.HTML(
                "<div style='font-weight:600;color:#b45309;margin:12px 0 2px 0;'>⚠️ Imports non résolus</div>"
                "<div style='font-size:0.8em;color:#6b7280;margin-bottom:6px;'>"
                "Remapper vers une source chargée ou saisir une URL/chemin.</div>"
            ))
            for slot, info in unresolved_all.items():
                iris     = info["iris"] if isinstance(info, dict) else info
                filename = info.get("filename", slot) if isinstance(info, dict) else slot
                panels.append(widgets.HTML(
                    f"<div style='font-size:0.82em;color:#374151;margin:6px 0 2px 0;'>"
                    f"Depuis <b>{slot}</b> <span style='color:#6b7280;'>({filename})</span> :</div>"
                ))
                for uri in iris:
                    imp_name = uri.rstrip("/").rsplit("/", 1)[-1]
                    uri_lbl  = widgets.HTML(
                        f"<code style='font-size:0.78em;color:#1e40af;background:#eff6ff;"
                        f"padding:2px 6px;border-radius:3px;'>{imp_name}</code>"
                    )
                    dd_opts  = ["(ignorer)"] + [lbl for lbl, _ in loaded_onto_opts] + ["(saisir...)"]
                    remap_dd = widgets.Dropdown(options=dd_opts, value="(ignorer)",
                                                 layout=widgets.Layout(width="240px"))
                    remap_txt = widgets.Text(placeholder="URL ou chemin local",
                                              layout=widgets.Layout(width="260px", display="none"))
                    apply_btn = widgets.Button(description="Appliquer", button_style="warning",
                                               layout=widgets.Layout(width="90px", height="28px"))
                    status    = widgets.HTML("")

                    def _on_dd(change, txt=remap_txt):
                        txt.layout.display = "flex" if change["new"] == "(saisir...)" else "none"
                    remap_dd.observe(_on_dd, names="value")

                    def _on_apply(_, u=uri, dd=remap_dd, txt=remap_txt,
                                  sl=status, opts=loaded_onto_opts):
                        sel = dd.value
                        if sel == "(ignorer)":
                            sl.value = "<span style='color:#9ca3af;font-size:0.8em;'>ignoré</span>"
                            return
                        target_key  = next((k for lbl, k in opts if lbl == sel), None)
                        new_iri_str = txt.value.strip() if sel == "(saisir...)" else None
                        from rdflib.namespace import OWL as _OWL
                        from rdflib import URIRef
                        old_uri = URIRef(u)
                        if target_key and target_key in self.state.data_raw:
                            tg = self.state.data_raw[target_key]
                            for g in self.state.data_raw.values():
                                if not hasattr(g, "remove"):
                                    continue
                                if (None, _OWL.imports, old_uri) in g:
                                    for t in tg:
                                        g.add(t)
                                    g.remove((None, _OWL.imports, old_uri))
                            sl.value = f"<span style='color:#059669;font-size:0.8em;'>✓ fusionné depuis {target_key}</span>"
                        elif new_iri_str:
                            new_uri = URIRef(new_iri_str)
                            for g in self.state.data_raw.values():
                                if not hasattr(g, "remove"):
                                    continue
                                for s, p, o in list(g.triples((None, _OWL.imports, old_uri))):
                                    g.remove((s, p, o)); g.add((s, p, new_uri))
                            sl.value = f"<span style='color:#059669;font-size:0.8em;'>✓ remappé → {new_iri_str[:40]}</span>"
                    apply_btn.on_click(_on_apply)
                    panels.append(widgets.HBox(
                        [uri_lbl, remap_dd, remap_txt, apply_btn, status],
                        layout=widgets.Layout(gap="6px", align_items="center",
                                              flex_wrap="wrap", margin="3px 0")))
        self._pending_unresolved = {}

        # ── 3. Preview graphe (toutes sources confondues) ─────────────────────
        try:
            import networkx as nx
            from rdflib.namespace import RDF, RDFS, OWL as _OWL

            # Fusionner tous les graphes ontologie chargés
            merged = rdflib.Graph()
            for k in onto_keys:
                g = self.state.data_raw.get(k)
                if g and hasattr(g, "__iter__"):
                    for t in g:
                        merged.add(t)

            # Construire un graphe NetworkX de classes/propriétés (max 60 nœuds)
            G = nx.DiGraph()
            def _short(uri):
                s = str(uri)
                return s.split("#")[-1] if "#" in s else s.rsplit("/", 1)[-1]

            edges_added = 0
            for s, p, o in merged:
                if edges_added >= 120:
                    break
                ps = _short(p)
                if ps in ("subClassOf", "domain", "range", "type",
                          "subPropertyOf", "equivalentClass"):
                    ss, os_ = _short(s), _short(o)
                    if ss and os_ and ss != os_:
                        G.add_edge(ss, os_, label=ps)
                        edges_added += 1

            if G.number_of_nodes() > 0:
                fig, ax = plt.subplots(figsize=(12, 7))
                fig.patch.set_facecolor("#f8fafc")
                ax.set_facecolor("#f8fafc")
                pos = nx.spring_layout(G, k=1.8, seed=42)
                # Colorier par type de nœud
                node_colors = []
                for node in G.nodes():
                    in_d, out_d = G.in_degree(node), G.out_degree(node)
                    if in_d == 0:
                        node_colors.append("#3b82f6")   # racine → bleu
                    elif out_d == 0:
                        node_colors.append("#10b981")   # feuille → vert
                    else:
                        node_colors.append("#8b5cf6")   # intermédiaire → violet
                nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                       node_size=600, alpha=0.85, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=7, font_color="white",
                                        font_weight="bold", ax=ax)
                edge_labels = nx.get_edge_attributes(G, "label")
                nx.draw_networkx_edges(G, pos, edge_color="#94a3b8",
                                       arrows=True, arrowsize=15,
                                       connectionstyle="arc3,rad=0.1", ax=ax)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                              font_size=6, font_color="#475569", ax=ax)
                ax.set_title(f"Aperçu des relations ontologiques — {merged.__len__()} triples fusionnés",
                             fontsize=10, color="#374151")
                ax.axis("off")
                plt.tight_layout()

                graph_out = widgets.Output()
                with graph_out:
                    ipy_display(fig)
                plt.close(fig)
                panels.append(widgets.HTML(
                    "<div style='font-weight:600;color:#374151;margin:14px 0 4px 0;'>"
                    "🕸️ Aperçu graphe (toutes sources)</div>"
                    "<div style='font-size:0.78em;color:#6b7280;margin-bottom:4px;'>"
                    "Bleu = racines · Violet = intermédiaires · Vert = feuilles · max 120 arêtes</div>"
                ))
                panels.append(graph_out)
        except Exception as e:
            panels.append(widgets.HTML(
                f"<div style='color:#9ca3af;font-size:0.8em;'>Aperçu graphe indisponible: {e}</div>"))

        # ── 4. Tableau de triplets par ontologie ──────────────────────────────
        panels.append(widgets.HTML(
            "<div style='font-weight:600;color:#374151;margin:14px 0 4px 0;'>"
            "📄 Aperçu des triplets par source</div>"
        ))
        tab_children, tab_titles = [], []
        for key in onto_keys:
            g = self.state.data_raw.get(key)
            if not g or not hasattr(g, "__iter__"):
                continue
            rows = []
            for s, p, o in g:
                rows.append({
                    "Subject":   str(s).split("#")[-1] if "#" in str(s) else str(s).rsplit("/",1)[-1],
                    "Predicate": str(p).split("#")[-1] if "#" in str(p) else str(p).rsplit("/",1)[-1],
                    "Object":    str(o).split("#")[-1] if "#" in str(o) else str(o).rsplit("/",1)[-1],
                })
                if len(rows) >= 200:
                    break
            df_triples = pd.DataFrame(rows)
            tab_out = widgets.Output()
            with tab_out:
                ipy_display(HTML(
                    f"<div style='font-size:0.8em;color:#6b7280;margin-bottom:4px;'>"
                    f"{len(g)} triples au total — affichage des 200 premiers</div>"
                ))
                ipy_display(df_triples)
            tab_children.append(tab_out)
            fn = getattr(g, "_orig_filename", key)
            tab_titles.append(f"{key} ({fn})")

        if tab_children:
            tabs = widgets.Tab(children=tab_children)
            for i, t in enumerate(tab_titles):
                tabs.set_title(i, t)
            panels.append(tabs)

        if panels:
            ipy_display(widgets.VBox(panels, layout=widgets.Layout(
                border="1px solid #e5e7eb", border_radius="6px",
                padding="12px 16px", margin="8px 0")))

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
