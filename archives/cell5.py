import sys, os, pathlib
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import io
import urllib.request
import traceback
import sqlite3
import pandas as pd
import tempfile
import zipfile
from PIL import Image
try:
    import networkx as nx
except ImportError:
    nx = None
try:
    import rdflib
except ImportError:
    rdflib = None
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

def _load_sklearn_dataset(dataset_name: str):
    from sklearn import datasets as _ds
    loader_func = getattr(_ds, f"load_{dataset_name}", None)
    if not loader_func:
        loader_func = getattr(_ds, f"fetch_{dataset_name}", None)
    if not loader_func:
        raise ValueError(f"Dataset '{dataset_name}' not found in sklearn.datasets")
    bunch = loader_func()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    if hasattr(bunch, "target"):
        df["target"] = bunch.target
    return df

class DataLoaderUI:
    def __init__(self, state):
        self.state = state
        if not self.state.config:
            self.ui = styles.error_msg("[ERROR] Configuration not loaded. Run Cell 1a first.")
            return
            
        self.config = state.config.get("loading", {})
        self.supported_types = self.config.get("supported_types", {})
        self.modes = self.config.get("modes", [])
        self._build_ui()
        
    def _build_ui(self):
        modes_opts = [m["label"] for m in self.modes] if self.modes else ["Single file"]
        self.mode_dd = widgets.Dropdown(options=modes_opts, value=modes_opts[0], description="Mode:", layout=styles.LAYOUT_DD_LONG)
        
        sources_opts = list(self.supported_types.keys()) if self.supported_types else ["csv"]
        self.source_dd = widgets.Dropdown(options=sources_opts, description="Source type:", layout=styles.LAYOUT_DD_LONG)
        
        self.slots_box = widgets.VBox([])
        
        # Advanced configurations Container
        self.adv_config_box = widgets.Accordion(children=[widgets.VBox([])])
        self.adv_config_box.set_title(0, "Advanced Configurations")
        self.adv_config_box.selected_index = None # closed by default
        
        # We will dynamically populate this based on source type
        self._adv_widgets = {}
        
        self.btn_load = widgets.Button(description="Load data", button_style=styles.BTN_PRIMARY, layout=styles.LAYOUT_BTN_STD)
        self.btn_preview = widgets.Button(description="Preview", button_style=styles.BTN_INFO, layout=styles.LAYOUT_BTN_STD)
        self.clear_cb = widgets.Checkbox(value=True, description="Clear existing data", layout=styles.LAYOUT_BTN_LARGE)
        self.out = widgets.Output()

        self.mode_dd.observe(self._on_mode_change, names="value")
        self.source_dd.observe(self._on_source_change, names="value")
        self.btn_load.on_click(self._on_load)
        self.btn_preview.on_click(self._on_preview)

        self._build_slots()
        self._build_adv_config()
        
        header_html = styles.card_html("Data Loader", "Import your data from multiple sources", "")

        self.ui = widgets.VBox([
            widgets.HTML(value=header_html),
            widgets.HBox([self.mode_dd, self.source_dd]),
            self.slots_box,
            self.adv_config_box,
            widgets.HBox([self.btn_load, self.btn_preview, self.clear_cb]),
            self.out
        ], layout=styles.LAYOUT_BOX)

    def _get_accept_for_source(self, src_type):
        return self.config.get("file_accepts", {}).get(src_type, "")

    def _get_current_mode_id(self):
        mode_val = self.mode_dd.value
        for m in self.modes:
            if m.get("label") == mode_val:
                return m.get("value", "single")
        return "single"

    def _build_slots(self):
        mode_id = self._get_current_mode_id()
        slots = ["Data"]
        for m in self.modes:
            if m.get("value") == mode_id:
                slots = m.get("slots", ["Data"])
                break
                
        src_type = self.supported_types.get(self.source_dd.value, "")
        is_sklearn = src_type == "sklearn"
        is_clipboard = src_type == "clipboard"
        options = self.config.get("sklearn_datasets", []) if is_sklearn else []
        accept_exts = self._get_accept_for_source(src_type)
        
        children = []
        for s in slots:
            lbl = widgets.Label(f"{s}:", layout=widgets.Layout(width="100px", font_weight="bold"))
            if is_sklearn:
                inp = widgets.Dropdown(options=options, layout=styles.LAYOUT_DD)
                children.append(widgets.HBox([lbl, inp]))
            elif is_clipboard:
                inp = widgets.Textarea(placeholder="Paste CSV text here...", layout=widgets.Layout(width='400px', height='100px'))
                children.append(widgets.HBox([lbl, inp]))
            else:
                inp_txt = widgets.Text(placeholder="URL or Local File Path", layout=styles.LAYOUT_TEXT)
                inp_file = widgets.FileUpload(accept=accept_exts, multiple=False, layout=widgets.Layout(width="max-content"))
                children.append(widgets.HBox([lbl, inp_txt, widgets.Label(" OR "), inp_file], layout=widgets.Layout(align_items='center', gap='10px', margin='4px 0')))
        self.slots_box.children = children

    def _build_adv_config(self):
        src_type = self.supported_types.get(self.source_dd.value, "")
        mode_id = self._get_current_mode_id()
        self._adv_widgets.clear()
        
        adv_config_list = self.config.get("adv_configs", {}).get(src_type, [])
        mode_config_list = self.config.get("mode_configs", {}).get(mode_id, [])
        combined_configs = adv_config_list + mode_config_list
        
        children = []
        for conf in combined_configs:
            w_type = conf.get("type")
            w_id = conf["id"]
            help_text = conf.get("help", "")
            
            if w_type == "text":
                w = widgets.Text(value=str(conf.get("value", "")), placeholder=str(conf.get("placeholder", "")), description=conf.get("description", ""), description_tooltip=help_text, layout=styles.LAYOUT_TEXT, style={'description_width': 'initial'})
            elif w_type == "dropdown":
                opts = conf.get("options", [])
                if "options_key" in conf:
                    opts = self.config.get(conf["options_key"], [])
                w = widgets.Dropdown(options=opts, value=conf.get("value", ""), description=conf.get("description", ""), description_tooltip=help_text, layout=styles.LAYOUT_TEXT, style={'description_width': 'initial'})
            elif w_type == "checkbox":
                w = widgets.Checkbox(value=conf.get("value", False), description=conf.get("description", ""), description_tooltip=help_text, layout=widgets.Layout(width='auto'), style={'description_width': 'initial'}, indent=False)
            elif w_type == "floatslider":
                w = widgets.FloatSlider(value=conf.get("value", 1.0), min=conf.get("min", 0.0), max=conf.get("max", 1.0), step=conf.get("step", 0.1), description=conf.get("description", ""), description_tooltip=help_text, layout=widgets.Layout(width='350px'), style={'description_width': 'initial'}, readout_format='.2f')
            elif w_type == "inttext":
                w = widgets.IntText(value=conf.get("value", 0), description=conf.get("description", ""), description_tooltip=help_text, layout=styles.LAYOUT_TEXT, style={'description_width': 'initial'})
            
            self._adv_widgets[w_id] = w
            
            # To make the help more visible, we can append a small HTML info icon next to the widget
            if help_text:
                help_icon = widgets.HTML(value=f"<span title='{help_text}' style='cursor:help; color:#3b82f6; margin-left:4px;'>&#9432;</span>")
                children.append(widgets.HBox([w, help_icon], layout=widgets.Layout(align_items='center')))
            else:
                children.append(w)
            
        if len(children) > 0:
            grid = widgets.GridBox(children, layout=widgets.Layout(grid_template_columns="repeat(auto-fit, minmax(320px, 1fr))", gap="10px", align_items="center"))
            self.adv_config_box.children = [grid]
            self.adv_config_box.layout.display = 'block'
        else:
            self.adv_config_box.children = [widgets.HTML("<i>No advanced configuration needed for this type/mode.</i>")]
            self.adv_config_box.layout.display = 'none'

    def _on_mode_change(self, _):
        self._build_slots()
        self._build_adv_config()
        
    def _on_source_change(self, _):
        self._build_slots()
        self._build_adv_config()

    def _get_common_kwargs(self):
        # We handle index_col, usecols, parse_dates universally in post_load
        return {}

    def _apply_common_post_load(self, df):
        if not isinstance(df, pd.DataFrame):
            return df
            
        usecols_w = self._adv_widgets.get("usecols")
        if usecols_w and usecols_w.value.strip():
            cols = [x.strip() for x in usecols_w.value.split(",")]
            # Filter cols if they exist
            # Try to handle indices if they passed e.g. "0, 1, 2"
            selected_cols = []
            for c in cols:
                try:
                    idx = int(c)
                    if idx < len(df.columns):
                        selected_cols.append(df.columns[idx])
                except ValueError:
                    if c in df.columns:
                        selected_cols.append(c)
            if selected_cols:
                df = df[selected_cols]
                
        idx_w = self._adv_widgets.get("index_col")
        if idx_w and idx_w.value.strip():
            val = idx_w.value.strip()
            try:
                val_int = int(val)
                if val_int < len(df.columns):
                    df = df.set_index(df.columns[val_int])
            except ValueError:
                if val in df.columns:
                    df = df.set_index(val)
                    
        pdates_w = self._adv_widgets.get("parse_dates")
        if pdates_w and pdates_w.value:
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except: pass
                    
        sample_w = self._adv_widgets.get("sample_frac")
        if sample_w:
            try:
                frac = float(sample_w.value)
                if 0.0 < frac < 1.0:
                    df = df.sample(frac=frac, random_state=42)
            except: pass
            
        return df

    def _load_csv(self, path: str, src_type: str = "csv"):
        header_w = self._adv_widgets.get("header")
        sep_w = self._adv_widgets.get("sep")
        enc_w = self._adv_widgets.get("enc")
        skiprows_w = self._adv_widgets.get("skiprows")
        nrows_w = self._adv_widgets.get("nrows")
        
        header_val = None if (header_w and header_w.value == "None") else (0 if (header_w and header_w.value == "0") else "infer")
        sep_val = sep_w.value if sep_w else ("," if src_type == "csv" else "\\t")
        enc_val = enc_w.value if enc_w else "utf-8"
        
        load_kwargs = {
            "sep": sep_val,
            "encoding": enc_val,
            "header": header_val,
        }
        
        if skiprows_w and skiprows_w.value.strip():
            try: load_kwargs["skiprows"] = int(skiprows_w.value.strip())
            except: pass
        if nrows_w and nrows_w.value.strip():
            try: load_kwargs["nrows"] = int(nrows_w.value.strip())
            except: pass
            
        load_kwargs.update(self._get_common_kwargs())
        
        if src_type == "clipboard":
            return pd.read_csv(io.StringIO(path), **load_kwargs)
            
        if str(path).startswith("http"):
            req = urllib.request.Request(path, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                return pd.read_csv(resp, **load_kwargs)
        return pd.read_csv(path, **load_kwargs)

    def _on_load(self, _):
        with self.out:
            clear_output()
            if self.clear_cb.value:
                self.state.data_raw.clear()
                self.state.data_types.clear()
                
            src_type = self.supported_types.get(self.source_dd.value, "")
            loaded = []
            
            for child in self.slots_box.children:
                try:
                    slot_name = child.children[0].value.strip(":")
                    path = ""
                    uploaded_file = None
                    
                    if len(child.children) == 2:
                        path = child.children[1].value
                    else:
                        inp_txt = child.children[1]
                        inp_file = child.children[3]
                        if inp_file.value:
                            uploaded_file = inp_file.value[0] if isinstance(inp_file.value, (list, tuple)) else list(inp_file.value.values())[0] if isinstance(inp_file.value, dict) else None
                        
                        if not uploaded_file:
                            path = inp_txt.value

                    if isinstance(path, str):
                        path = path.strip()
                    
                    if not path and not uploaded_file:
                        continue
                        
                    print(f"Loading {slot_name}...")
                    
                    if uploaded_file:
                        ext = os.path.splitext(uploaded_file['name'])[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                            tmp.write(uploaded_file['content'])
                            path = tmp.name
                    
                    if src_type == "sklearn":
                        df = _load_sklearn_dataset(path)
                    elif src_type == "clipboard":
                        df = self._load_csv(path, src_type)
                    elif src_type == "sqlite":
                        tbl_w = self._adv_widgets.get("table")
                        tbl_val = tbl_w.value.strip() if tbl_w else ""
                        with sqlite3.connect(path) as conn:
                            if tbl_val:
                                df = pd.read_sql(f"SELECT * FROM [{tbl_val}]", conn)
                            else:
                                tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
                                if not tables:
                                    raise ValueError("No tables found in SQLite database.")
                                df = pd.read_sql(f"SELECT * FROM [{tables[0]}]", conn)
                    elif src_type == "json":
                        orient_w = self._adv_widgets.get("orient")
                        lines_w = self._adv_widgets.get("lines")
                        orient_val = orient_w.value if orient_w else "columns"
                        lines_val = lines_w.value if lines_w else False
                        df = pd.read_json(path, orient=orient_val, lines=lines_val)
                    elif src_type in ["excel", "xls", "xlsx"]:
                        sheet_w = self._adv_widgets.get("sheet")
                        header_w = self._adv_widgets.get("header")
                        skiprows_w = self._adv_widgets.get("skiprows")
                        nrows_w = self._adv_widgets.get("nrows")
                        
                        sheet_val = sheet_w.value if sheet_w else 0
                        try:
                            sheet_val = int(sheet_val)
                        except ValueError:
                            pass
                        header_val = None if (header_w and header_w.value == "None") else (0 if (header_w and header_w.value == "0") else "infer")
                        
                        kwargs = {"sheet_name": sheet_val, "header": header_val}
                        if skiprows_w and skiprows_w.value.strip():
                            try: kwargs["skiprows"] = int(skiprows_w.value.strip())
                            except: pass
                        if nrows_w and nrows_w.value.strip():
                            try: kwargs["nrows"] = int(nrows_w.value.strip())
                            except: pass
                            
                        if str(path).startswith("http"):
                            req = urllib.request.Request(path, headers={"User-Agent": "Mozilla/5.0"})
                            with urllib.request.urlopen(req) as resp:
                                df = pd.read_excel(resp.read(), **kwargs)
                        else:
                            df = pd.read_excel(path, **kwargs)
                    elif src_type == "parquet":
                        df = pd.read_parquet(path)
                    elif src_type == "feather":
                        df = pd.read_feather(path)
                    elif src_type == "hdf5":
                        df = pd.read_hdf(path)
                    elif src_type == "orc":
                        df = pd.read_orc(path)
                    elif src_type in ["image", "text", "graph", "ontology", "zip"]:
                        if src_type == "text":
                            enc_w = self._adv_widgets.get("enc")
                            with open(path, "r", encoding=enc_w.value if enc_w else "utf-8") as f:
                                df = f.read()
                        elif src_type == "image":
                            mode_w = self._adv_widgets.get("mode")
                            resize_w = self._adv_widgets.get("resize")
                            df = Image.open(path)
                            if mode_w and mode_w.value:
                                df = df.convert(mode_w.value)
                            if resize_w and resize_w.value.strip():
                                try:
                                    w, h = map(int, resize_w.value.lower().split("x"))
                                    df = df.resize((w, h))
                                except: pass
                        elif src_type == "graph":
                            format_w = self._adv_widgets.get("format")
                            fmt = format_w.value if format_w else "auto"
                            if not nx:
                                raise ImportError("networkx is not installed")
                            if fmt == "graphml" or (fmt == "auto" and path.endswith(".graphml")):
                                df = nx.read_graphml(path)
                            elif fmt == "gml" or (fmt == "auto" and path.endswith(".gml")):
                                df = nx.read_gml(path)
                            elif fmt == "edgelist" or (fmt == "auto" and path.endswith(".txt")):
                                df = nx.read_edgelist(path)
                            elif fmt == "adjlist":
                                df = nx.read_adjlist(path)
                            else:
                                df = nx.read_graphml(path)
                        elif src_type == "ontology":
                            format_w = self._adv_widgets.get("format")
                            fmt = format_w.value if format_w else "auto"
                            if not rdflib:
                                raise ImportError("rdflib is not installed")
                            df = rdflib.Graph()
                            if fmt == "auto":
                                df.parse(path)
                            else:
                                df.parse(path, format=fmt)
                        elif src_type == "zip":
                            ctype_w = self._adv_widgets.get("content_type")
                            ctype = ctype_w.value if ctype_w else "csv"
                            df = []
                            with zipfile.ZipFile(path, 'r') as z:
                                for name in z.namelist():
                                    if name.endswith("/") or "__MACOSX" in name: continue
                                    with z.open(name) as f:
                                        if ctype == "csv":
                                            df.append(pd.read_csv(f))
                                        elif ctype == "json":
                                            df.append(pd.read_json(f))
                                        elif ctype == "text":
                                            df.append(f.read().decode('utf-8', errors='ignore'))
                                        elif ctype == "image":
                                            df.append(Image.open(f).copy())
                    else:
                        df = self._load_csv(path, src_type)
                        
                    df = self._apply_common_post_load(df)
                        
                    auto_split_val = "None"
                    if self._get_current_mode_id() == "single":
                        autosw = self._adv_widgets.get("auto_split")
                        if autosw and autosw.value != "None" and isinstance(df, pd.DataFrame):
                            auto_split_val = autosw.value
                            
                    if auto_split_val != "None":
                        try:
                            from sklearn.model_selection import train_test_split
                            tsw = self._adv_widgets.get("test_size")
                            test_size = tsw.value if tsw else 0.2
                            vsw = self._adv_widgets.get("val_size")
                            val_size = vsw.value if vsw else 0.1
                            stratw = self._adv_widgets.get("stratify")
                            strat = stratw.value.strip() if stratw and stratw.value.strip() else None
                            strat_data = df[strat] if strat and strat in df.columns else None
                            
                            if auto_split_val == "Train/Val/Test":
                                tr_val, te = train_test_split(df, test_size=test_size, stratify=strat_data, random_state=42)
                                strat_data_val = tr_val[strat] if strat and strat in tr_val.columns else None
                                val_ratio = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else 0.0
                                val_ratio = max(min(val_ratio, 0.99), 0.01)
                                tr, va = train_test_split(tr_val, test_size=val_ratio, stratify=strat_data_val, random_state=42)
                                
                                self.state.data_raw["Train"] = tr
                                self.state.data_raw["Validation"] = va
                                self.state.data_raw["Test"] = te
                                self.state.data_types["Train"] = "tabular"
                                self.state.data_types["Validation"] = "tabular"
                                self.state.data_types["Test"] = "tabular"
                                loaded.extend(["Train", "Validation", "Test"])
                                print(f" -> Auto-split Success! Train: {tr.shape}, Val: {va.shape}, Test: {te.shape}")
                            else:
                                tr, te = train_test_split(df, test_size=test_size, stratify=strat_data, random_state=42)
                                
                                self.state.data_raw["Train"] = tr
                                self.state.data_raw["Test"] = te
                                self.state.data_types["Train"] = "tabular"
                                self.state.data_types["Test"] = "tabular"
                                loaded.extend(["Train", "Test"])
                                print(f" -> Auto-split Success! Train: {tr.shape}, Test: {te.shape}")
                        except Exception as e:
                            print(f" -> Auto-split failed: {e}")
                            self.state.data_raw[slot_name] = df
                            self.state.data_types[slot_name] = "tabular" if src_type not in ["image", "text", "graph", "ontology", "zip"] else src_type
                            loaded.append(slot_name)
                    else:
                        self.state.data_raw[slot_name] = df
                        self.state.data_types[slot_name] = "tabular" if src_type not in ["image", "text", "graph", "ontology", "zip"] else src_type
                        loaded.append(slot_name)
                        
                        size_str = ""
                        if hasattr(df, "shape"): size_str = str(df.shape)
                        elif hasattr(df, "__len__"): size_str = str(len(df))
                        elif "Graph" in getattr(type(df), "__name__", ""): size_str = f"{len(df)} elements"
                        print(f" -> Success! Shape (or size): {size_str}")
                except Exception as e:
                    print(f" -> Error loading {slot_name}: {e}")
                    traceback.print_exc()

            if loaded:
                self.state.log_step("loader", "load_data", {"loaded_keys": loaded, "source": src_type})
                
                mode_id = self._get_current_mode_id()
                if mode_id in ["train_test", "train_val_test"]:
                    alignw = self._adv_widgets.get("align_cols")
                    if alignw and alignw.value and "Train" in self.state.data_raw:
                        tr_df = self.state.data_raw["Train"]
                        if isinstance(tr_df, pd.DataFrame):
                            train_cols = tr_df.columns
                            for key in ["Test", "Validation"]:
                                if key in self.state.data_raw and isinstance(self.state.data_raw[key], pd.DataFrame):
                                    df_k = self.state.data_raw[key]
                                    missing = set(train_cols) - set(df_k.columns)
                                    for c in missing:
                                        df_k[c] = pd.NA
                                    self.state.data_raw[key] = df_k[train_cols] 
                                    print(f"[{key}] Aligned columns to match Train.")
                                    
                if mode_id == "multi_source":
                    concatw = self._adv_widgets.get("concat")
                    if concatw and concatw.value:
                        dfs_to_concat = [self.state.data_raw[k] for k in loaded if isinstance(self.state.data_raw[k], pd.DataFrame)]
                        if dfs_to_concat:
                            combined = pd.concat(dfs_to_concat, ignore_index=True)
                            print(f"Concatenated {len(dfs_to_concat)} sources into single dataset shape {combined.shape}")
                            self.state.data_raw.clear()
                            self.state.data_types.clear()
                            self.state.data_raw["Data"] = combined
                            self.state.data_types["Data"] = "tabular"
                            loaded = ["Data"]

    def _on_preview(self, _):
        with self.out:
            clear_output()
            if not self.state.data_raw:
                print("No data loaded. Please load data first.")
                return
            for key, data in self.state.data_raw.items():
                if isinstance(data, pd.DataFrame):
                    display(HTML(f"<h4>{key} <span style='font-weight:normal;color:#666;'>({data.shape[0]} rows, {data.shape[1]} cols)</span></h4>"))
                    display(data.head())
                else:
                    size_str = ""
                    if hasattr(data, "shape"): size_str = str(data.shape)
                    elif hasattr(data, "__len__"): size_str = str(len(data))
                    elif rdflib and isinstance(data, rdflib.Graph): size_str = f"{len(data)} triples"
                    elif callable(getattr(data, "size", None)): size_str = str(data.size)
                    elif hasattr(data, "size"): size_str = str(data.size)
                    
                    display(HTML(f"<h4>{key} <span style='font-weight:normal;color:#666;'>(Type: {type(data).__name__}, Size: {size_str})</span></h4>"))
                    if isinstance(data, str):
                        print(data[:500] + ("..." if len(data) > 500 else ""))
                    elif isinstance(data, Image.Image):
                        w, h = data.size
                        max_w = 400
                        if w > max_w:
                            data = data.resize((max_w, int(h * max_w / w)))
                        display(data)
                    else:
                        print(repr(data)[:500] + ("..." if len(repr(data)) > 500 else ""))

def runner(state):
    loader = DataLoaderUI(state)
    if hasattr(loader, 'ui'):
        display(loader.ui)
    return loader

try:
    runner(state)
except NameError:
    pass