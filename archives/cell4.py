import sys, os, pathlib
_repo_root = str(pathlib.Path().absolute())
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import ipywidgets as widgets
from IPython.display import display, HTML

def load_config(state):
    state.config = {
        "loading": {
            "supported_types": {
                "CSV (.csv)":                       "csv",
                "TSV (.tsv)":                       "tsv",
                "Excel (.xlsx / .xls)":             "excel",
                "JSON (.json)":                     "json",
                "Parquet (.parquet)":               "parquet",
                "Feather (.feather)":               "feather",
                "HDF5 (.h5 / .hdf)":               "hdf5",
                "ORC (.orc)":                       "orc",
                "SQLite (.db / .sqlite)":           "sqlite",
                "Time Series (CSV + datetime col)": "timeseries",
                "Image (.png / .jpg / .bmp)":       "image",
                "Graph (.graphml / .gml)":          "graph",
                "Text file (.txt)":                 "text",
                "Ontology (.owl / .rdf / .ttl)":    "ontology",
                "ZIP batch (folder of files)":      "zip",
                "Clipboard (paste CSV text)":       "clipboard",
                "sklearn toy dataset":              "sklearn"
            },
            "file_accepts": {
                "csv": ".csv", "tsv": ".tsv", "excel": ".xlsx,.xls", "json": ".json",
                "parquet": ".parquet", "feather": ".feather", "hdf5": ".h5,.hdf5",
                "orc": ".orc", "sqlite": ".db,.sqlite", "timeseries": ".csv",
                "image": ".png,.jpg,.jpeg,.bmp", "graph": ".graphml,.gml,.txt",
                "text": ".txt", "ontology": ".owl,.rdf,.ttl", "zip": ".zip"
            },
            "adv_configs": {
                "csv": [
                    {"id": "sep", "type": "text", "description": "Separator:", "value": ",", "help": "Character used to separate fields (e.g. ',', ';', '\\t')"},
                    {"id": "enc", "type": "dropdown", "description": "Encoding:", "options_key": "encodings", "value": "utf-8", "help": "File encoding to read special characters correctly"},
                    {"id": "header", "type": "dropdown", "description": "Header:", "options": ["infer", "None", "0"], "value": "infer", "help": "Row number containing column names (infer automatically detects)"},
                    {"id": "skiprows", "type": "text", "description": "Skip rows:", "value": "", "placeholder": "0", "help": "Number of rows to skip at the beginning of the file"},
                    {"id": "nrows", "type": "text", "description": "Num rows:", "value": "", "placeholder": "All", "help": "Number of rows to read to limit memory usage"},
                    {"id": "sample_frac", "type": "floatslider", "description": "Sample %:", "value": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "help": "Fraction of data to randomly sample across the entire dataset"},
                    {"id": "index_col", "type": "text", "description": "Index Col:", "value": "", "placeholder": "col name or idx", "help": "Column to use as the row labels"},
                    {"id": "usecols", "type": "text", "description": "Use Cols:", "value": "", "placeholder": "col1,col2", "help": "Specific columns to load (comma-separated list)"},
                    {"id": "parse_dates", "type": "checkbox", "description": "Parse Dates", "value": False, "help": "Try parsing index or specific columns as datetimes"}
                ],
                "tsv": [
                    {"id": "sep", "type": "text", "description": "Separator:", "value": "\\t", "help": "Character used to separate fields (usually Tab for TSV)"},
                    {"id": "enc", "type": "dropdown", "description": "Encoding:", "options_key": "encodings", "value": "utf-8", "help": "Text encoding format"},
                    {"id": "header", "type": "dropdown", "description": "Header:", "options": ["infer", "None", "0"], "value": "infer", "help": "Row number for column names"},
                    {"id": "skiprows", "type": "text", "description": "Skip rows:", "value": "", "placeholder": "0", "help": "Number of lines to skip initially"},
                    {"id": "nrows", "type": "text", "description": "Num rows:", "value": "", "placeholder": "All", "help": "Maximum number of rows to read"},
                    {"id": "sample_frac", "type": "floatslider", "description": "Sample %:", "value": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "help": "Proportion of the dataset to include randomly"},
                    {"id": "index_col", "type": "text", "description": "Index Col:", "value": "", "placeholder": "col name or idx", "help": "Column to act as index"},
                    {"id": "usecols", "type": "text", "description": "Use Cols:", "value": "", "placeholder": "col1,col2", "help": "List of columns to read"},
                    {"id": "parse_dates", "type": "checkbox", "description": "Parse Dates", "value": False, "help": "Auto-parse date columns"}
                ],
                "clipboard": [
                    {"id": "sep", "type": "text", "description": "Separator:", "value": ",", "help": "Separator used in the pasted data"},
                    {"id": "enc", "type": "dropdown", "description": "Encoding:", "options_key": "encodings", "value": "utf-8", "help": "Expected text encoding"},
                    {"id": "header", "type": "dropdown", "description": "Header:", "options": ["infer", "None", "0"], "value": "infer", "help": "Header row inference"}
                ],
                "timeseries": [
                    {"id": "sep", "type": "text", "description": "Separator:", "value": ",", "help": "Value separator"},
                    {"id": "enc", "type": "dropdown", "description": "Encoding:", "options_key": "encodings", "value": "utf-8", "help": "Encoding"},
                    {"id": "header", "type": "dropdown", "description": "Header:", "options": ["infer", "None", "0"], "value": "infer", "help": "Row for header strings"},
                    {"id": "index_col", "type": "text", "description": "Time Col (Index):", "value": "0", "placeholder": "col name or idx", "help": "The column that contains the time components"},
                    {"id": "parse_dates", "type": "checkbox", "description": "Parse Dates", "value": True, "help": "Convert the Time Col to natively Datetime objects"}
                ],
                "excel": [
                    {"id": "sheet", "type": "text", "description": "Sheet:", "value": "0", "help": "Sheet name or zero-indexed sheet number to load"},
                    {"id": "header", "type": "dropdown", "description": "Header:", "options": ["0", "None"], "value": "0", "help": "Row containing the header"},
                    {"id": "skiprows", "type": "text", "description": "Skip rows:", "value": "", "placeholder": "0", "help": "Skip lines before the header"},
                    {"id": "nrows", "type": "text", "description": "Num rows:", "value": "", "placeholder": "All", "help": "Number of rows to parse"},
                    {"id": "sample_frac", "type": "floatslider", "description": "Sample %:", "value": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "help": "Fraction of data to randomly sample"},
                    {"id": "index_col", "type": "text", "description": "Index Col:", "value": "", "placeholder": "col name or idx", "help": "Column to use as the index"},
                    {"id": "usecols", "type": "text", "description": "Use Cols:", "value": "", "placeholder": "A:C, E", "help": "Excel columns to include e.g. 'A:C, E'"},
                    {"id": "parse_dates", "type": "checkbox", "description": "Parse Dates", "value": False, "help": "Attempt parsing dates natively"}
                ],
                "sqlite": [
                    {"id": "table", "type": "text", "description": "Table:", "value": "", "placeholder": "Auto (first table)", "help": "Name of the SQL table to load"},
                    {"id": "sample_frac", "type": "floatslider", "description": "Sample %:", "value": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "help": "Proportion of data to randomly select"},
                    {"id": "index_col", "type": "text", "description": "Index Col:", "value": "", "placeholder": "col name", "help": "Column to use as index"}
                ],
                "json": [
                    {"id": "orient", "type": "dropdown", "description": "Orient:", "options": ["records", "columns", "split", "index", "values"], "value": "columns", "help": "Expected JSON string format"},
                    {"id": "lines", "type": "checkbox", "description": "Lines (JSONL)", "value": False, "help": "Parse as a JSONLines file (one object per line)"},
                    {"id": "sample_frac", "type": "floatslider", "description": "Sample %:", "value": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "help": "Proportion of data to randomly select"}
                ],
                "parquet": [
                    {"id": "usecols", "type": "text", "description": "Use Cols:", "value": "", "placeholder": "col1,col2", "help": "Comma-separated list of column names"},
                    {"id": "sample_frac", "type": "floatslider", "description": "Sample %:", "value": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "help": "Proportion of data to randomly select"}
                ],
                "feather": [
                    {"id": "usecols", "type": "text", "description": "Use Cols:", "value": "", "placeholder": "col1,col2", "help": "Comma-separated list of column names"},
                    {"id": "sample_frac", "type": "floatslider", "description": "Sample %:", "value": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "help": "Proportion of data to randomly select"}
                ],
                "hdf5": [
                    {"id": "key", "type": "text", "description": "Key (Group):", "value": "", "placeholder": "auto", "help": "Group identifier within the HDF5 store"},
                    {"id": "sample_frac", "type": "floatslider", "description": "Sample %:", "value": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "help": "Proportion of data to randomly select"}
                ],
                "graph": [
                    {"id": "format", "type": "dropdown", "description": "Format:", "options_key": "graph_formats", "value": "auto", "help": "Underlying graph definition structure"}
                ],
                "image": [
                    {"id": "mode", "type": "dropdown", "description": "Color Mode:", "options_key": "image_modes", "value": "RGB", "help": "Colorspace target to use natively for loaded images"},
                    {"id": "resize", "type": "text", "description": "Resize (WxH):", "value": "", "placeholder": "e.g. 224x224", "help": "Rescale size to standardize dimensions on load"}
                ],
                "text": [
                    {"id": "enc", "type": "dropdown", "description": "Encoding:", "options_key": "encodings", "value": "utf-8", "help": "Text reading format like utf-8 or latin-1"}
                ],
                "ontology": [
                    {"id": "format", "type": "dropdown", "description": "Format:", "options": ["auto", "xml", "n3", "turtle", "nt", "pretty-xml", "trix"], "value": "auto", "help": "Serialization standard to apply initially"}
                ],
                "zip": [
                    {"id": "content_type", "type": "dropdown", "description": "Content Type:", "options_key": "zip_formats", "value": "csv", "help": "Data type expected inside the batch"},
                    {"id": "sample_frac", "type": "floatslider", "description": "Sample %:", "value": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "help": "Proportion of data to randomly select"}
                ]
            },
            "mode_configs": {
                "single": [
                    {"id": "auto_split", "type": "dropdown", "description": "Auto Split:", "options": ["None", "Train/Test", "Train/Val/Test"], "value": "None", "help": "Automatically split the single dataset into subsets on load"},
                    {"id": "test_size", "type": "floatslider", "description": "Test Size %:", "value": 0.2, "min": 0.01, "max": 0.99, "step": 0.01, "help": "Fraction of the total dataset to allocate for test purposes"},
                    {"id": "val_size", "type": "floatslider", "description": "Val Size %:", "value": 0.1, "min": 0.00, "max": 0.99, "step": 0.01, "help": "Fraction of the total dataset to allocate for validation purposes (if Train/Val/Test is selected)"},
                    {"id": "stratify", "type": "text", "description": "Stratify Col:", "value": "", "placeholder": "column name", "help": "Target column for stratified sampling to maintain class proportions"}
                ],
                "train_test": [
                    {"id": "align_cols", "type": "checkbox", "description": "Align Columns", "value": False, "help": "Ensure Test data has exactly the same columns in the same order as Train data"}
                ],
                "train_val_test": [
                    {"id": "align_cols", "type": "checkbox", "description": "Align Columns", "value": False, "help": "Ensure Test and Validation data has exactly the same columns in the same order as Train data"}
                ],
                "multi_source": [
                    {"id": "concat", "type": "checkbox", "description": "Auto-Concat", "value": False, "help": "Try automatically concatenating matching sources vertically"}
                ],
                "custom": []
            },
            "modes": [
                {"label": "Single file", "value": "single", "slots": ["Data"]},
                {"label": "Train / Test split", "value": "train_test", "slots": ["Train", "Test"]},
                {"label": "Train / Validation / Test split", "value": "train_val_test", "slots": ["Train", "Validation", "Test"]},
                {"label": "Multi-source (heterogeneous)", "value": "multi_source", "slots": ["Source A", "Source B", "Source C", "Source D"]},
                {"label": "Autres sources (autant que souhaite)", "value": "custom", "slots": ["Sources"]}
            ],
            "sklearn_datasets": ["iris", "wine", "breast_cancer", "diabetes", "digits", "california_housing"],
            "encodings": ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"],
            "ts_freqs": ["(auto)", "D", "W", "MS", "QS", "AS", "H", "T"],
            "image_modes": ["RGB", "RGBA", "L", "1"],
            "graph_formats": ["auto", "graphml", "gml", "edgelist", "adjlist"],
            "zip_formats": ["csv", "image", "text", "json"]
        },
        "feature_engineering": {
            "math_operations": ['+', '-', '*', '/', 'log(A)', 'exp(A)', 'sqrt(A)', 'A^2', 'A^B/C', 'Abs(A)', 'Modulo'],
            "text_operations": ['Lowercase', 'Uppercase', 'Length', 'Extract Regex', 'Replace', 'Split & Keep N'],
            "date_operations": ['Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'Minute', 'IsWeekend'],
            "binning_strategies": ['Equal Width (Cut)', 'Equal Frequency (Qcut)', 'Custom Edges'],
            "viz_types": ['auto', 'scatter', 'line', 'bar', 'box', 'violin', 'hist', 'kde'],
            "manage_actions": ['Set Type (Meta)', 'Duplicate', 'Delete']
        },
        "eda": {
            "univariate_plots": ['auto', 'hist', 'kde', 'box', 'violin', 'bar', 'pie'],
            "bivariate_plots": ['auto', 'scatter', 'hexbin', 'hist2d', 'kde', 'box', 'violin', 'strip', 'swarm', 'heatmap', 'stacked_bar', 'pie'],
            "multivariate_analysis": ['Correlation Matrix', 'Pairplot'],
            "correlation_methods": ['pearson', 'spearman', 'kendall'],
            "palettes": ['Set1', 'Set2', 'Set3', 'viridis', 'plasma', 'coolwarm', 'husl'],
            "ts_plots": ['Line', 'Scatter', 'Area', 'Box (by month/year)', 'Autocorrelation']
        },
        "domain_tasks": {
            "ontology": {
                "tasks": ["Consistency Checking", "Knowledge Graph Completion", "Entity Linking", "Semantic Reasoning"],
                "inference": ["RDFS", "OWL-DL", "OWL-Full", "Custom Rule System"]
            },
            "nlp": {
                "tasks": ["Text Classification", "Sentiment Analysis", "Named Entity Recognition (NER)", "Summarization", "Question Answering"]
            },
            "computer_vision": {
                "tasks": ["Image Classification", "Object Detection", "Image Segmentation", "Face Recognition"]
            }
        },
        "domains": [
            {"label": "Binary Classification", "value": "classification_binary", "task": "classification", "subtask": "binary"},
            {"label": "Multiclass Classification", "value": "classification_multiclass", "task": "classification", "subtask": "multiclass"},
            {"label": "Regression", "value": "regression_continuous", "task": "regression", "subtask": "continuous"},
            {"label": "Clustering", "value": "clustering", "task": "clustering", "subtask": None},
            {"label": "Time Series", "value": "timeseries", "task": "timeseries", "subtask": None},
            {"label": "NLP", "value": "nlp", "task": "nlp", "subtask": None},
            {"label": "Computer Vision", "value": "computer_vision", "task": "computer_vision", "subtask": None},
            {"label": "Graph Analysis", "value": "graph", "task": "graph", "subtask": None},
            {"label": "Ontology / Knowledge Graph", "value": "ontology", "task": "ontology", "subtask": None}
        ],
        "balancing": [
            {"label": "None", "value": "none"},
            {"label": "Class Weights (Model Level)", "value": "class_weights"},
            {"label": "Oversampling (imbalanced-learn)", "value": "oversample"},
            {"label": "Undersampling (imbalanced-learn)", "value": "undersample"},
            {"label": "SMOTE (imbalanced-learn)", "value": "smote"}
        ],
        "splitting": {
            "strategies": [
                {"label": "Random Split", "value": "random"},
                {"label": "Stratified Split", "value": "stratified"},
                {"label": "Temporal/Sequential", "value": "sequential"}
            ],
            "default_test_size": 0.2
        },
        "cleaning": {
            "missing": [
                {
                    "label": "Do nothing",
                    "value": "none",
                    "imports": [],
                    "params": {},
                    "code": "# no-op — column left unchanged"
                },
                {
                    "label": "Drop rows with NaN",
                    "value": "drop_rows",
                    "imports": [],
                    "params": {},
                    "code": "df.dropna(subset=[col], inplace=True)"
                },
                {
                    "label": "Drop column",
                    "value": "drop_cols",
                    "imports": [],
                    "params": {},
                    "code": "df.drop(columns=[col], inplace=True)"
                },
                {
                    "label": "Replace: Mean",
                    "value": "mean",
                    "imports": [],
                    "params": {},
                    "code": "df[col].fillna(df[col].mean(), inplace=True)"
                },
                {
                    "label": "Replace: Median",
                    "value": "median",
                    "imports": [],
                    "params": {},
                    "code": "df[col].fillna(df[col].median(), inplace=True)"
                },
                {
                    "label": "Replace: Mode",
                    "value": "mode",
                    "imports": [],
                    "params": {},
                    "code": "df[col].fillna(df[col].mode()[0], inplace=True)"
                },
                {
                    "label": "Replace: 0",
                    "value": "zero",
                    "imports": [],
                    "params": {},
                    "code": "df[col].fillna(0, inplace=True)"
                },
                {
                    "label": "Replace: Custom constant",
                    "value": "constant",
                    "imports": [],
                    "params": {"fill_value": "MISSING"},
                    "code": "df[col].fillna(params['fill_value'], inplace=True)"
                },
                {
                    "label": "Forward fill (ffill)",
                    "value": "ffill",
                    "imports": [],
                    "params": {"limit": None},
                    "code": "df[col].fillna(method='ffill', limit=params.get('limit'), inplace=True)"
                },
                {
                    "label": "Backward fill (bfill)",
                    "value": "bfill",
                    "imports": [],
                    "params": {"limit": None},
                    "code": "df[col].fillna(method='bfill', limit=params.get('limit'), inplace=True)"
                },
                {
                    "label": "Interpolate (linear)",
                    "value": "interpolate_linear",
                    "imports": [],
                    "params": {},
                    "code": "df[col] = df[col].interpolate(method='linear')"
                },
                {
                    "label": "Interpolate (time)",
                    "value": "interpolate_time",
                    "imports": [],
                    "params": {},
                    "code": "df[col] = df[col].interpolate(method='time')"
                },
                {
                    "label": "KNN Imputer",
                    "value": "knn",
                    "imports": ["from sklearn.impute import KNNImputer"],
                    "params": {"n_neighbors": 5, "weights": "uniform"},
                    "code": (
                        "from sklearn.impute import KNNImputer\n"
                        "imp = KNNImputer(n_neighbors=params['n_neighbors'], weights=params['weights'])\n"
                        "df[[col]] = imp.fit_transform(df[[col]])"
                    )
                },
                {
                    "label": "Iterative Imputer (MICE)",
                    "value": "mice",
                    "imports": ["from sklearn.experimental import enable_iterative_imputer", "from sklearn.impute import IterativeImputer"],
                    "params": {"max_iter": 10, "random_state": 42},
                    "code": (
                        "from sklearn.experimental import enable_iterative_imputer\n"
                        "from sklearn.impute import IterativeImputer\n"
                        "imp = IterativeImputer(max_iter=params['max_iter'], random_state=params['random_state'])\n"
                        "df[[col]] = imp.fit_transform(df[[col]])"
                    )
                },
                {
                    "label": "SimpleImputer (most_frequent)",
                    "value": "simple_most_frequent",
                    "imports": ["from sklearn.impute import SimpleImputer"],
                    "params": {},
                    "code": (
                        "from sklearn.impute import SimpleImputer\n"
                        "imp = SimpleImputer(strategy='most_frequent')\n"
                        "df[[col]] = imp.fit_transform(df[[col]])"
                    )
                },
                {
                    "label": "Flag missing then fill 0",
                    "value": "flag_and_zero",
                    "imports": [],
                    "params": {},
                    "code": (
                        "df[col + '_was_missing'] = df[col].isna().astype(int)\n"
                        "df[col].fillna(0, inplace=True)"
                    )
                },
            ],
            "outliers": [
                {
                    "label": "Do nothing",
                    "value": "none",
                    "imports": [],
                    "params": {},
                    "code": "# no-op"
                },
                {
                    "label": "Clip IQR (1.5×)",
                    "value": "clip_iqr",
                    "imports": [],
                    "params": {"factor": 1.5},
                    "code": (
                        "Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)\n"
                        "IQR = Q3 - Q1\n"
                        "lo, hi = Q1 - params['factor'] * IQR, Q3 + params['factor'] * IQR\n"
                        "df[col] = df[col].clip(lo, hi)"
                    )
                },
                {
                    "label": "Clip percentile 1%–99%",
                    "value": "clip_percentile",
                    "imports": [],
                    "params": {"lower_pct": 1, "upper_pct": 99},
                    "code": (
                        "lo = df[col].quantile(params['lower_pct'] / 100)\n"
                        "hi = df[col].quantile(params['upper_pct'] / 100)\n"
                        "df[col] = df[col].clip(lo, hi)"
                    )
                },
                {
                    "label": "Drop rows — IQR",
                    "value": "drop_iqr",
                    "imports": [],
                    "params": {"factor": 1.5},
                    "code": (
                        "Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)\n"
                        "IQR = Q3 - Q1\n"
                        "mask = (df[col] >= Q1 - params['factor'] * IQR) & (df[col] <= Q3 + params['factor'] * IQR)\n"
                        "df = df[mask]"
                    )
                },
                {
                    "label": "Drop rows — Z-score",
                    "value": "drop_zscore",
                    "imports": ["import numpy as np"],
                    "params": {"threshold": 3.0},
                    "code": (
                        "import numpy as np\n"
                        "z = np.abs((df[col] - df[col].mean()) / df[col].std())\n"
                        "df = df[z < params['threshold']]"
                    )
                },
                {
                    "label": "Winsorize",
                    "value": "winsorize",
                    "imports": ["from scipy.stats.mstats import winsorize"],
                    "params": {"limits": [0.05, 0.05]},
                    "code": (
                        "from scipy.stats.mstats import winsorize\n"
                        "df[col] = winsorize(df[col], limits=params['limits'])"
                    )
                },
                {
                    "label": "Isolation Forest (flag)",
                    "value": "isolation_forest",
                    "imports": ["from sklearn.ensemble import IsolationForest"],
                    "params": {"contamination": 0.05, "random_state": 42},
                    "code": (
                        "from sklearn.ensemble import IsolationForest\n"
                        "iso = IsolationForest(contamination=params['contamination'], random_state=params['random_state'])\n"
                        "df[col + '_outlier'] = iso.fit_predict(df[[col]])"
                    )
                },
                {
                    "label": "Local Outlier Factor (flag)",
                    "value": "lof",
                    "imports": ["from sklearn.neighbors import LocalOutlierFactor"],
                    "params": {"n_neighbors": 20, "contamination": 0.05},
                    "code": (
                        "from sklearn.neighbors import LocalOutlierFactor\n"
                        "lof = LocalOutlierFactor(n_neighbors=params['n_neighbors'], contamination=params['contamination'])\n"
                        "df[col + '_outlier'] = lof.fit_predict(df[[col]])"
                    )
                },
            ],
            "duplicates": [
                {
                    "label": "Drop exact duplicates (keep first)",
                    "value": "drop_first",
                    "imports": [],
                    "params": {},
                    "code": "df.drop_duplicates(keep='first', inplace=True)"
                },
                {
                    "label": "Drop exact duplicates (keep last)",
                    "value": "drop_last",
                    "imports": [],
                    "params": {},
                    "code": "df.drop_duplicates(keep='last', inplace=True)"
                },
                {
                    "label": "Drop exact duplicates (drop all)",
                    "value": "drop_all",
                    "imports": [],
                    "params": {},
                    "code": "df.drop_duplicates(keep=False, inplace=True)"
                },
                {
                    "label": "Flag duplicates column",
                    "value": "flag",
                    "imports": [],
                    "params": {},
                    "code": "df['is_duplicate'] = df.duplicated(keep='first').astype(int)"
                },
            ],
            "type_cast": [
                {
                    "label": "Cast to int",
                    "value": "to_int",
                    "imports": [],
                    "params": {},
                    "code": "df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')"
                },
                {
                    "label": "Cast to float",
                    "value": "to_float",
                    "imports": [],
                    "params": {},
                    "code": "df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)"
                },
                {
                    "label": "Cast to string",
                    "value": "to_str",
                    "imports": [],
                    "params": {},
                    "code": "df[col] = df[col].astype(str)"
                },
                {
                    "label": "Cast to datetime",
                    "value": "to_datetime",
                    "imports": [],
                    "params": {"format": None, "dayfirst": False},
                    "code": (
                        "df[col] = pd.to_datetime(df[col], "
                        "format=params.get('format'), dayfirst=params.get('dayfirst', False), errors='coerce')"
                    )
                },
                {
                    "label": "Cast to category",
                    "value": "to_category",
                    "imports": [],
                    "params": {},
                    "code": "df[col] = df[col].astype('category')"
                },
                {
                    "label": "Cast to boolean",
                    "value": "to_bool",
                    "imports": [],
                    "params": {},
                    "code": "df[col] = df[col].astype(bool)"
                },
            ],
            "text_cleaning": [
                {
                    "label": "Lowercase",
                    "value": "lowercase",
                    "imports": [],
                    "params": {},
                    "code": "df[col] = df[col].str.lower()"
                },
                {
                    "label": "Strip whitespace",
                    "value": "strip",
                    "imports": [],
                    "params": {},
                    "code": "df[col] = df[col].str.strip()"
                },
                {
                    "label": "Remove punctuation",
                    "value": "remove_punct",
                    "imports": ["import re"],
                    "params": {},
                    "code": "import re\ndf[col] = df[col].apply(lambda x: re.sub(r'[^\\w\\s]', '', str(x)))"
                },
                {
                    "label": "Remove digits",
                    "value": "remove_digits",
                    "imports": ["import re"],
                    "params": {},
                    "code": "import re\ndf[col] = df[col].apply(lambda x: re.sub(r'\\d+', '', str(x)))"
                },
                {
                    "label": "Remove stopwords (English)",
                    "value": "remove_stopwords_en",
                    "imports": ["from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS"],
                    "params": {},
                    "code": (
                        "from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS\n"
                        "df[col] = df[col].apply(lambda x: ' '.join("
                        "w for w in str(x).split() if w.lower() not in ENGLISH_STOP_WORDS))"
                    )
                },
                {
                    "label": "Remove stopwords (French — NLTK)",
                    "value": "remove_stopwords_fr",
                    "imports": ["import nltk", "from nltk.corpus import stopwords"],
                    "params": {},
                    "code": (
                        "import nltk; nltk.download('stopwords', quiet=True)\n"
                        "from nltk.corpus import stopwords\n"
                        "_sw = set(stopwords.words('french'))\n"
                        "df[col] = df[col].apply(lambda x: ' '.join(w for w in str(x).split() if w.lower() not in _sw))"
                    )
                },
                {
                    "label": "Stemming (English — NLTK)",
                    "value": "stem_en",
                    "imports": ["import nltk", "from nltk.stem import PorterStemmer"],
                    "params": {},
                    "code": (
                        "import nltk; nltk.download('punkt', quiet=True)\n"
                        "from nltk.stem import PorterStemmer\n"
                        "_ps = PorterStemmer()\n"
                        "df[col] = df[col].apply(lambda x: ' '.join(_ps.stem(w) for w in str(x).split()))"
                    )
                },
                {
                    "label": "Lemmatization (English — spaCy)",
                    "value": "lemma_en_spacy",
                    "imports": ["import spacy"],
                    "params": {"model": "en_core_web_sm"},
                    "code": (
                        "import spacy\n"
                        "_nlp = spacy.load(params.get('model', 'en_core_web_sm'))\n"
                        "df[col] = df[col].apply(lambda x: ' '.join(t.lemma_ for t in _nlp(str(x))))"
                    )
                },
                {
                    "label": "Regex replace",
                    "value": "regex_replace",
                    "imports": ["import re"],
                    "params": {"pattern": r"\\s+", "replacement": " "},
                    "code": (
                        "import re\n"
                        "df[col] = df[col].apply(lambda x: re.sub(params['pattern'], params['replacement'], str(x)))"
                    )
                },
            ],
        },
        "encoding": {
            "tabular": {
                "numeric": [
                    {
                        "label": "Passthrough (none)",
                        "value": "none",
                        "imports": [],
                        "params": {},
                        "code": "# passthrough — no transformation"
                    },
                    {
                        "label": "Standardisation (Z-score)",
                        "value": "std",
                        "imports": ["from sklearn.preprocessing import StandardScaler"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import StandardScaler\n"
                            "_sc = StandardScaler()\n"
                            "df[[col]] = _sc.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "Min-Max Scaling [0,1]",
                        "value": "minmax",
                        "imports": ["from sklearn.preprocessing import MinMaxScaler"],
                        "params": {"feature_range": [0, 1]},
                        "code": (
                            "from sklearn.preprocessing import MinMaxScaler\n"
                            "_sc = MinMaxScaler(feature_range=tuple(params.get('feature_range', [0, 1])))\n"
                            "df[[col]] = _sc.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "Robust Scaler (median/IQR)",
                        "value": "robust",
                        "imports": ["from sklearn.preprocessing import RobustScaler"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import RobustScaler\n"
                            "_sc = RobustScaler()\n"
                            "df[[col]] = _sc.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "MaxAbs Scaler [-1,1]",
                        "value": "maxabs",
                        "imports": ["from sklearn.preprocessing import MaxAbsScaler"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import MaxAbsScaler\n"
                            "_sc = MaxAbsScaler()\n"
                            "df[[col]] = _sc.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "Power Transform (Yeo-Johnson)",
                        "value": "power_yeo",
                        "imports": ["from sklearn.preprocessing import PowerTransformer"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import PowerTransformer\n"
                            "_pt = PowerTransformer(method='yeo-johnson')\n"
                            "df[[col]] = _pt.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "Power Transform (Box-Cox)",
                        "value": "power_boxcox",
                        "imports": ["from sklearn.preprocessing import PowerTransformer"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import PowerTransformer\n"
                            "_pt = PowerTransformer(method='box-cox')  # requires positive values\n"
                            "df[[col]] = _pt.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "Quantile Transform (uniform)",
                        "value": "quantile_uniform",
                        "imports": ["from sklearn.preprocessing import QuantileTransformer"],
                        "params": {"n_quantiles": 1000},
                        "code": (
                            "from sklearn.preprocessing import QuantileTransformer\n"
                            "_qt = QuantileTransformer(n_quantiles=params.get('n_quantiles', 1000), output_distribution='uniform')\n"
                            "df[[col]] = _qt.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "Quantile Transform (normal)",
                        "value": "quantile_normal",
                        "imports": ["from sklearn.preprocessing import QuantileTransformer"],
                        "params": {"n_quantiles": 1000},
                        "code": (
                            "from sklearn.preprocessing import QuantileTransformer\n"
                            "_qt = QuantileTransformer(n_quantiles=params.get('n_quantiles', 1000), output_distribution='normal')\n"
                            "df[[col]] = _qt.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "Log1p transform",
                        "value": "log1p",
                        "imports": ["import numpy as np"],
                        "params": {},
                        "code": "import numpy as np\ndf[col] = np.log1p(df[col])"
                    },
                    {
                        "label": "Square root transform",
                        "value": "sqrt",
                        "imports": ["import numpy as np"],
                        "params": {},
                        "code": "import numpy as np\ndf[col] = np.sqrt(df[col].clip(0))"
                    },
                    {
                        "label": "L1 Normalizer (row-wise)",
                        "value": "normalizer_l1",
                        "imports": ["from sklearn.preprocessing import Normalizer"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import Normalizer\n"
                            "_n = Normalizer(norm='l1')\n"
                            "df[[col]] = _n.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "L2 Normalizer (row-wise)",
                        "value": "normalizer_l2",
                        "imports": ["from sklearn.preprocessing import Normalizer"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import Normalizer\n"
                            "_n = Normalizer(norm='l2')\n"
                            "df[[col]] = _n.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "Binarize",
                        "value": "binarize",
                        "imports": ["from sklearn.preprocessing import Binarizer"],
                        "params": {"threshold": 0.0},
                        "code": (
                            "from sklearn.preprocessing import Binarizer\n"
                            "_b = Binarizer(threshold=params.get('threshold', 0.0))\n"
                            "df[[col]] = _b.fit_transform(df[[col]])"
                        )
                    },
                    {
                        "label": "Polynomial Features (degree 2)",
                        "value": "poly2",
                        "imports": ["from sklearn.preprocessing import PolynomialFeatures", "import pandas as pd"],
                        "params": {"degree": 2, "include_bias": False},
                        "code": (
                            "from sklearn.preprocessing import PolynomialFeatures\n"
                            "import pandas as pd\n"
                            "_pf = PolynomialFeatures(degree=params.get('degree', 2), include_bias=params.get('include_bias', False))\n"
                            "_arr = _pf.fit_transform(df[[col]])\n"
                            "_names = _pf.get_feature_names_out([col])\n"
                            "df = df.join(pd.DataFrame(_arr, columns=_names, index=df.index))"
                        )
                    },
                    {
                        "label": "Drop column",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "df.drop(columns=[col], inplace=True)"
                    },
                ],
                "categorical": [
                    {
                        "label": "One-Hot Encoding (pandas)",
                        "value": "onehot_pandas",
                        "imports": ["import pandas as pd"],
                        "params": {"prefix": None, "drop_first": False},
                        "code": (
                            "import pandas as pd\n"
                            "_dummies = pd.get_dummies(df[col], prefix=params.get('prefix') or col, "
                            "drop_first=params.get('drop_first', False))\n"
                            "df = df.drop(columns=[col]).join(_dummies)"
                        )
                    },
                    {
                        "label": "One-Hot Encoding (sklearn)",
                        "value": "onehot_sklearn",
                        "imports": ["from sklearn.preprocessing import OneHotEncoder", "import pandas as pd"],
                        "params": {"drop": None, "sparse_output": False},
                        "code": (
                            "from sklearn.preprocessing import OneHotEncoder\n"
                            "import pandas as pd\n"
                            "_ohe = OneHotEncoder(drop=params.get('drop'), sparse_output=False)\n"
                            "_arr = _ohe.fit_transform(df[[col]])\n"
                            "_cols = _ohe.get_feature_names_out([col])\n"
                            "df = df.drop(columns=[col]).join(pd.DataFrame(_arr, columns=_cols, index=df.index))"
                        )
                    },
                    {
                        "label": "Label Encoding",
                        "value": "label",
                        "imports": ["from sklearn.preprocessing import LabelEncoder"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import LabelEncoder\n"
                            "_le = LabelEncoder()\n"
                            "df[col] = _le.fit_transform(df[col].astype(str))"
                        )
                    },
                    {
                        "label": "Ordinal Encoding",
                        "value": "ordinal",
                        "imports": ["from sklearn.preprocessing import OrdinalEncoder"],
                        "params": {"categories": "auto"},
                        "code": (
                            "from sklearn.preprocessing import OrdinalEncoder\n"
                            "_oe = OrdinalEncoder(categories=params.get('categories', 'auto'), handle_unknown='use_encoded_value', unknown_value=-1)\n"
                            "df[[col]] = _oe.fit_transform(df[[col]].astype(str))"
                        )
                    },
                    {
                        "label": "Target Encoding",
                        "value": "target",
                        "imports": ["from sklearn.preprocessing import TargetEncoder"],
                        "params": {"target_type": "auto", "smooth": "auto"},
                        "code": (
                            "from sklearn.preprocessing import TargetEncoder\n"
                            "_te = TargetEncoder(target_type=params.get('target_type','auto'), smooth=params.get('smooth','auto'))\n"
                            "df[[col]] = _te.fit_transform(df[[col]], y)"
                        )
                    },
                    {
                        "label": "Frequency Encoding",
                        "value": "freq",
                        "imports": [],
                        "params": {"normalize": True},
                        "code": (
                            "_freq = df[col].value_counts(normalize=params.get('normalize', True))\n"
                            "df[col] = df[col].map(_freq)"
                        )
                    },
                    {
                        "label": "Count Encoding",
                        "value": "count",
                        "imports": [],
                        "params": {},
                        "code": (
                            "_cnt = df[col].value_counts()\n"
                            "df[col] = df[col].map(_cnt)"
                        )
                    },
                    {
                        "label": "Binary Encoding (category_encoders)",
                        "value": "binary",
                        "imports": ["import category_encoders as ce"],
                        "params": {},
                        "code": (
                            "import category_encoders as ce\n"
                            "_enc = ce.BinaryEncoder(cols=[col])\n"
                            "df = _enc.fit_transform(df)"
                        )
                    },
                    {
                        "label": "BaseN Encoding (category_encoders)",
                        "value": "basen",
                        "imports": ["import category_encoders as ce"],
                        "params": {"base": 2},
                        "code": (
                            "import category_encoders as ce\n"
                            "_enc = ce.BaseNEncoder(cols=[col], base=params.get('base', 2))\n"
                            "df = _enc.fit_transform(df)"
                        )
                    },
                    {
                        "label": "Hashing Encoding",
                        "value": "hashing",
                        "imports": ["from sklearn.feature_extraction import FeatureHasher"],
                        "params": {"n_features": 8},
                        "code": (
                            "from sklearn.feature_extraction import FeatureHasher\n"
                            "_fh = FeatureHasher(n_features=params.get('n_features', 8), input_type='string')\n"
                            "_arr = _fh.transform(df[col].astype(str).apply(lambda x: [x]))\n"
                            "import pandas as pd\n"
                            "_hcols = [f'{col}_hash_{i}' for i in range(params.get('n_features', 8))]\n"
                            "df = df.drop(columns=[col]).join(pd.DataFrame(_arr.toarray(), columns=_hcols, index=df.index))"
                        )
                    },
                    {
                        "label": "Leave-One-Out Encoding (category_encoders)",
                        "value": "loo",
                        "imports": ["import category_encoders as ce"],
                        "params": {"sigma": None},
                        "code": (
                            "import category_encoders as ce\n"
                            "_enc = ce.LeaveOneOutEncoder(cols=[col], sigma=params.get('sigma'))\n"
                            "df[[col]] = _enc.fit_transform(df[[col]], y)"
                        )
                    },
                    {
                        "label": "Drop column",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "df.drop(columns=[col], inplace=True)"
                    },
                ],
                "binary": [
                    {
                        "label": "Label Encoding (0/1)",
                        "value": "label",
                        "imports": ["from sklearn.preprocessing import LabelEncoder"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import LabelEncoder\n"
                            "_le = LabelEncoder()\n"
                            "df[col] = _le.fit_transform(df[col].astype(str))"
                        )
                    },
                    {
                        "label": "Map True/False → 1/0",
                        "value": "bool_map",
                        "imports": [],
                        "params": {},
                        "code": "df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0, 'Yes': 1, 'No': 0, 'yes': 1, 'no': 0})"
                    },
                    {
                        "label": "Drop column",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "df.drop(columns=[col], inplace=True)"
                    },
                ],
                "datetime": [
                    {
                        "label": "Extract Year / Month / Day",
                        "value": "extract_ymd",
                        "imports": [],
                        "params": {},
                        "code": (
                            "_dt = pd.to_datetime(df[col], errors='coerce')\n"
                            "df[col + '_year']  = _dt.dt.year\n"
                            "df[col + '_month'] = _dt.dt.month\n"
                            "df[col + '_day']   = _dt.dt.day\n"
                            "df.drop(columns=[col], inplace=True)"
                        )
                    },
                    {
                        "label": "Extract full date features",
                        "value": "extract_full",
                        "imports": [],
                        "params": {},
                        "code": (
                            "_dt = pd.to_datetime(df[col], errors='coerce')\n"
                            "df[col + '_year']        = _dt.dt.year\n"
                            "df[col + '_month']       = _dt.dt.month\n"
                            "df[col + '_day']         = _dt.dt.day\n"
                            "df[col + '_hour']        = _dt.dt.hour\n"
                            "df[col + '_minute']      = _dt.dt.minute\n"
                            "df[col + '_weekday']     = _dt.dt.weekday\n"
                            "df[col + '_weekofyear']  = _dt.dt.isocalendar().week.astype(int)\n"
                            "df[col + '_is_weekend']  = (_dt.dt.weekday >= 5).astype(int)\n"
                            "df.drop(columns=[col], inplace=True)"
                        )
                    },
                    {
                        "label": "Epoch (Unix timestamp)",
                        "value": "epoch",
                        "imports": [],
                        "params": {},
                        "code": "df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64') // 10**9"
                    },
                    {
                        "label": "Cyclical sin/cos encoding (month)",
                        "value": "cyclical_month",
                        "imports": ["import numpy as np"],
                        "params": {},
                        "code": (
                            "import numpy as np\n"
                            "_m = pd.to_datetime(df[col], errors='coerce').dt.month\n"
                            "df[col + '_sin'] = np.sin(2 * np.pi * _m / 12)\n"
                            "df[col + '_cos'] = np.cos(2 * np.pi * _m / 12)\n"
                            "df.drop(columns=[col], inplace=True)"
                        )
                    },
                    {
                        "label": "Drop column",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "df.drop(columns=[col], inplace=True)"
                    },
                ],
                "text": [
                    {
                        "label": "TF-IDF (50 features)",
                        "value": "tfidf_50",
                        "imports": ["from sklearn.feature_extraction.text import TfidfVectorizer", "import pandas as pd"],
                        "params": {"max_features": 50, "ngram_range": [1, 1], "min_df": 1},
                        "code": (
                            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
                            "import pandas as pd\n"
                            "_tv = TfidfVectorizer(max_features=params.get('max_features', 50), "
                            "ngram_range=tuple(params.get('ngram_range', [1,1])), min_df=params.get('min_df', 1))\n"
                            "_arr = _tv.fit_transform(df[col].astype(str))\n"
                            "_names = [f'{col}_tfidf_{n}' for n in _tv.get_feature_names_out()]\n"
                            "df = df.drop(columns=[col]).join(pd.DataFrame(_arr.toarray(), columns=_names, index=df.index))"
                        )
                    },
                    {
                        "label": "TF-IDF (100 features)",
                        "value": "tfidf_100",
                        "imports": ["from sklearn.feature_extraction.text import TfidfVectorizer", "import pandas as pd"],
                        "params": {"max_features": 100, "ngram_range": [1, 2], "min_df": 1},
                        "code": (
                            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
                            "import pandas as pd\n"
                            "_tv = TfidfVectorizer(max_features=params.get('max_features', 100), "
                            "ngram_range=tuple(params.get('ngram_range', [1,2])), min_df=params.get('min_df', 1))\n"
                            "_arr = _tv.fit_transform(df[col].astype(str))\n"
                            "_names = [f'{col}_tfidf_{n}' for n in _tv.get_feature_names_out()]\n"
                            "df = df.drop(columns=[col]).join(pd.DataFrame(_arr.toarray(), columns=_names, index=df.index))"
                        )
                    },
                    {
                        "label": "Count Vectorizer (Bag of Words)",
                        "value": "bow",
                        "imports": ["from sklearn.feature_extraction.text import CountVectorizer", "import pandas as pd"],
                        "params": {"max_features": 100, "ngram_range": [1, 1]},
                        "code": (
                            "from sklearn.feature_extraction.text import CountVectorizer\n"
                            "import pandas as pd\n"
                            "_cv = CountVectorizer(max_features=params.get('max_features', 100), "
                            "ngram_range=tuple(params.get('ngram_range', [1,1])))\n"
                            "_arr = _cv.fit_transform(df[col].astype(str))\n"
                            "_names = [f'{col}_bow_{n}' for n in _cv.get_feature_names_out()]\n"
                            "df = df.drop(columns=[col]).join(pd.DataFrame(_arr.toarray(), columns=_names, index=df.index))"
                        )
                    },
                    {
                        "label": "Hashing Vectorizer",
                        "value": "hashing_vec",
                        "imports": ["from sklearn.feature_extraction.text import HashingVectorizer", "import pandas as pd"],
                        "params": {"n_features": 64},
                        "code": (
                            "from sklearn.feature_extraction.text import HashingVectorizer\n"
                            "import pandas as pd\n"
                            "_hv = HashingVectorizer(n_features=params.get('n_features', 64), alternate_sign=False)\n"
                            "_arr = _hv.transform(df[col].astype(str))\n"
                            "_names = [f'{col}_hash_{i}' for i in range(params.get('n_features', 64))]\n"
                            "df = df.drop(columns=[col]).join(pd.DataFrame(_arr.toarray(), columns=_names, index=df.index))"
                        )
                    },
                    {
                        "label": "Sentence Embeddings (sentence-transformers)",
                        "value": "sentence_embed",
                        "imports": ["from sentence_transformers import SentenceTransformer", "import pandas as pd"],
                        "params": {"model_name": "all-MiniLM-L6-v2"},
                        "code": (
                            "from sentence_transformers import SentenceTransformer\n"
                            "import pandas as pd\n"
                            "_smodel = SentenceTransformer(params.get('model_name', 'all-MiniLM-L6-v2'))\n"
                            "_emb = _smodel.encode(df[col].astype(str).tolist())\n"
                            "_names = [f'{col}_emb_{i}' for i in range(_emb.shape[1])]\n"
                            "df = df.drop(columns=[col]).join(pd.DataFrame(_emb, columns=_names, index=df.index))"
                        )
                    },
                    {
                        "label": "Drop column",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "df.drop(columns=[col], inplace=True)"
                    },
                ],
                "id_like": [
                    {
                        "label": "Drop column (recommended)",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "df.drop(columns=[col], inplace=True)"
                    },
                    {
                        "label": "Label Encoding (if needed as key)",
                        "value": "label",
                        "imports": ["from sklearn.preprocessing import LabelEncoder"],
                        "params": {},
                        "code": (
                            "from sklearn.preprocessing import LabelEncoder\n"
                            "_le = LabelEncoder()\n"
                            "df[col] = _le.fit_transform(df[col].astype(str))"
                        )
                    },
                    {
                        "label": "Passthrough",
                        "value": "none",
                        "imports": [],
                        "params": {},
                        "code": "# passthrough"
                    },
                ],
            },
            "non_tabular": {
                "image": [
                    {
                        "label": "Flatten (→ 1D array)",
                        "value": "flatten",
                        "imports": ["import numpy as np"],
                        "params": {},
                        "code": (
                            "import numpy as np\n"
                            "data[label] = np.array([np.array(img).flatten() for img in data[label]])"
                        )
                    },
                    {
                        "label": "Normalize pixels [0,1]",
                        "value": "normalize_01",
                        "imports": ["import numpy as np"],
                        "params": {},
                        "code": (
                            "import numpy as np\n"
                            "data[label] = np.array([np.array(img) / 255.0 for img in data[label]])"
                        )
                    },
                    {
                        "label": "Resize + normalize",
                        "value": "resize_normalize",
                        "imports": ["import numpy as np", "from PIL import Image"],
                        "params": {"size": [224, 224]},
                        "code": (
                            "import numpy as np\n"
                            "from PIL import Image\n"
                            "_sz = tuple(params.get('size', [224, 224]))\n"
                            "data[label] = np.array([np.array(img.resize(_sz)) / 255.0 for img in data[label]])"
                        )
                    },
                    {
                        "label": "Grayscale + flatten",
                        "value": "gray_flatten",
                        "imports": ["import numpy as np", "import cv2"],
                        "params": {"size": [64, 64]},
                        "code": (
                            "import numpy as np, cv2\n"
                            "_sz = tuple(params.get('size', [64, 64]))\n"
                            "data[label] = np.array([cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY), _sz).flatten() for img in data[label]])"
                        )
                    },
                    {
                        "label": "ResNet50 feature extraction (torchvision)",
                        "value": "resnet50_features",
                        "imports": ["import torch", "import torchvision.models as models", "import torchvision.transforms as T", "import numpy as np"],
                        "params": {},
                        "code": (
                            "import torch, numpy as np\n"
                            "import torchvision.models as models, torchvision.transforms as T\n"
                            "_m = models.resnet50(pretrained=True); _m.eval()\n"
                            "_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), "
                            "T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])\n"
                            "with torch.no_grad():\n"
                            "    _feats = [_m(torch.unsqueeze(_tf(img.convert('RGB')), 0)).squeeze().numpy() for img in data[label]]\n"
                            "data[label] = np.array(_feats)"
                        )
                    },
                    {
                        "label": "Drop",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "del data[label]"
                    },
                ],
                "text": [
                    {
                        "label": "TF-IDF (50 features)",
                        "value": "tfidf_50",
                        "imports": ["from sklearn.feature_extraction.text import TfidfVectorizer"],
                        "params": {"max_features": 50},
                        "code": (
                            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
                            "_tv = TfidfVectorizer(max_features=params.get('max_features', 50))\n"
                            "data[label] = _tv.fit_transform(data[label]).toarray()"
                        )
                    },
                    {
                        "label": "Sentence Embeddings (sentence-transformers)",
                        "value": "sentence_embed",
                        "imports": ["from sentence_transformers import SentenceTransformer"],
                        "params": {"model_name": "all-MiniLM-L6-v2"},
                        "code": (
                            "from sentence_transformers import SentenceTransformer\n"
                            "_smodel = SentenceTransformer(params.get('model_name', 'all-MiniLM-L6-v2'))\n"
                            "data[label] = _smodel.encode(data[label])"
                        )
                    },
                    {
                        "label": "Drop",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "del data[label]"
                    },
                ],
                "graph": [
                    {
                        "label": "Degree centrality features",
                        "value": "degree_centrality",
                        "imports": ["import networkx as nx", "import pandas as pd"],
                        "params": {},
                        "code": (
                            "import networkx as nx, pandas as pd\n"
                            "_G = data[label]\n"
                            "_feats = {'degree': dict(nx.degree(_G)), 'betweenness': nx.betweenness_centrality(_G), "
                            "'closeness': nx.closeness_centrality(_G)}\n"
                            "data[label + '_graph_features'] = pd.DataFrame(_feats)"
                        )
                    },
                    {
                        "label": "Adjacency matrix (dense)",
                        "value": "adjacency_dense",
                        "imports": ["import networkx as nx", "import numpy as np"],
                        "params": {},
                        "code": (
                            "import networkx as nx, numpy as np\n"
                            "data[label] = nx.to_numpy_array(data[label])"
                        )
                    },
                    {
                        "label": "Drop",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "del data[label]"
                    },
                ],
                "timeseries": [
                    {
                        "label": "Lag features",
                        "value": "lag_features",
                        "imports": [],
                        "params": {"lags": [1, 2, 3], "target_col": "value"},
                        "code": (
                            "_tcol = params.get('target_col', 'value')\n"
                            "for _lag in params.get('lags', [1,2,3]):\n"
                            "    df[f'{_tcol}_lag{_lag}'] = df[_tcol].shift(_lag)"
                        )
                    },
                    {
                        "label": "Rolling mean features",
                        "value": "rolling_mean",
                        "imports": [],
                        "params": {"windows": [3, 7, 14], "target_col": "value"},
                        "code": (
                            "_tcol = params.get('target_col', 'value')\n"
                            "for _w in params.get('windows', [3,7,14]):\n"
                            "    df[f'{_tcol}_roll{_w}'] = df[_tcol].rolling(_w).mean()"
                        )
                    },
                    {
                        "label": "Seasonal decomposition (STL)",
                        "value": "stl_decompose",
                        "imports": ["from statsmodels.tsa.seasonal import STL"],
                        "params": {"target_col": "value", "period": 12},
                        "code": (
                            "from statsmodels.tsa.seasonal import STL\n"
                            "_tcol = params.get('target_col', 'value')\n"
                            "_res = STL(df[_tcol], period=params.get('period', 12)).fit()\n"
                            "df['trend'] = _res.trend\n"
                            "df['seasonal'] = _res.seasonal\n"
                            "df['residual'] = _res.resid"
                        )
                    },
                    {
                        "label": "Passthrough",
                        "value": "none",
                        "imports": [],
                        "params": {},
                        "code": "# passthrough"
                    },
                ],
                "ontology": [
                    {
                        "label": "Extract Classes",
                        "value": "extract_classes",
                        "imports": ["from rdflib.namespace import OWL, RDF", "import pandas as pd"],
                        "params": {},
                        "code": (
                            "from rdflib.namespace import OWL, RDF\n"
                            "import pandas as pd\n"
                            "_g = data[label]\n"
                            "_classes = [str(s) for s, p, o in _g.triples((None, RDF.type, OWL.Class))]\n"
                            "data[label + '_classes'] = pd.DataFrame({'Class_URI': _classes})"
                        )
                    },
                    {
                        "label": "Extract Object Properties",
                        "value": "extract_object_props",
                        "imports": ["from rdflib.namespace import OWL, RDF", "import pandas as pd"],
                        "params": {},
                        "code": (
                            "from rdflib.namespace import OWL, RDF\n"
                            "import pandas as pd\n"
                            "_g = data[label]\n"
                            "_props = [str(s) for s, p, o in _g.triples((None, RDF.type, OWL.ObjectProperty))]\n"
                            "data[label + '_object_props'] = pd.DataFrame({'Property_URI': _props})"
                        )
                    },
                    {
                        "label": "Extract Datatype Properties",
                        "value": "extract_data_props",
                        "imports": ["from rdflib.namespace import OWL, RDF", "import pandas as pd"],
                        "params": {},
                        "code": (
                            "from rdflib.namespace import OWL, RDF\n"
                            "import pandas as pd\n"
                            "_g = data[label]\n"
                            "_props = [str(s) for s, p, o in _g.triples((None, RDF.type, OWL.DatatypeProperty))]\n"
                            "data[label + '_data_props'] = pd.DataFrame({'Property_URI': _props})"
                        )
                    },
                    {
                        "label": "Convert to NetworkX Graph",
                        "value": "to_networkx",
                        "imports": ["import networkx as nx"],
                        "params": {},
                        "code": (
                            "import networkx as nx\n"
                            "_g = data[label]\n"
                            "_nx_g = nx.DiGraph()\n"
                            "for s, p, o in _g:\n"
                            "    _nx_g.add_edge(str(s), str(o), label=str(p))\n"
                            "data[label + '_nx'] = _nx_g\n"
                            "del data[label]  # Replace with NetworkX version"
                        )
                    },
                    {
                        "label": "Drop",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "del data[label]"
                    },
                ],
                "unknown": [
                    {
                        "label": "Drop",
                        "value": "drop",
                        "imports": [],
                        "params": {},
                        "code": "del data[label]"
                    },
                    {
                        "label": "Passthrough (as-is)",
                        "value": "none",
                        "imports": [],
                        "params": {},
                        "code": "# passthrough"
                    },
                ],
            },
        },
        "feature_selection": [
            {
                "label": "Variance Threshold",
                "value": "variance_threshold",
                "imports": ["from sklearn.feature_selection import VarianceThreshold"],
                "params": {"threshold": 0.0},
                "code": (
                    "from sklearn.feature_selection import VarianceThreshold\n"
                    "_vs = VarianceThreshold(threshold=params.get('threshold', 0.0))\n"
                    "_arr = _vs.fit_transform(df)\n"
                    "df = pd.DataFrame(_arr, columns=df.columns[_vs.get_support()])"
                )
            },
            {
                "label": "SelectKBest (f_classif)",
                "value": "selectkbest_f",
                "imports": ["from sklearn.feature_selection import SelectKBest, f_classif"],
                "params": {"k": 10},
                "code": (
                    "from sklearn.feature_selection import SelectKBest, f_classif\n"
                    "_skb = SelectKBest(f_classif, k=params.get('k', 10))\n"
                    "_arr = _skb.fit_transform(df, y)\n"
                    "df = pd.DataFrame(_arr, columns=df.columns[_skb.get_support()])"
                )
            },
            {
                "label": "SelectKBest (mutual_info)",
                "value": "selectkbest_mi",
                "imports": ["from sklearn.feature_selection import SelectKBest, mutual_info_classif"],
                "params": {"k": 10},
                "code": (
                    "from sklearn.feature_selection import SelectKBest, mutual_info_classif\n"
                    "_skb = SelectKBest(mutual_info_classif, k=params.get('k', 10))\n"
                    "_arr = _skb.fit_transform(df, y)\n"
                    "df = pd.DataFrame(_arr, columns=df.columns[_skb.get_support()])"
                )
            },
            {
                "label": "Correlation-based (Pearson threshold)",
                "value": "corr_threshold",
                "imports": [],
                "params": {"threshold": 0.95},
                "code": (
                    "_corr = df.corr().abs()\n"
                    "_upper = _corr.where(np.triu(np.ones(_corr.shape), k=1).astype(bool))\n"
                    "_to_drop = [c for c in _upper.columns if any(_upper[c] > params.get('threshold', 0.95))]\n"
                    "df.drop(columns=_to_drop, inplace=True)"
                )
            },
            {
                "label": "RFE (RandomForest)",
                "value": "rfe_rf",
                "imports": ["from sklearn.feature_selection import RFE", "from sklearn.ensemble import RandomForestClassifier"],
                "params": {"n_features_to_select": 10},
                "code": (
                    "from sklearn.feature_selection import RFE\n"
                    "from sklearn.ensemble import RandomForestClassifier\n"
                    "_rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), "
                    "n_features_to_select=params.get('n_features_to_select', 10))\n"
                    "_arr = _rfe.fit_transform(df, y)\n"
                    "df = pd.DataFrame(_arr, columns=df.columns[_rfe.get_support()])"
                )
            },
            {
                "label": "PCA",
                "value": "pca",
                "imports": ["from sklearn.decomposition import PCA"],
                "params": {"n_components": 10, "random_state": 42},
                "code": (
                    "from sklearn.decomposition import PCA\n"
                    "_pca = PCA(n_components=params.get('n_components', 10), random_state=params.get('random_state', 42))\n"
                    "_arr = _pca.fit_transform(df)\n"
                    "df = pd.DataFrame(_arr, columns=[f'pca_{i}' for i in range(_arr.shape[1])])"
                )
            },
            {
                "label": "UMAP (2D)",
                "value": "umap",
                "imports": ["import umap"],
                "params": {"n_components": 2, "n_neighbors": 15, "random_state": 42},
                "code": (
                    "import umap\n"
                    "_um = umap.UMAP(n_components=params.get('n_components', 2), "
                    "n_neighbors=params.get('n_neighbors', 15), random_state=params.get('random_state', 42))\n"
                    "_arr = _um.fit_transform(df)\n"
                    "df = pd.DataFrame(_arr, columns=[f'umap_{i}' for i in range(_arr.shape[1])])"
                )
            },
        ],
        "modeling": {
            "classification": {
                "binary": [
                    {
                        "name": "Logistic Regression",
                        "class_name": "LogisticRegression",
                        "module": "sklearn.linear_model",
                        "imports": ["from sklearn.linear_model import LogisticRegression"],
                        "params": {"max_iter": 1000, "C": 1.0, "solver": "lbfgs"},
                        "code": (
                            "from sklearn.linear_model import LogisticRegression\n"
                            "model = LogisticRegression(**params)"
                        )
                    },
                    {
                        "name": "Random Forest",
                        "class_name": "RandomForestClassifier",
                        "module": "sklearn.ensemble",
                        "imports": ["from sklearn.ensemble import RandomForestClassifier"],
                        "params": {"n_estimators": 200, "max_depth": None, "random_state": 42},
                        "code": (
                            "from sklearn.ensemble import RandomForestClassifier\n"
                            "model = RandomForestClassifier(**params)"
                        )
                    },
                    {
                        "name": "Gradient Boosting (sklearn)",
                        "class_name": "GradientBoostingClassifier",
                        "module": "sklearn.ensemble",
                        "imports": ["from sklearn.ensemble import GradientBoostingClassifier"],
                        "params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
                        "code": (
                            "from sklearn.ensemble import GradientBoostingClassifier\n"
                            "model = GradientBoostingClassifier(**params)"
                        )
                    },
                    {
                        "name": "XGBoost",
                        "class_name": "XGBClassifier",
                        "module": "xgboost",
                        "imports": ["from xgboost import XGBClassifier"],
                        "params": {"n_estimators": 200, "learning_rate": 0.1, "use_label_encoder": False, "eval_metric": "logloss"},
                        "code": (
                            "from xgboost import XGBClassifier\n"
                            "model = XGBClassifier(**params)"
                        )
                    },
                    {
                        "name": "LightGBM",
                        "class_name": "LGBMClassifier",
                        "module": "lightgbm",
                        "imports": ["from lightgbm import LGBMClassifier"],
                        "params": {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31},
                        "code": (
                            "from lightgbm import LGBMClassifier\n"
                            "model = LGBMClassifier(**params)"
                        )
                    },
                    {
                        "name": "CatBoost",
                        "class_name": "CatBoostClassifier",
                        "module": "catboost",
                        "imports": ["from catboost import CatBoostClassifier"],
                        "params": {"iterations": 200, "learning_rate": 0.05, "verbose": 0},
                        "code": (
                            "from catboost import CatBoostClassifier\n"
                            "model = CatBoostClassifier(**params)"
                        )
                    },
                    {
                        "name": "SVM (RBF)",
                        "class_name": "SVC",
                        "module": "sklearn.svm",
                        "imports": ["from sklearn.svm import SVC"],
                        "params": {"C": 1.0, "kernel": "rbf", "probability": True},
                        "code": (
                            "from sklearn.svm import SVC\n"
                            "model = SVC(**params)"
                        )
                    },
                    {
                        "name": "K-Nearest Neighbors",
                        "class_name": "KNeighborsClassifier",
                        "module": "sklearn.neighbors",
                        "imports": ["from sklearn.neighbors import KNeighborsClassifier"],
                        "params": {"n_neighbors": 5},
                        "code": (
                            "from sklearn.neighbors import KNeighborsClassifier\n"
                            "model = KNeighborsClassifier(**params)"
                        )
                    },
                    {
                        "name": "Naive Bayes (Gaussian)",
                        "class_name": "GaussianNB",
                        "module": "sklearn.naive_bayes",
                        "imports": ["from sklearn.naive_bayes import GaussianNB"],
                        "params": {},
                        "code": (
                            "from sklearn.naive_bayes import GaussianNB\n"
                            "model = GaussianNB(**params)"
                        )
                    },
                    {
                        "name": "AdaBoost",
                        "class_name": "AdaBoostClassifier",
                        "module": "sklearn.ensemble",
                        "imports": ["from sklearn.ensemble import AdaBoostClassifier"],
                        "params": {"n_estimators": 100, "learning_rate": 0.5},
                        "code": (
                            "from sklearn.ensemble import AdaBoostClassifier\n"
                            "model = AdaBoostClassifier(**params)"
                        )
                    },
                    {
                        "name": "MLP Neural Network",
                        "class_name": "MLPClassifier",
                        "module": "sklearn.neural_network",
                        "imports": ["from sklearn.neural_network import MLPClassifier"],
                        "params": {"hidden_layer_sizes": [100, 50], "max_iter": 300, "random_state": 42},
                        "code": (
                            "from sklearn.neural_network import MLPClassifier\n"
                            "model = MLPClassifier(**params)"
                        )
                    },
                ],
                "multiclass": [
                    {
                        "name": "Random Forest",
                        "class_name": "RandomForestClassifier",
                        "module": "sklearn.ensemble",
                        "imports": ["from sklearn.ensemble import RandomForestClassifier"],
                        "params": {"n_estimators": 200, "random_state": 42},
                        "code": (
                            "from sklearn.ensemble import RandomForestClassifier\n"
                            "model = RandomForestClassifier(**params)"
                        )
                    },
                    {
                        "name": "XGBoost (multiclass)",
                        "class_name": "XGBClassifier",
                        "module": "xgboost",
                        "imports": ["from xgboost import XGBClassifier"],
                        "params": {"n_estimators": 200, "objective": "multi:softprob", "eval_metric": "mlogloss"},
                        "code": (
                            "from xgboost import XGBClassifier\n"
                            "model = XGBClassifier(**params)"
                        )
                    },
                    {
                        "name": "LightGBM (multiclass)",
                        "class_name": "LGBMClassifier",
                        "module": "lightgbm",
                        "imports": ["from lightgbm import LGBMClassifier"],
                        "params": {"n_estimators": 200, "objective": "multiclass"},
                        "code": (
                            "from lightgbm import LGBMClassifier\n"
                            "model = LGBMClassifier(**params)"
                        )
                    },
                ],
            },
            "regression": {
                "continuous": [
                    {
                        "name": "Linear Regression",
                        "class_name": "LinearRegression",
                        "module": "sklearn.linear_model",
                        "imports": ["from sklearn.linear_model import LinearRegression"],
                        "params": {},
                        "code": (
                            "from sklearn.linear_model import LinearRegression\n"
                            "model = LinearRegression(**params)"
                        )
                    },
                    {
                        "name": "Ridge Regression",
                        "class_name": "Ridge",
                        "module": "sklearn.linear_model",
                        "imports": ["from sklearn.linear_model import Ridge"],
                        "params": {"alpha": 1.0},
                        "code": (
                            "from sklearn.linear_model import Ridge\n"
                            "model = Ridge(**params)"
                        )
                    },
                    {
                        "name": "Lasso Regression",
                        "class_name": "Lasso",
                        "module": "sklearn.linear_model",
                        "imports": ["from sklearn.linear_model import Lasso"],
                        "params": {"alpha": 1.0, "max_iter": 1000},
                        "code": (
                            "from sklearn.linear_model import Lasso\n"
                            "model = Lasso(**params)"
                        )
                    },
                    {
                        "name": "ElasticNet",
                        "class_name": "ElasticNet",
                        "module": "sklearn.linear_model",
                        "imports": ["from sklearn.linear_model import ElasticNet"],
                        "params": {"alpha": 1.0, "l1_ratio": 0.5, "max_iter": 1000},
                        "code": (
                            "from sklearn.linear_model import ElasticNet\n"
                            "model = ElasticNet(**params)"
                        )
                    },
                    {
                        "name": "Random Forest Regressor",
                        "class_name": "RandomForestRegressor",
                        "module": "sklearn.ensemble",
                        "imports": ["from sklearn.ensemble import RandomForestRegressor"],
                        "params": {"n_estimators": 200, "random_state": 42},
                        "code": (
                            "from sklearn.ensemble import RandomForestRegressor\n"
                            "model = RandomForestRegressor(**params)"
                        )
                    },
                    {
                        "name": "Gradient Boosting Regressor",
                        "class_name": "GradientBoostingRegressor",
                        "module": "sklearn.ensemble",
                        "imports": ["from sklearn.ensemble import GradientBoostingRegressor"],
                        "params": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
                        "code": (
                            "from sklearn.ensemble import GradientBoostingRegressor\n"
                            "model = GradientBoostingRegressor(**params)"
                        )
                    },
                    {
                        "name": "XGBoost Regressor",
                        "class_name": "XGBRegressor",
                        "module": "xgboost",
                        "imports": ["from xgboost import XGBRegressor"],
                        "params": {"n_estimators": 200, "learning_rate": 0.1},
                        "code": (
                            "from xgboost import XGBRegressor\n"
                            "model = XGBRegressor(**params)"
                        )
                    },
                    {
                        "name": "LightGBM Regressor",
                        "class_name": "LGBMRegressor",
                        "module": "lightgbm",
                        "imports": ["from lightgbm import LGBMRegressor"],
                        "params": {"n_estimators": 200, "learning_rate": 0.05},
                        "code": (
                            "from lightgbm import LGBMRegressor\n"
                            "model = LGBMRegressor(**params)"
                        )
                    },
                    {
                        "name": "SVR (RBF)",
                        "class_name": "SVR",
                        "module": "sklearn.svm",
                        "imports": ["from sklearn.svm import SVR"],
                        "params": {"C": 1.0, "kernel": "rbf"},
                        "code": (
                            "from sklearn.svm import SVR\n"
                            "model = SVR(**params)"
                        )
                    },
                    {
                        "name": "MLP Regressor",
                        "class_name": "MLPRegressor",
                        "module": "sklearn.neural_network",
                        "imports": ["from sklearn.neural_network import MLPRegressor"],
                        "params": {"hidden_layer_sizes": [100, 50], "max_iter": 300, "random_state": 42},
                        "code": (
                            "from sklearn.neural_network import MLPRegressor\n"
                            "model = MLPRegressor(**params)"
                        )
                    },
                ],
            },
            "clustering": [
                {
                    "name": "K-Means",
                    "class_name": "KMeans",
                    "module": "sklearn.cluster",
                    "imports": ["from sklearn.cluster import KMeans"],
                    "params": {"n_clusters": 8, "random_state": 42, "n_init": "auto"},
                    "code": (
                        "from sklearn.cluster import KMeans\n"
                        "model = KMeans(**params)"
                    )
                },
                {
                    "name": "DBSCAN",
                    "class_name": "DBSCAN",
                    "module": "sklearn.cluster",
                    "imports": ["from sklearn.cluster import DBSCAN"],
                    "params": {"eps": 0.5, "min_samples": 5},
                    "code": (
                        "from sklearn.cluster import DBSCAN\n"
                        "model = DBSCAN(**params)"
                    )
                },
                {
                    "name": "Agglomerative Clustering",
                    "class_name": "AgglomerativeClustering",
                    "module": "sklearn.cluster",
                    "imports": ["from sklearn.cluster import AgglomerativeClustering"],
                    "params": {"n_clusters": 8, "linkage": "ward"},
                    "code": (
                        "from sklearn.cluster import AgglomerativeClustering\n"
                        "model = AgglomerativeClustering(**params)"
                    )
                },
            ],
            "non_tabular": {
                "image": [
                    {
                        "name": "CNN (PyTorch)",
                        "class_name": "CNNClassifier",
                        "module": "custom",
                        "imports": ["import torch", "import torch.nn as nn"],
                        "params": {"learning_rate": 0.001, "epochs": 10},
                        "code": "# PyTorch CNN definition..."
                    },
                    {
                        "name": "ResNet50 (Transfer Learning)",
                        "class_name": "ResNet50FT",
                        "module": "torchvision.models",
                        "imports": ["from torchvision.models import resnet50", "import torch.nn as nn"],
                        "params": {"pretrained": True, "freeze_backbone": True},
                        "code": "# Fine-tune ResNet50..."
                    }
                ],
                "text": [
                    {
                        "name": "LSTM Classifier",
                        "class_name": "LSTMText",
                        "module": "custom",
                        "imports": ["import torch", "import torch.nn as nn"],
                        "params": {"hidden_dim": 128, "epochs": 5},
                        "code": "# LSTM model..."
                    },
                    {
                        "name": "BERT (Fine-Tuning)",
                        "class_name": "BERTFT",
                        "module": "transformers",
                        "imports": ["from transformers import BertForSequenceClassification"],
                        "params": {"model_name": "bert-base-uncased"},
                        "code": "# BERT sequence classification..."
                    }
                ],
                "graph": [
                    {
                        "name": "Graph Convolutional Network (GCN)",
                        "class_name": "GCN",
                        "module": "torch_geometric.nn",
                        "imports": ["from torch_geometric.nn import GCNConv"],
                        "params": {"hidden_channels": 64},
                        "code": "# GCN implementation..."
                    },
                    {
                        "name": "GraphSAGE",
                        "class_name": "GraphSAGE",
                        "module": "torch_geometric.nn",
                        "imports": ["from torch_geometric.nn import SAGEConv"],
                        "params": {"hidden_channels": 64},
                        "code": "# GraphSAGE..."
                    }
                ],
                "timeseries": [
                    {
                        "name": "LSTM Timeseries",
                        "class_name": "LSTMTFS",
                        "module": "custom",
                        "imports": ["import torch", "import torch.nn as nn"],
                        "params": {"hidden_size": 64, "num_layers": 2},
                        "code": "# Timeseries LSTM..."
                    },
                    {
                        "name": "Prophet",
                        "class_name": "Prophet",
                        "module": "prophet",
                        "imports": ["from prophet import Prophet"],
                        "params": {"yearly_seasonality": True},
                        "code": "# Prophet forecast..."
                    }
                ],
                "ontology": [
                    {
                        "name": "HermiT Reasoner (Symbolic)",
                        "class_name": "HermiTReasoner",
                        "module": "owlready2",
                        "imports": ["from owlready2 import sync_reasoner"],
                        "params": {"infer_property_values": True},
                        "code": (
                            "from owlready2 import sync_reasoner\n"
                            "sync_reasoner(model, infer_property_values=params.get('infer_property_values', True))"
                        )
                    },
                    {
                        "name": "Pellet Reasoner",
                        "class_name": "PelletReasoner",
                        "module": "owlready2",
                        "imports": ["from owlready2 import sync_reasoner_pellet"],
                        "params": {"infer_property_values": True, "infer_data_property_values": True},
                        "code": (
                            "from owlready2 import sync_reasoner_pellet\n"
                            "sync_reasoner_pellet(model, infer_property_values=params.get('infer_property_values', True), "
                            "infer_data_property_values=params.get('infer_data_property_values', True))"
                        )
                    },
                    {
                        "name": "Neuro-Symbolic Model (GNN + Reasoner)",
                        "class_name": "NeuroSymbolicOntology",
                        "module": "custom",
                        "imports": ["from owlready2 import sync_reasoner", "from torch_geometric.nn import GCNConv"],
                        "params": {"embedding_dim": 64, "gnn_layers": 2},
                        "code": (
                            "# Neuro-Symbolic Pipeline\n"
                            "# 1. Run symbolic reasoner to infer new relationships\n"
                            "# 2. Convert expanded ontology to graph\n"
                            "# 3. Run GNN on the augmented graph structure"
                        )
                    }
                ]
            }
        },
        "metrics": {
            "classification": {
                "binary": [
                    {"name": "Accuracy", "func": "accuracy_score", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import accuracy_score"]},
                    {"name": "ROC-AUC", "func": "roc_auc_score", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import roc_auc_score"]},
                    {"name": "F1 (binary)", "func": "f1_score", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import f1_score"], "kwargs": {"average": "binary"}},
                    {"name": "Precision", "func": "precision_score", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import precision_score"]},
                    {"name": "Recall", "func": "recall_score", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import recall_score"]},
                    {"name": "Log Loss", "func": "log_loss", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import log_loss"]},
                    {"name": "Matthews Correlation Coefficient", "func": "matthews_corrcoef", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import matthews_corrcoef"]},
                ],
                "multiclass": [
                    {"name": "Accuracy", "func": "accuracy_score", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import accuracy_score"]},
                    {"name": "F1 (macro)", "func": "f1_score", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import f1_score"], "kwargs": {"average": "macro"}},
                    {"name": "F1 (weighted)", "func": "f1_score", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import f1_score"], "kwargs": {"average": "weighted"}},
                ],
            },
            "regression": [
                {"name": "R²", "func": "r2_score", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import r2_score"]},
                {"name": "MAE", "func": "mean_absolute_error", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import mean_absolute_error"]},
                {"name": "MSE", "func": "mean_squared_error", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import mean_squared_error"]},
                {"name": "RMSE", "func": "mean_squared_error", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import mean_squared_error"], "post": "np.sqrt"},
                {"name": "MAPE", "func": "mean_absolute_percentage_error", "module": "sklearn.metrics", "imports": ["from sklearn.metrics import mean_absolute_percentage_error"]},
            ],
        },
        "explainability": {
            "tabular": [
                {
                    "name": "SHAP (Tree/Kernel)",
                    "value": "shap",
                    "imports": ["import shap"],
                    "params": {},
                    "code": (
                        "import shap\n"
                        "explainer = shap.Explainer(model, X_train)\n"
                        "shap_values = explainer(X_test)"
                    )
                },
                {
                    "name": "LIME (Tabular)",
                    "value": "lime",
                    "imports": ["import lime", "import lime.lime_tabular"],
                    "params": {},
                    "code": (
                        "import lime, lime.lime_tabular\n"
                        "explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns, mode='classification')\n"
                        "exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)"
                    )
                }
            ],
            "image": [
                {
                    "name": "Grad-CAM",
                    "value": "gradcam",
                    "imports": ["from pytorch_grad_cam import GradCAM"],
                    "params": {},
                    "code": (
                        "from pytorch_grad_cam import GradCAM\n"
                        "cam = GradCAM(model=model, target_layers=[model.layer4[-1]])\n"
                        "grayscale_cam = cam(input_tensor=input_tensor)"
                    )
                },
                {
                    "name": "Integrated Gradients",
                    "value": "intgrad",
                    "imports": ["from captum.attr import IntegratedGradients"],
                    "params": {},
                    "code": (
                        "from captum.attr import IntegratedGradients\n"
                        "ig = IntegratedGradients(model)\n"
                        "attr = ig.attribute(input_tensor, target=target_class)"
                    )
                }
            ],
            "text": [
                {
                    "name": "LIME (Text)",
                    "value": "lime_text",
                    "imports": ["from lime.lime_text import LimeTextExplainer"],
                    "params": {},
                    "code": (
                        "from lime.lime_text import LimeTextExplainer\n"
                        "explainer = LimeTextExplainer(class_names=class_names)\n"
                        "exp = explainer.explain_instance(text_instance, pipeline.predict_proba, num_features=6)"
                    )
                },
                {
                    "name": "SHAP (Text)",
                    "value": "shap_text",
                    "imports": ["import shap"],
                    "params": {},
                    "code": (
                        "import shap\n"
                        "explainer = shap.Explainer(model, tokenizer)\n"
                        "shap_values = explainer(text_list)"
                    )
                }
            ],
            "graph": [
                {
                    "name": "GNNExplainer",
                    "value": "gnnexplainer",
                    "imports": ["from torch_geometric.explain import Explainer", "from torch_geometric.explain import GNNExplainer"],
                    "params": {"epochs": 200},
                    "code": (
                        "from torch_geometric.explain import Explainer, GNNExplainer\n"
                        "explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=params.get('epochs', 200)), explanation_type='model', node_mask_type='attributes', edge_mask_type='object')\n"
                        "explanation = explainer(x, edge_index, target=target)"
                    )
                }
            ],
            "ontology": [
                {
                    "name": "Justification (Proof Trees)",
                    "value": "proof_trees",
                    "imports": ["from owlready2 import default_world"],
                    "params": {},
                    "code": (
                        "# Extract justification for inferred axioms\n"
                        "# Works with Pellet reasoner trace\n"
                        "justifications = default_world.get_justifications(axiom)\n"
                    )
                },
                {
                    "name": "Neuro-Symbolic Attention Weights",
                    "value": "neuro_symbolic_attention",
                    "imports": [],
                    "params": {},
                    "code": (
                        "# Extract attention weights over the symbolic knowledge graph\n"
                        "# to explain which logical axioms contributed to the neural prediction.\n"
                        "attention_scores = model.get_attention_weights()\n"
                    )
                }
            ]
        }
    }
def runner(state):
    load_config(state)
    cfg = state.config
    
    # Detailed counts and lists
    domains = ", ".join([d['label'] for d in cfg['domains']])
    loader_count = len(cfg['loading']['supported_types'])
    loader_types = ", ".join(cfg['loading']['supported_types'].keys())
    
    clean = cfg['cleaning']
    c_counts = f"Missing: {len(clean['missing'])} | Outliers: {len(clean['outliers'])} | Types: {len(clean['type_cast'])} | Text: {len(clean['text_cleaning'])}"
    
    enc = cfg['encoding']
    e_tabular = sum(len(v) for v in enc['tabular'].values())
    e_non_tabular = sum(len(v) for v in enc['non_tabular'].values())
    e_fs = len(cfg['feature_selection'])
    e_summary = f"Tabular: {e_tabular} | Non-Tabular: {e_non_tabular} | Feat. Sel: {e_fs}"
    
    mod = cfg['modeling']
    clf = mod['classification']
    reg = mod['regression']
    clu = mod['clustering']
    m_clf_bin = len(clf['binary'])
    m_clf_mul = len(clf['multiclass'])
    m_reg = len(reg['continuous'])
    m_clu = len(clu)
    m_nt = sum(len(v) for v in mod.get('non_tabular', {}).values())
    m_summary = f"Clf-Bin: {m_clf_bin} | Clf-Mul: {m_clf_mul} | Reg: {m_reg} | Clu: {m_clu} | Non-Tab: {m_nt}"
    
    exp = sum(len(v) for v in cfg.get('explainability', {}).values())
    met = len(cfg.get('evaluation_metrics', {}).get('classification', {}).get('binary', [])) + len(cfg.get('evaluation_metrics', {}).get('regression', []))
    
    content = f"""
    <div class='pipeline-kv-grid' style='grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));'>
        
        <div class='pipeline-kv-cell'>
            <div class='kv-key'>Supported Domains</div>
            <div class='kv-val' style='font-size:0.95em; white-space:normal;'>{domains}</div>
        </div>
        
        <div class='pipeline-kv-cell'>
            <div class='kv-key'>Loaders ({loader_count})</div>
            <div class='kv-val' style='font-size:0.85em; white-space:normal; font-weight:normal; color:#475569;'>{loader_types}</div>
        </div>
        
        <div class='pipeline-kv-cell'>
            <div class='kv-key'>Cleaning & Preprocessing</div>
            <div class='kv-val' style='font-size:0.95em;'>{c_counts}</div>
        </div>
        
        <div class='pipeline-kv-cell'>
            <div class='kv-key'>Feature Engineering</div>
            <div class='kv-val' style='font-size:0.95em;'>{e_summary}</div>
        </div>
        
        <div class='pipeline-kv-cell'>
            <div class='kv-key'>Modeling</div>
            <div class='kv-val' style='font-size:0.95em;'>{m_summary}</div>
        </div>
        
        <div class='pipeline-kv-cell'>
            <div class='kv-key'>Eval & Interpretability</div>
            <div class='kv-val' style='font-size:0.95em;'>Metrics: {met} | Explainers: {exp}</div>
        </div>
        
    </div>
    <div style='margin-top:12px; font-size:0.85em; color:#10b981; font-weight:bold; text-align:right;'>
        [OK] Global configuration dictionary injected into 'state.config'
    </div>
    """
    
    html_out = styles.card_html("Configuration Loaded", "Global Pipeline Specs", content)
    
    dashboard = widgets.VBox([widgets.HTML(value=html_out)])
    display(dashboard)
    return dashboard
try:
    runner(state)
except NameError:
    pass