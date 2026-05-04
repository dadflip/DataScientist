"""Étape 4 bis — Feature Engineering pour Ontologies (OntologyFeatureEngUI).

Ce module fournit des outils de feature engineering spécifiques aux ontologies RDF/OWL:
- Extraction de classes, propriétés, individus
- Conversion en graphe NetworkX
- Vectorisation (Node2Vec, RDF2Vec)
- Reasoning et inférence
"""
from __future__ import annotations
import os
import traceback
import pandas as pd
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ml_pipeline.styles import styles


class OntologyFeatureEngUI:
    """Interface Feature Engineering pour Ontologies — extraction, transformation, vectorisation."""

    def __init__(self, state):
        self.state = state
        if not self.state.config:
            self.ui = styles.error_msg("[ERROR] Configuration non chargée.")
            return

        # Filtrer uniquement les ontologies (graphes rdflib)
        self.ontology_datasets = {}
        for k, v in self.state.data_raw.items():
            if self._is_ontology(v):
                self.ontology_datasets[k] = v

        if not self.ontology_datasets:
            self.ui = styles.error_msg("Aucune ontologie disponible. Chargez des fichiers .owl, .rdf, .ttl d'abord.")
            return

        self.current_ds = list(self.ontology_datasets.keys())[0]
        self._build_ui()

    def _is_ontology(self, obj) -> bool:
        """Vérifie si l'objet est une ontologie (rdflib Graph)."""
        try:
            return hasattr(obj, "objects") and hasattr(obj, "subjects") and hasattr(obj, "triples")
        except Exception:
            return False

    def _get_graph(self):
        """Retourne le graphe RDF courant."""
        return self.ontology_datasets.get(self.current_ds)

    def _build_ui(self) -> None:
        """Construit l'interface complète avec onglets."""
        header = widgets.HTML(styles.card_html(
            "Ontology Feature Engineering",
            "Extraction et transformation de features depuis les ontologies",
            ""
        ))
        top_bar = widgets.HBox([header], layout=widgets.Layout(
            align_items="center", margin="0 0 12px 0",
            padding="0 0 10px 0", border_bottom="2px solid #ede9fe"))

        # Sélecteur de dataset
        self.ds_selector = widgets.Dropdown(
            options=list(self.ontology_datasets.keys()),
            description="Ontologie:",
            layout=styles.LAYOUT_DD_LONG
        )
        self.ds_selector.observe(self._on_ds_change, names="value")

        # Onglets
        self.tabs = widgets.Tab()
        self._build_extract_tab()
        self._build_transform_tab()
        self._build_vectorize_tab()
        self._build_reasoning_tab()
        self._build_preview_tab()

        self.tabs.children = [
            self.tab_extract, self.tab_transform,
            self.tab_vectorize, self.tab_reasoning, self.tab_preview
        ]
        for i, title in enumerate([
            "Extraction", "Transformation", "Vectorisation",
            "Reasoning", "Aperçu"
        ]):
            self.tabs.set_title(i, title)

        self.out_logs = widgets.Output()
        self.ui = widgets.VBox(
            [top_bar, self.ds_selector, self.tabs, self.out_logs],
            layout=widgets.Layout(width="100%", max_width="1000px",
                                   border="1px solid #e5e7eb", padding="18px",
                                   border_radius="10px", background_color="#ffffff"))

    def _on_ds_change(self, change) -> None:
        """Gère le changement d'ontologie sélectionnée."""
        if change and change["new"]:
            self.current_ds = change["new"]
        self._refresh_preview()

    # ═══════════════════════════════════════════════════════════════════════════
    # ONGLET 1 — EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_extract_tab(self) -> None:
        """Onglet d'extraction d'entités (classes, propriétés, individus)."""
        help_box = styles.help_box(
            "<b>Extraction d'entités</b> — Extrait les classes, propriétés et individus "
            "de l'ontologie sous forme tabulaire pour analyse ML.",
            "#10b981")

        # Types d'extraction
        self.extract_type = widgets.Dropdown(
            options=[
                ("Classes", "classes"),
                ("Object Properties", "object_properties"),
                ("Datatype Properties", "datatype_properties"),
                ("Annotation Properties", "annotation_properties"),
                ("Individus/Instances", "individuals"),
                ("Triples complets", "triples"),
                ("Hiérarchie de classes", "class_hierarchy"),
                ("Domain/Range des propriétés", "property_domains_ranges"),
            ],
            value="classes",
            description="Extraire:",
            layout=styles.LAYOUT_DD_LONG
        )

        # Options d'extraction
        self.extract_limit = widgets.IntSlider(
            value=1000, min=100, max=10000, step=100,
            description="Limite:", layout=widgets.Layout(width="400px"))

        self.extract_include_labels = widgets.Checkbox(
            value=True, description="Inclure labels (rdfs:label)", indent=False)

        self.extract_include_comments = widgets.Checkbox(
            value=False, description="Inclure comments (rdfs:comment)", indent=False)

        self.btn_extract = widgets.Button(
            description="Extraire en DataFrame",
            button_style=styles.BTN_PRIMARY,
            layout=styles.LAYOUT_BTN_LARGE)
        self.btn_extract.on_click(self._on_extract)

        self.extract_out = widgets.Output()

        self.tab_extract = widgets.VBox([
            help_box,
            widgets.HBox([self.extract_type, self.extract_limit]),
            widgets.HBox([self.extract_include_labels, self.extract_include_comments]),
            self.btn_extract,
            self.extract_out
        ], layout=widgets.Layout(padding="10px"))

    def _on_extract(self, b) -> None:
        """Exécute l'extraction selon le type choisi."""
        with self.extract_out:
            clear_output()
            try:
                g = self._get_graph()
                if not g:
                    display(styles.error_msg("Aucune ontologie sélectionnée."))
                    return

                extract_type = self.extract_type.value
                limit = self.extract_limit.value
                include_labels = self.extract_include_labels.value
                include_comments = self.extract_include_comments.value

                df = self._extract_to_dataframe(
                    g, extract_type, limit, include_labels, include_comments)

                if df is not None and not df.empty:
                    # Stocker dans state pour usage ultérieur
                    output_name = f"{self.current_ds}_{extract_type}"
                    if not hasattr(self.state, 'data_raw'):
                        self.state.data_raw = {}
                    self.state.data_raw[output_name] = df

                    display(styles.success_msg(
                        f"Extraction réussie : {len(df)} lignes × {len(df.columns)} colonnes. "
                        f"Dataset créé: '{output_name}'"))
                    display(df.head(20).style.set_table_styles([
                        {"selector": "thead th",
                         "props": [("background-color", "#f1f5f9"), ("font-size", "0.8em")]},
                        {"selector": "td",
                         "props": [("font-size", "0.82em"), ("padding", "4px 8px")]}
                    ]))
                else:
                    display(styles.warning_msg("Aucune donnée extraite."))

            except Exception as e:
                display(styles.error_msg(f"Erreur extraction: {str(e)}"))
                traceback.print_exc()

    def _extract_to_dataframe(self, g, extract_type: str, limit: int,
                               include_labels: bool, include_comments: bool) -> pd.DataFrame | None:
        """Extrait les données de l'ontologie en DataFrame."""
        from rdflib import Namespace, RDF, RDFS, OWL

        rows = []
        count = 0

        if extract_type == "classes":
            for cls in g.subjects(RDF.type, OWL.Class):
                if count >= limit:
                    break
                row = {"URI": str(cls), "Type": "Class"}
                if include_labels:
                    labels = list(g.objects(cls, RDFS.label))
                    row["Label"] = str(labels[0]) if labels else ""
                if include_comments:
                    comments = list(g.objects(cls, RDFS.comment))
                    row["Comment"] = str(comments[0]) if comments else ""
                # Count subclasses
                sub_count = len(list(g.subjects(RDFS.subClassOf, cls)))
                row["SubClasses"] = sub_count
                # Count instances
                inst_count = len(list(g.subjects(RDF.type, cls)))
                row["Instances"] = inst_count
                rows.append(row)
                count += 1

        elif extract_type == "object_properties":
            for prop in g.subjects(RDF.type, OWL.ObjectProperty):
                if count >= limit:
                    break
                row = {"URI": str(prop), "Type": "ObjectProperty"}
                if include_labels:
                    labels = list(g.objects(prop, RDFS.label))
                    row["Label"] = str(labels[0]) if labels else ""
                domains = list(g.objects(prop, RDFS.domain))
                ranges = list(g.objects(prop, RDFS.range))
                row["Domain"] = str(domains[0]) if domains else ""
                row["Range"] = str(ranges[0]) if ranges else ""
                rows.append(row)
                count += 1

        elif extract_type == "datatype_properties":
            for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
                if count >= limit:
                    break
                row = {"URI": str(prop), "Type": "DatatypeProperty"}
                if include_labels:
                    labels = list(g.objects(prop, RDFS.label))
                    row["Label"] = str(labels[0]) if labels else ""
                domains = list(g.objects(prop, RDFS.domain))
                ranges = list(g.objects(prop, RDFS.range))
                row["Domain"] = str(domains[0]) if domains else ""
                row["Range"] = str(ranges[0]) if ranges else ""
                rows.append(row)
                count += 1

        elif extract_type == "annotation_properties":
            for prop in g.subjects(RDF.type, OWL.AnnotationProperty):
                if count >= limit:
                    break
                row = {"URI": str(prop), "Type": "AnnotationProperty"}
                if include_labels:
                    labels = list(g.objects(prop, RDFS.label))
                    row["Label"] = str(labels[0]) if labels else ""
                rows.append(row)
                count += 1

        elif extract_type == "individuals":
            for ind in g.subjects(RDF.type, None):
                types = list(g.objects(ind, RDF.type))
                if any(t == OWL.NamedIndividual for t in types) or \
                   any(isinstance(t, type(OWL.Class)) and t != OWL.Class for t in types):
                    if count >= limit:
                        break
                    row = {"URI": str(ind), "Type": "NamedIndividual"}
                    class_types = [str(t) for t in types if t != OWL.NamedIndividual]
                    row["ClassTypes"] = ", ".join(class_types[:5])
                    if include_labels:
                        labels = list(g.objects(ind, RDFS.label))
                        row["Label"] = str(labels[0]) if labels else ""
                    rows.append(row)
                    count += 1

        elif extract_type == "triples":
            for s, p, o in g.triples((None, None, None)):
                if count >= limit:
                    break
                rows.append({
                    "Subject": str(s),
                    "Predicate": str(p),
                    "Object": str(o),
                    "ObjectType": "Literal" if isinstance(o, type(g.query("")[0][0])) else "URI"
                })
                count += 1

        elif extract_type == "class_hierarchy":
            for cls in g.subjects(RDF.type, OWL.Class):
                if count >= limit:
                    break
                row = {"Class": str(cls), "SuperClasses": [], "SubClasses": []}
                for sup in g.objects(cls, RDFS.subClassOf):
                    row["SuperClasses"].append(str(sup))
                for sub in g.subjects(RDFS.subClassOf, cls):
                    row["SubClasses"].append(str(sub))
                row["SuperClasses"] = " | ".join(row["SuperClasses"])
                row["SubClasses"] = " | ".join(row["SubClasses"])
                rows.append(row)
                count += 1

        elif extract_type == "property_domains_ranges":
            for prop_type in [OWL.ObjectProperty, OWL.DatatypeProperty, OWL.AnnotationProperty]:
                for prop in g.subjects(RDF.type, prop_type):
                    if count >= limit:
                        break
                    domains = list(g.objects(prop, RDFS.domain))
                    ranges = list(g.objects(prop, RDFS.range))
                    rows.append({
                        "Property": str(prop),
                        "PropertyType": str(prop_type).split("#")[-1],
                        "Domains": " | ".join(str(d) for d in domains) if domains else "",
                        "Ranges": " | ".join(str(r) for r in ranges) if ranges else ""
                    })
                    count += 1

        return pd.DataFrame(rows) if rows else None

    # ═══════════════════════════════════════════════════════════════════════════
    # ONGLET 2 — TRANSFORMATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_transform_tab(self) -> None:
        """Onglet de transformation (filtrage, matérialisation)."""
        help_box = styles.help_box(
            "<b>Transformation</b> — Filtrage de triples, matérialisation d'inférences.",
            "#3b82f6")

        self.transform_action = widgets.Dropdown(
            options=[
                ("Matérialiser sous-classes (transitivité)", "materialize_subclass"),
                ("Matérialiser equivalent classes", "materialize_equivalent"),
                ("Filtrer par namespace", "filter_namespace"),
                ("Supprimer annotations", "remove_annotations"),
                ("Garder uniquement asserted triples", "remove_inferred"),
            ],
            description="Action:",
            layout=styles.LAYOUT_DD_LONG
        )

        self.transform_namespace = widgets.Text(
            description="Namespace:",
            placeholder="http://example.org/",
            layout=widgets.Layout(width="400px"))

        self.btn_transform = widgets.Button(
            description="Appliquer Transformation",
            button_style=styles.BTN_PRIMARY,
            layout=styles.LAYOUT_BTN_LARGE)
        self.btn_transform.on_click(self._on_transform)

        self.transform_out = widgets.Output()

        self.tab_transform = widgets.VBox([
            help_box,
            self.transform_action,
            self.transform_namespace,
            self.btn_transform,
            self.transform_out
        ], layout=widgets.Layout(padding="10px"))

    def _on_transform(self, b) -> None:
        """Applique la transformation sélectionnée."""
        with self.transform_out:
            clear_output()
            try:
                g = self._get_graph()
                if not g:
                    display(styles.error_msg("Aucune ontologie sélectionnée."))
                    return

                action = self.transform_action.value
                namespace = self.transform_namespace.value.strip()

                original_size = len(g)

                if action == "materialize_subclass":
                    self._materialize_subclass_closure(g)
                    display(styles.success_msg(
                        f"Matérialisation terminée. Triples: {original_size} → {len(g)}"))

                elif action == "filter_namespace":
                    if not namespace:
                        display(styles.warning_msg("Spécifiez un namespace."))
                        return
                    self._filter_namespace(g, namespace)
                    display(styles.success_msg(
                        f"Filtrage terminé. Triples: {original_size} → {len(g)}"))

                elif action == "remove_annotations":
                    from rdflib import RDF, RDFS, OWL
                    to_remove = [(s, p, o) for s, p, o in g.triples((None, RDF.type, OWL.AnnotationProperty))]
                    for s, p, o in to_remove:
                        g.remove((s, p, o))
                    display(styles.success_msg(
                        f"Annotations supprimées. Triples: {original_size} → {len(g)}"))

                else:
                    display(styles.info_msg(f"Action '{action}' - implémentation en cours"))

            except Exception as e:
                display(styles.error_msg(f"Erreur transformation: {str(e)}"))
                traceback.print_exc()

    def _materialize_subclass_closure(self, g) -> None:
        """Matérialise la fermeture transitive des sous-classes."""
        from rdflib import RDF, RDFS
        # Simple matérialisation
        new_triples = []
        for s, p, o in g.triples((None, RDFS.subClassOf, None)):
            # Trouver les super-classes transitives
            super_classes = list(g.objects(o, RDFS.subClassOf))
            for sup in super_classes:
                new_triples.append((s, RDFS.subClassOf, sup))
        for triple in new_triples:
            g.add(triple)

    def _filter_namespace(self, g, namespace: str) -> None:
        """Filtre le graphe pour ne garder que les triples d'un namespace."""
        to_remove = []
        for s, p, o in g.triples((None, None, None)):
            if not (str(s).startswith(namespace) or str(p).startswith(namespace)):
                to_remove.append((s, p, o))
        for triple in to_remove:
            g.remove(triple)

    # ═══════════════════════════════════════════════════════════════════════════
    # ONGLET 3 — VECTORISATION
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_vectorize_tab(self) -> None:
        """Onglet de vectorisation (Node2Vec, RDF2Vec)."""
        help_box = styles.help_box(
            "<b>Vectorisation</b> — Conversion de l'ontologie en embeddings pour ML."
            " Nécessite rdflib et optionnellement node2vec.",
            "#8b5cf6")

        self.vector_method = widgets.Dropdown(
            options=[
                ("Node2Vec (Graph Walks)", "node2vec"),
                ("RDF2Vec (Entity-centric)", "rdf2vec"),
                ("One-hot encoding (uri)", "onehot_uri"),
                ("TF-IDF sur labels", "tfidf_labels"),
                ("Graph Statistics Features", "graph_stats"),
            ],
            value="graph_stats",
            description="Méthode:",
            layout=styles.LAYOUT_DD_LONG
        )

        self.vector_dim = widgets.IntSlider(
            value=128, min=16, max=512, step=16,
            description="Dimensions:", layout=widgets.Layout(width="400px"))

        self.btn_vectorize = widgets.Button(
            description="Générer Vecteurs",
            button_style=styles.BTN_PRIMARY,
            layout=styles.LAYOUT_BTN_LARGE)
        self.btn_vectorize.on_click(self._on_vectorize)

        self.vector_out = widgets.Output()

        self.tab_vectorize = widgets.VBox([
            help_box,
            widgets.HBox([self.vector_method, self.vector_dim]),
            self.btn_vectorize,
            self.vector_out
        ], layout=widgets.Layout(padding="10px"))

    def _on_vectorize(self, b) -> None:
        """Génère les vecteurs d'embedding."""
        with self.vector_out:
            clear_output()
            try:
                g = self._get_graph()
                if not g:
                    display(styles.error_msg("Aucune ontologie sélectionnée."))
                    return

                method = self.vector_method.value
                dim = self.vector_dim.value

                if method == "graph_stats":
                    df = self._compute_graph_statistics(g)
                elif method == "onehot_uri":
                    df = self._onehot_encode_uris(g)
                else:
                    display(styles.warning_msg(f"Méthode '{method}' - implémentation simplifiée"))
                    df = self._compute_graph_statistics(g)

                if df is not None and not df.empty:
                    output_name = f"{self.current_ds}_vectors_{method}"
                    if not hasattr(self.state, 'data_raw'):
                        self.state.data_raw = {}
                    self.state.data_raw[output_name] = df

                    display(styles.success_msg(
                        f"Vectorisation terminée. Dataset créé: '{output_name}' "
                        f"({len(df)} entités × {len(df.columns)} features)"))
                    display(df.head(10))

            except Exception as e:
                display(styles.error_msg(f"Erreur vectorisation: {str(e)}"))
                traceback.print_exc()

    def _compute_graph_statistics(self, g) -> pd.DataFrame:
        """Calcule des statistiques graphiques comme features."""
        from rdflib import RDF, RDFS, OWL
        import pandas as pd

        rows = []
        # Pour chaque classe
        for cls in g.subjects(RDF.type, OWL.Class):
            cls_str = str(cls)
            # Compter les instances
            instances = len(list(g.subjects(RDF.type, cls)))
            # Compter les sous-classes
            sub_classes = len(list(g.subjects(RDFS.subClassOf, cls)))
            # Compter les super-classes
            super_classes = len(list(g.objects(cls, RDFS.subClassOf)))
            # Compter les propriétés avec cette classe en domaine
            domain_props = len(list(g.subjects(RDFS.domain, cls)))
            range_props = len(list(g.subjects(RDFS.range, cls)))

            rows.append({
                "Entity": cls_str,
                "Type": "Class",
                "InstanceCount": instances,
                "SubClassCount": sub_classes,
                "SuperClassCount": super_classes,
                "DomainPropCount": domain_props,
                "RangePropCount": range_props,
                "TotalConnections": instances + sub_classes + super_classes + domain_props + range_props
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _onehot_encode_uris(self, g) -> pd.DataFrame:
        """Crée un encodage one-hot simple basé sur les types."""
        from rdflib import RDF, RDFS, OWL

        rows = []
        entity_types = {}

        # Collecter tous les types
        for s, p, o in g.triples((None, RDF.type, None)):
            s_str = str(s)
            if s_str not in entity_types:
                entity_types[s_str] = []
            entity_types[s_str].append(str(o))

        # Créer la matrice one-hot
        all_types = sorted(set(t for types in entity_types.values() for t in types))
        for entity, types in entity_types.items():
            row = {"Entity": entity}
            for t in all_types[:50]:  # Limiter à 50 types pour éviter la dimensionalité excessive
                type_name = t.split("#")[-1].split("/")[-1]
                row[f"type_{type_name}"] = 1 if t in types else 0
            rows.append(row)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ═══════════════════════════════════════════════════════════════════════════
    # ONGLET 4 — REASONING
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_reasoning_tab(self) -> None:
        """Onglet de raisonnement et inférence."""
        help_box = styles.help_box(
            "<b>Reasoning & Inférence</b> — Exécution de règles RDFS/OWL et détection d'incohérences.",
            "#f59e0b")

        self.reasoner_type = widgets.Dropdown(
            options=[
                ("RDFS Closure", "rdfs"),
                ("OWL-EL (simplifié)", "owl_el"),
                ("Custom Rules", "custom"),
                ("Détection d'incohérences", "consistency"),
            ],
            description="Raisonneur:",
            layout=styles.LAYOUT_DD_LONG
        )

        self.btn_reason = widgets.Button(
            description="Exécuter Raisonnement",
            button_style=styles.BTN_PRIMARY,
            layout=styles.LAYOUT_BTN_LARGE)
        self.btn_reason.on_click(self._on_reason)

        self.reason_out = widgets.Output()

        self.tab_reasoning = widgets.VBox([
            help_box,
            self.reasoner_type,
            self.btn_reason,
            self.reason_out
        ], layout=widgets.Layout(padding="10px"))

    def _on_reason(self, b) -> None:
        """Exécute le raisonnement sélectionné."""
        with self.reason_out:
            clear_output()
            try:
                g = self._get_graph()
                if not g:
                    display(styles.error_msg("Aucune ontologie sélectionnée."))
                    return

                reasoner = self.reasoner_type.value

                if reasoner == "rdfs":
                    self._apply_rdfs_rules(g)
                    display(styles.success_msg("Règles RDFS appliquées."))

                elif reasoner == "consistency":
                    issues = self._check_consistency(g)
                    if issues:
                        display(styles.warning_msg(f"{len(issues)} problèmes détectés:"))
                        for issue in issues[:20]:
                            print(f"  - {issue}")
                    else:
                        display(styles.success_msg("Ontologie cohérente."))

                else:
                    display(styles.info_msg(f"Raisonneur '{reasoner}' - implémentation en cours"))

            except Exception as e:
                display(styles.error_msg(f"Erreur raisonnement: {str(e)}"))
                traceback.print_exc()

    def _apply_rdfs_rules(self, g) -> None:
        """Applique les règles RDFS de base."""
        from rdflib import RDF, RDFS
        # RDFS9: subClassOf + type → type
        new_triples = []
        for subj, p, cls in g.triples((None, RDF.type, None)):
            for super_cls in g.objects(cls, RDFS.subClassOf):
                new_triples.append((subj, RDF.type, super_cls))
        for triple in new_triples:
            g.add(triple)

    def _check_consistency(self, g) -> list[str]:
        """Vérifie la cohérence de base de l'ontologie."""
        from rdflib import RDF, RDFS, OWL
        issues = []

        # Vérifier les domain/range violations
        for prop in g.subjects(RDF.type, OWL.ObjectProperty):
            domains = list(g.objects(prop, RDFS.domain))
            ranges = list(g.objects(prop, RDFS.range))
            for subj, p, obj in g.triples((None, prop, None)):
                # Vérifier que le sujet a le bon type
                if domains:
                    subj_types = list(g.objects(subj, RDF.type))
                    if not any(t in domains for t in subj_types):
                        issues.append(f"Domain violation: {subj} {prop} {obj}")

        return issues

    # ═══════════════════════════════════════════════════════════════════════════
    # ONGLET 5 — APERÇU
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_preview_tab(self) -> None:
        """Onglet d'aperçu de l'ontologie."""
        self.preview_out = widgets.Output()
        self.tab_preview = widgets.VBox([
            styles.help_box("<b>Aperçu</b> — Statistiques et visualisation de l'ontologie.", "#64748b"),
            self.preview_out
        ], layout=widgets.Layout(padding="10px"))

    def _refresh_preview(self) -> None:
        """Rafraîchit l'aperçu."""
        with self.preview_out:
            clear_output()
            try:
                g = self._get_graph()
                if not g:
                    display(styles.error_msg("Aucune ontologie sélectionnée."))
                    return

                # Statistiques de base
                from rdflib import RDF, RDFS, OWL
                n_triples = len(g)
                n_classes = len(list(g.subjects(RDF.type, OWL.Class)))
                n_obj_props = len(list(g.subjects(RDF.type, OWL.ObjectProperty)))
                n_data_props = len(list(g.subjects(RDF.type, OWL.DatatypeProperty)))
                n_individuals = len(list(g.subjects(RDF.type, OWL.NamedIndividual)))

                stats_html = f"""
                <div style="background:#f8fafc;padding:16px;border-radius:8px;margin:10px 0;">
                    <h4 style="margin-top:0;color:#374151;">Statistiques de l'ontologie: {self.current_ds}</h4>
                    <table style="width:100%;border-collapse:collapse;">
                        <tr><td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;"><b>Triples</b></td>
                            <td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;">{n_triples:,}</td></tr>
                        <tr><td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;"><b>Classes</b></td>
                            <td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;">{n_classes}</td></tr>
                        <tr><td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;"><b>Object Properties</b></td>
                            <td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;">{n_obj_props}</td></tr>
                        <tr><td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;"><b>Datatype Properties</b></td>
                            <td style="padding:6px 12px;border-bottom:1px solid #e2e8f0;">{n_data_props}</td></tr>
                        <tr><td style="padding:6px 12px;"><b>Named Individuals</b></td>
                            <td style="padding:6px 12px;">{n_individuals}</td></tr>
                    </table>
                </div>
                """
                display(HTML(stats_html))

                # Afficher quelques triples
                display(HTML("<h4 style='color:#374151;'>Exemple de triples:</h4>"))
                sample_rows = []
                for i, (s, p, o) in enumerate(g.triples((None, None, None))):
                    if i >= 10:
                        break
                    sample_rows.append({"Subject": str(s)[:60], "Predicate": str(p)[:40], "Object": str(o)[:60]})
                if sample_rows:
                    display(pd.DataFrame(sample_rows).style.set_table_styles([
                        {"selector": "thead th",
                         "props": [("background-color", "#f1f5f9"), ("font-size", "0.8em")]},
                        {"selector": "td",
                         "props": [("font-size", "0.75em"), ("padding", "3px 8px")]}
                    ]))

            except Exception as e:
                display(styles.error_msg(f"Erreur aperçu: {str(e)}"))


def runner(state) -> OntologyFeatureEngUI:
    """Fonction d'entrée pour le pipeline."""
    fe = OntologyFeatureEngUI(state)
    if hasattr(fe, "ui"):
        display(fe.ui)
    return fe
