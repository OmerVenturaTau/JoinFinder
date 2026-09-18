# results_analysis/clusters

Geniza graph analysis scripts.

- `geniza_graphs_sql.py`: shared SQL/Excel graph builders and HTML exporters.
- `geniza_manuscript_graph.py`: manuscript-level graph CLI.
- `geniza_image_ego_graph.py`: image-level graph CLI.
- `geniza_manuscript_cliques.py`: filters manuscript graphs to clique-like
  subgraphs from generated HTML.
- `geniza_inter_library_manuscript_graph.py`: keeps only links between
  manuscripts held by different known libraries.

Interactive manuscript and image graphs include matching "From libraries" and
"To libraries" checkbox dropdowns. Both menus close on outside click or Escape.
In either menu, "All libraries" selects every individual library checkbox;
clicking it again clears that entire selection, making it easy to start with all
libraries and exclude only a few. Edges are matched as undirected connections
between the selected From and To library sets. Choosing the same library on
both sides shows intra-library links. Unrelated nodes are hidden, while nodes
from explicitly selected libraries receive a stronger border and shadow. A
single selected library automatically focuses its first matching node and may
zoom in to a readable scale, but never zooms out. For one or multiple libraries,
the "Focus result" button cycles through matching nodes in stable library,
shelfmark, and ID order and displays the current position as `N / total`.
Changing join, similarity, or weight filters never moves the camera. If no nodes
match, the detail panel says so explicitly. Node counts are displayed beside
every library, and the controls report the matching edge count.
Node labels and detail panels include the manuscript ID, shelfmark, and library
when that metadata is available. Manuscript graphs use inclusive From/To
edge-weight bounds on one dual-handle slider, with editable numeric bounds on
its left and right; setting both bounds to the same value performs an exact
weight filter. The source and destination library controls use matching widths.
The "Find joins" action is a toggle with a hover explanation and a fixed-width,
visually distinct pressed state that does not move the toolbar when activated.
It preserves the current From/To library, similarity, and edge-weight filters,
then additionally requires each result to connect two different known
libraries. The filters can be adjusted while joins mode remains active; clicking
the button again removes only the inter-library requirement.

Graph interpretation:

- Image-level graphs use image/page rows as nodes and KNN similarities as edges.
- Manuscript-level graphs aggregate image-neighbor evidence into manuscript nodes;
  edge weight is evidence that pages/fragments from two manuscripts are close in
  the learned latent space.
- Same-manuscript neighbors and other-manuscript neighbors come from different
  DB output tables, so check which table/source a script is using before drawing
  conclusions.
- Communities or cliques are graph summaries over learned latent similarities,
  not direct OCR/text matches.

Generated `.html` files in this folder are snapshots, not library code.
