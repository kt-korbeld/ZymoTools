"""
Shiny app for visually inspecting hotspot patches in NGL.
Load a PDB and a patch_summaries.csv as produced by the hotspots command.
select patch rows to highlight their residues in red, optionally type target residues
to mark them yellow, and export the selected rows as a curated CSV. 
The exported can be fed into the screen phase.
"""

import ast
import base64
import re

import pandas as pd
from shiny import App, reactive, render, ui


NGL_HTML = """
<div id="viewport" style="width:100%; height:650px; border:1px solid #ddd;"></div>
<script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
<script>
var stage = new NGL.Stage("viewport", {backgroundColor: "white"});
window.addEventListener("resize", function(){ stage.handleResize(); }, false);

var hotspotReprs = [];
var targetReprs = [];
var baseSurfaceRepr = null;
var baseCartoonRepr = null;
var lastHighlight = null;

function clearReprs(arr) {
    arr.forEach(function(r) {
        try { if (r && r.parent) r.parent.removeRepresentation(r); } catch(e) {}
    });
    arr.length = 0;
}

// Avoid z-fighting between overlapping surfaces by carving the lower layer's
// selection to exclude residues that an upper layer will draw on top. Yellow
// wins over red so that target coverage stays visible inside hotspot patches.
function applyHighlight(message) {
    if (stage.compList.length === 0) return;
    var comp = stage.compList[0];
    clearReprs(hotspotReprs);
    clearReprs(targetReprs);

    var red = message.red_residues || [];
    var yellow = message.yellow_residues || [];

    var baseSele;
    if (red.length + yellow.length > 0) {
        baseSele = "not (" + red.concat(yellow).join(" or ") + ")";
    } else {
        baseSele = "*";
    }
    if (baseSurfaceRepr) baseSurfaceRepr.setSelection(baseSele);
    if (baseCartoonRepr) baseCartoonRepr.setSelection(baseSele);

    if (red.length > 0) {
        var redCore = "(" + red.join(" or ") + ")";
        if (yellow.length > 0) {
            redCore = "(" + redCore + " and not (" + yellow.join(" or ") + "))";
        }
        var seleR = redCore + " and protein";
        hotspotReprs.push(comp.addRepresentation("cartoon", {sele: seleR, color: "red"}));
        hotspotReprs.push(comp.addRepresentation("surface", {
            sele: seleR, color: "red", opacity: 1, side: "double"
        }));
    }
    if (yellow.length > 0) {
        var seleY = "(" + yellow.join(" or ") + ") and protein";
        targetReprs.push(comp.addRepresentation("cartoon", {sele: seleY, color: "yellow"}));
        targetReprs.push(comp.addRepresentation("surface", {
            sele: seleY, color: "yellow", opacity: 1, side: "double"
        }));
    }
}

Shiny.addCustomMessageHandler("load_pdb", function(message) {
    stage.removeAllComponents();
    hotspotReprs = [];
    targetReprs = [];
    baseSurfaceRepr = null;
    baseCartoonRepr = null;

    var byteChars = atob(message.file_content);
    var byteNumbers = new Array(byteChars.length);
    for (var i = 0; i < byteChars.length; i++) {
        byteNumbers[i] = byteChars.charCodeAt(i);
    }
    var byteArray = new Uint8Array(byteNumbers);
    var blob = new Blob([byteArray], {type: "text/plain"});

    stage.loadFile(blob, {ext: "pdb"}).then(function(comp) {
        baseCartoonRepr = comp.addRepresentation("cartoon", {color: "lightgrey"});
        baseSurfaceRepr = comp.addRepresentation("surface", {
            color: "lightgrey", opacity: 0.3, side: "double"
        });
        comp.autoView();
        if (lastHighlight) applyHighlight(lastHighlight);
    });
});

Shiny.addCustomMessageHandler("highlight_residues", function(message) {
    lastHighlight = message;
    applyHighlight(message);
});
</script>
"""


app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("pdbfile", "PDB structure", accept=[".pdb"]),
        ui.input_file("csvfile", "Hotspot patches CSV", accept=[".csv"]),
        ui.hr(),
        ui.input_text(
            "target_residues",
            "Target residues (yellow)",
            placeholder="e.g. 25, 30, 47-50",
        ),
        ui.help_text("Comma-separated residue numbers; ranges with '-' allowed."),
        ui.hr(),
        ui.download_button("download_filtered", "Download selected rows as CSV"),
        ui.help_text("Exports the currently selected rows. If nothing is selected, exports the full table."),
        width=340,
    ),
    ui.h2("Hotspot patch visualizer"),
    ui.layout_columns(
        ui.HTML(NGL_HTML),
        ui.output_data_frame("hotspot_table"),
        col_widths=[7, 5],
    ),
)


def parse_residues_field(value):
    """
    Parse a residues cell like '[25, 30, 47]' into a list[int].
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = ast.literal_eval(value)
        return [int(x) for x in parsed]
    except (ValueError, SyntaxError, TypeError):
        return []


def parse_target_residues(text):
    """
    Parse user text like '25, 30, 47-50' into a list[int].
    """
    if not text:
        return []
    result = []
    for token in re.split(r"[,\s]+", text.strip()):
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            try:
                lo, hi = int(a), int(b)
                result.extend(range(min(lo, hi), max(lo, hi) + 1))
            except ValueError:
                continue
        else:
            try:
                result.append(int(token))
            except ValueError:
                continue
    return result


def server(input, output, session):

    @reactive.calc
    def patches_df():
        f = input.csvfile()
        if not f:
            return pd.DataFrame()
        d = pd.read_csv(f[0]["datapath"])
        # Drop the unnamed pandas-index column written by the producer script.
        d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
        return d

    @render.data_frame
    def hotspot_table():
        d = patches_df()
        if d.empty:
            return render.DataGrid(d, selection_mode="rows")
        preferred = ["patch_id", "tot_score", "patch_size", "residue_labels", "residues"]
        cols = [c for c in preferred if c in d.columns] + \
               [c for c in d.columns if c not in preferred]
        return render.DataGrid(
            d[cols],
            selection_mode="rows",
            width="100%",
            height="650px",
        )

    @reactive.effect
    async def loadpdb():
        fileinfo = input.pdbfile()
        if not fileinfo:
            return
        with open(fileinfo[0]["datapath"], "rb") as f:
            content = f.read()
        await session.send_custom_message("load_pdb", {
            "file_content": base64.b64encode(content).decode("utf-8"),
        })

    @reactive.effect
    async def update_highlight():
        red = []
        try:
            sel_df = hotspot_table.data_view(selected=True)
        except Exception:
            sel_df = pd.DataFrame()
        if not sel_df.empty and "residues" in sel_df.columns:
            for value in sel_df["residues"]:
                red.extend(parse_residues_field(value))

        yellow = parse_target_residues(input.target_residues())

        await session.send_custom_message("highlight_residues", {
            "red_residues": [str(r) for r in red],
            "yellow_residues": [str(r) for r in yellow],
        })

    @render.download(filename="filtered_patches.csv")
    def download_filtered():
        try:
            sel_df = hotspot_table.data_view(selected=True)
        except Exception:
            sel_df = pd.DataFrame()
        if sel_df.empty:
            sel_df = patches_df()
        yield sel_df.to_csv(index=False)


app = App(app_ui, server)
