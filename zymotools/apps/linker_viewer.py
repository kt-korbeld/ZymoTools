"""
Shiny app for interactively estimating fusion-linker length.
Upload a PDB, pick a start and end residue (typically the two chain termini to
be fused), and the app traces the shortest solvent path around the protein and
reports the minimum linker length in residues.
"""

import base64

import MDAnalysis
from shiny import App, ui, reactive, render

from ..linker import shortest_linker_path, residues_for_length


app_ui = ui.page_fluid(
    ui.h2("Calculate Linker Length"),
    ui.input_file("pdbfile", "Upload a PDB file", accept=[".pdb"]),
    ui.output_ui("ngl_viewer"),
    ui.HTML("""
      <div id="viewport" style="width:800px; height:600px;"></div>
      <script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
      <script>
      var stage = new NGL.Stage("viewport");
      window.addEventListener("resize", function(){ stage.handleResize(); }, false);
      Shiny.addCustomMessageHandler("load_pdb", function(message) {
          stage.removeAllComponents();
          var byteChars = atob(message.file_content);
          var byteNumbers = new Array(byteChars.length);
          for (var i = 0; i < byteChars.length; i++) {
              byteNumbers[i] = byteChars.charCodeAt(i);
          }
          var byteArray = new Uint8Array(byteNumbers);
          var blob = new Blob([byteArray], {type: "text/plain"});
          stage.loadFile(blob, { ext: "pdb" }).then(function(comp) {
              comp.addRepresentation("cartoon");
              comp.autoView();
              message.coords.forEach(function(c) {
                  var shape = new NGL.Shape("marker");
                  shape.addSphere([c.x, c.y, c.z], [1,0,0], 1.0);
                  stage.addComponentFromObject(shape).addRepresentation("surface");
              });
          });
      });
      </script>
      """),
    ui.layout_columns(
        ui.card(
            ui.card_header("Start Path"),
            ui.input_select("sel_chain1", "Select protein chain", {"A": "Chain A"}),
            ui.input_numeric("startres", "Select Residue", 1000, min=1, max=1000),
            ui.output_text("res_val_1"),
        ),
        ui.card(
            ui.card_header("End Path"),
            ui.input_select("sel_chain2", "Select protein chain", {"A": "Chain A"}),
            ui.input_numeric("endres", "Select Residue", 1, min=1, max=1000),
            ui.output_text("res_val_2"),
        ),
    ),
    ui.input_action_button("calcpath", "Calculate Shortest Path"),
    ui.output_text_verbatim("showlen"),
)


def server(input, output, session):
    @render.text
    def res_val_1():
        return "Current selection: ChainID {} and Residue {}".format(
            input.sel_chain1(), input.startres())

    @render.text
    def res_val_2():
        return "Current selection: ChainID {} and Residue {}".format(
            input.sel_chain2(), input.endres())

    @reactive.effect
    async def loadpdb():
        fileinfo = input.pdbfile()
        if not fileinfo:
            return
        filepath = fileinfo[0]["datapath"]
        u_in = MDAnalysis.Universe(filepath)
        chainids = list(set().union(*[set(i) for i in u_in.segments.chainIDs]))
        chain_1 = input.sel_chain1()
        chain_2 = input.sel_chain2()
        ui.update_select("sel_chain1", choices={i: "Chain {}".format(i) for i in chainids}, selected=chain_1)
        ui.update_select("sel_chain2", choices={i: "Chain {}".format(i) for i in chainids}, selected=chain_2)
        c1 = u_in.select_atoms("protein and chainID {}".format(chain_1))
        c2 = u_in.select_atoms("protein and chainID {}".format(chain_2))
        c1_max, c1_min = max(c1.residues.resids), min(c1.residues.resids)
        c2_max, c2_min = max(c2.residues.resids), min(c2.residues.resids)
        ui.update_numeric("startres", min=int(c1_min), max=int(c1_max),
                          value=int(min(max(input.startres(), c1_min), c1_max)))
        ui.update_numeric("endres", min=int(c2_min), max=int(c2_max),
                          value=int(min(max(input.endres(), c2_min), c2_max)))
        with open(filepath, "rb") as f:
            content = f.read()
        b64 = base64.b64encode(content).decode("utf-8")
        await session.send_custom_message("load_pdb", {"file_content": b64, "coords": []})

    @reactive.calc
    @reactive.event(input.calcpath)
    def path():
        fileinfo = input.pdbfile()
        filepath = fileinfo[0]["datapath"]
        u_in = MDAnalysis.Universe(filepath)
        sel_start = "chainID {} and resid {}".format(input.sel_chain1(), input.startres())
        sel_end = "chainID {} and resid {}".format(input.sel_chain2(), input.endres())
        return shortest_linker_path(
            u_in, sel_start, sel_end, gridstep=1, padding=4, rad=3,
            progress=lambda m: print("[linker]", m),
        )

    @reactive.effect
    @reactive.event(input.calcpath)
    async def renderpath():
        try:
            fileinfo = input.pdbfile()
            filepath = fileinfo[0]["datapath"]
            with open(filepath, "rb") as f:
                content = f.read()
            _, path_u = path()
            coords = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                      for p in path_u.atoms.positions]
            b64 = base64.b64encode(content).decode("utf-8")
            await session.send_custom_message("load_pdb", {"file_content": b64, "coords": coords})
        except Exception as e:  # noqa: BLE001
            print("error in renderpath", e)

    @output
    @render.text
    async def showlen():
        if input.calcpath() == 0:
            return "Calculate shortest path to estimate minimum linker length"
        length, _ = path()
        minimum, buffered = residues_for_length(length, buffer=5)
        return (
            "Minimum linker length is {}Å\n"
            "This requires at least {} amino acids (3.8Å/aa).\n"
            "Add 5 as a buffer: {} amino acids".format(round(length, 2), minimum, buffered)
        )


app = App(app_ui, server)
