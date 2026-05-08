from __future__ import annotations

import gradio as gr

import gui_wrapper
import stills_extractor_app


with gr.Blocks(title="Splatter Unified App") as demo:
    gr.Markdown("## Splatter")
    gr.Markdown(
        "Use one app for the full workflow: extract stills + COLMAP, train splats, and (soon) display splats."
    )

    with gr.Tabs():
        with gr.Tab("Produce Stills and COLMAP Files"):
            stills_extractor_app.demo.render()

        with gr.Tab("Train Splat"):
            gui_wrapper.demo.render()

        with gr.Tab("Display Splat"):
            gr.Markdown("### Display Splat (Coming Soon)")
            gr.Markdown(
                "This tab is reserved for an interactive splat viewer.\n\n"
                "Planned: load trained runs and preview/export renders from one place."
            )


if __name__ == "__main__":
    demo.queue().launch()

