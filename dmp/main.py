import typer

import dmp.commands.app_infer as infer

app = typer.Typer(
    name="dmp_app",
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)
app.add_typer(infer.app, name="infer", help="Inference commands")


@app.callback()
def explain():
    """

    dmp_app: Deep Learning Model Performance Analysis
    This is a command line tool for analyzing the performance of deep learning models.
    It provides commands for inference and evaluation of models.
    """


if __name__ == "__main__":
    app()
