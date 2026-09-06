from autotransition.generative_dance.video import _concat_filter_graph


def test_concat_filter_graph_names_and_consumes_every_input() -> None:
    graph = _concat_filter_graph(
        4,
        input_filter="scale=480:832,format=yuv420p",
        output_filter="fps=24,format=yuv420p",
    )

    assert "concat=n=4:v=1:a=0" in graph
    assert graph.count(":v:0]scale=480:832,format=yuv420p") == 4
    assert graph.count("[concat") == 8
