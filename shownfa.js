var redraw, g, renderer;

window.onload = function() {
    var width = $(document).width() - 20;
    var height = $(document).height() - 60;

    g = new Graph();

    g.addEdge("Char c", "Split", { directed: true });

    /* layout the graph using the Spring layout implementation */
    var layouter = new Graph.Layout.Spring(g);

    /* draw the graph using the RaphaelJS draw implementation */
    renderer = new Graph.Renderer.Raphael('canvas', g, width, height);

    redraw = function() {
        layouter.layout();
        renderer.draw();
    };
    hide = function(id) {
        g.nodes[id].hide();
    };
    show = function(id) {
        g.nodes[id].show();
    };
};
