var redraw, g, renderer;



// gets called when cross-domain server responds
function __parseJSONPResponse(data) {
    var width = $(document).width() - 20;
    var height = $(document).height() - 60;
    g = new Graph();

    nfa = JSON.parse(data);
    g.addNode("START");
    for (var i = 0; i < nfa.length; i++) {
        g.addNode(nfa[i].id, { label: nfa[i].data });
    }
    for (var i = 0; i < nfa.length; i++) {
        if (nfa[i].out != -1) {
            g.addEdge(nfa[i].id, nfa[i].out, { directed: true });
        }
        if (nfa[i].out1 != -1) {
            g.addEdge(nfa[i].id, nfa[i].out1, { directed: true });
        }
    }
    g.addEdge("START", nfa[0].id, {directed: true });


    /* layout the graph using the Spring layout implementation */
    var layouter = new Graph.Layout.Spring(g);

    /* draw the graph using the RaphaelJS draw implementation */
    renderer = new Graph.Renderer.Raphael('canvas', g, width, height);

    redraw = function() {
        layouter.layout();
        renderer.draw();
    };
}

window.onload = function() {
    /*JSONp magic*/
    var script = document.createElement('script');
    script.src = 'http://128.237.92.16:3000/nfa.json';  
    document.body.appendChild(script);
};
