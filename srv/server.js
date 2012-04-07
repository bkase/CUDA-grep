var express = require("express");
var fs = require('fs');

var server = express.createServer();
server.get("/nfa.json", function(req, res) {
    
    
    fs.readFile('nfa.json', function(err,data){
      if(err) {
        console.error("Could not open file: %s", err);
        process.exit(1);
      }

      res.writeHead(200, {'Content-Type': 'application/javascript'});
      res.end("__parseJSONPResponse(" + JSON.stringify(data.toString()) + ");");
    
    });
    
});
server.listen(3000);
