"use strict";
var page = require('webpage').create();
var fs = require('fs');

var data = [{
			"position": "absolute",
			"left": 110,
			"top": 110,
			"font-size": 40,
      "angle": 20
		} ];
var width = 28;
var height = 28;

var content = fs.read('./template_one_line.html');
page.content = content;
page.evaluate(function(w, h) {
  document.body.style.width = w + "px";
  document.body.style.height = h + "px";

  var target =  document.getElementById('target');
  target.transform =  "translate(-50%,-50%) rotate(-20deg)";

}, width, height);
page.clipRect = {top: 0, left: 0, width: width, height: height};  
page.render('../imi-masterarbeit-data/text/text1.png');
phantom.exit();
