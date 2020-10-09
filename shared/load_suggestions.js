
var publicSpreadsheetUrl = 'https://docs.google.com/spreadsheets/d/1sbyMINQHPsJctjAtMW0lCfLrcpMqoGMOJj6AN-sNQrc/pubhtml';

function init() {
  Tabletop.init( { key: "https://docs.google.com/spreadsheets/d/1ES1YOWYfr7XbXl9aHWI2-77yRglYEfK_PNrsC5HkBnI/edit?usp=sharing",
                   callback: showInfo,
                   simpleSheet: true } );
}

function showInfo(data, tabletop) {

  document.querySelector("#ideas").innerHTML = "";
  for (i in data) {
    // select
    var t = document.querySelector('#mytemplate');

    // set
    t.content.querySelector('.title').textContent = data[i]['Project-Title'];
    t.content.querySelector('.key').textContent = data[i]['Keywords'];
    t.content.querySelector('.abs').textContent = data[i]['Abstract'];
    t.content.querySelector('.avail').textContent= data[i]['Available'];
    t.content.querySelector('.contact').textContent= data[i]['Contact'];

    // add to document DOM
    var clone = document.importNode(t.content, true); // where true means deep copy
    document.querySelector("#ideas").appendChild(clone);
  }
}

window.addEventListener('DOMContentLoaded', init)
