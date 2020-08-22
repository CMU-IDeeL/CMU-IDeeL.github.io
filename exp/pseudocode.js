$(window).on("load", function() {
    var tbody = document.querySelector("#lec4Accordion");
    var template = document.querySelector('#pseudocode-listing');
    console.log(tbody, template)

    // Clone the new row and insert it into the table
    var clone = template.content.cloneNode(true);
    var body = clone.querySelector(".card-body");
    var script = clone.querySelector(".script")
    tbody.appendChild(clone);

    var item = tbody.querySelector("")
    src = "https://emgithub.com/embed.js?target=https%3A%2F%2Fgithub.com%2FCMU-IDeeL%2Fpseudocode%2Fblob%2Fmaster%2Fbackprop%2Flec4_forward_pass.plaintext&style=github&showLineNumbers=on&showFileMeta=on"

    /*
    td[0].textContent = "1235646565";
    td[1].textContent = "Stuff";
    */


    // Clone the new row and insert it into the table
    /*var clone2 = template.content.cloneNode(true);
    td = clone2.querySelectorAll("td");
    td[0].textContent = "0384928528";
    td[1].textContent = "Acme Kidney Beans 2";

    tbody.appendChild(clone2);*/
})
