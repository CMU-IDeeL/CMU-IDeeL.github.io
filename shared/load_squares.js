for (var i in projects) {
    var project = projects[i];
    // select
    var t = document.querySelector('#project-square');

    t.content.querySelector('.project').dataset.target = "#modal" + i;
    t.content.querySelector('.modal').id = "modal" + i;
    t.content.querySelector('.modal').id = "modal" + i;
    t.content.querySelector('.modal').setAttribute('aria-labelledby', "modal" + i + "label");
    t.content.querySelector('.modal-title').id = "modal" + i + "label";


    // set
    t.content.querySelector('.project-title').textContent = project['title'];
    t.content.querySelector('.modal-title').textContent = project['title'];
    t.content.querySelector('.team').textContent = project['team'];
    t.content.querySelector('.authors').textContent= project['authors'];
    t.content.querySelector('.summary-text').textContent = project['summary'];
    t.content.querySelector('.report').href = project['report'];
    t.content.querySelector('.video').href = project['video'];
    t.content.querySelector('.modal-image').src = "pics/" + project['pic'];
    t.content.querySelector('.teaser').src = "pics/" + project['pic'];

    // add to document DOM
    var clone = document.importNode(t.content, true); // where true means deep copy
    if (project['top5'] === true) {
        console.log("top 5")
        clone.querySelector(".project").classList.add("top-project");
    }

    document.querySelector("#projs").appendChild(clone);
}
