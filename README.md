# Info to know

A brain dump about the website... feel free to disregard if you choose a different direction...


## Transitioning between semesters

1. Copy the last semester's folder, e.g. copy F70 to S71
2. Change the index.html's URL for the new semester, e.g. url=./S71/index.html (to update main page)
3. On the mobile menus and desktop headers for the past and current websites, add a link to the new semester
4. Yeet out the content from the new website!


## Organization

- F20: F20 website
- S20: S20 website
- index.html: just links to index.html of current semester
- exp: you can push experimental changes here, and then link other course staff to https://deeplearning.cs.cmu.edu/exp/
- shared: pages that don't belong to a particular semester
  - favicon.ico: website icon, maybe should be the LTI icon
  - project.css, project.html, projects.js, load_suggestions.js: F20 specific project information, maybe should be in F20
  - S20TAs.png, TAs.css, TAs.html: stuff for the acknowledgements page
  - pics, load_squares, projects.js, gallery.html: stuff for project gallery. If it looks good, would be cool to link from a more central place!


## Pseudocode Page

http://deeplearning.cs.cmu.edu/F20/pseudocode.html
We got a decent amount done, but didn't finish or verify... see the pseudocode repository for more. Once this gets finished, it would be a good idea for the slides to have code listing numbers, so that this is maintanable. Also, note that this stuff is from S20 lectures. We have a spreadsheet with info about which slides the codes came from.


## Technologies

- The main page is built with bootstrap 3, it would be cool to create a new branch, update to latest bootstrap, and then merge. But perhaps time consuming / not necessary
- Most other pages are with bootstrap 4, font sizes are bigger by default, there's more features...
- Acknowledgements page should probably be ported to bootstrap, right now it's just html/css...
- If you continue with the table calendar, it takes a while to make if you do it from scratch; you can make an outline of the table with www.tablesgenerator.com (or just edit the current one)


## Misc

- We probably don't need to link quizzes in lectures table, just in the calendar table / bulletin. Not sure it's worth the effort.
- The grand vision for the bulletin is that if there's something due soon, you can go to the bulletin and it links to everything relevant about that assignment... piazza posts, kaggles, autolab, canvas assignment, etc. 
- The assignments table should mirror all of these links ^
- If you continue with the chatbot, Akshat has information about it

Or do your own thing :)
