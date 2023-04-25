function showMenu() {
    console.log("Menu Shown");
    $('.mobile-menu').addClass('vis');
    $('.menu-backdrop').addClass('vis');
    //$('#mobile-menu, #menu-backdrop').css('display', 'block');
}

function hideMenu() {
    $('.mobile-menu').removeClass('vis');
    $('.menu-backdrop').removeClass('vis');
}

const estimatedHeaderHeight = 60;
function toSection(target) {
    var offset = $(target).offset();
    var scrollto = offset.top - estimatedHeaderHeight;
    $('html, body').animate({scrollTop:scrollto}, 100);
}

function mobileToSection(target) {
    hideMenu();
    toSection(target);
}

function showChat() {
  var chatBox = document.getElementById("chatBox");
  chatBox.classList.toggle("show");
}
