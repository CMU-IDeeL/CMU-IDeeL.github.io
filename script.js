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

var flag = 0
function PopThisUp() {
  var popup = document.getElementById("myPopup");
  if (flag == 0){
    var iframe = document.createElement('iframe');
    iframe.allow = "microphone"
    iframe.width = "300"
    iframe.height = "430"
    iframe.src = "https://console.dialogflow.com/api-client/demo/embedded/11785-Fall2020"
    popup.appendChild(iframe) 
    flag = 1
  }
  popup.classList.toggle("show");
}
