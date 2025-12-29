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

// List of section IDs that correspond to nav items (in order)
const sectionIds = ['#title', '#events', '#oh', '#syllabus', '#lectures', '#recitations', '#homework', '#docs-tools'];

function setActiveNavItem(target) {
    // Remove active class from all nav items
    $('.header-item').removeClass('active');
    $('.m-header-item').removeClass('active');
    
    // Find and activate the matching nav item
    $('.header-item').each(function() {
        if ($(this).attr('onclick') && $(this).attr('onclick').includes(target)) {
            $(this).addClass('active');
        }
    });
    $('.m-header-item').each(function() {
        if ($(this).attr('onclick') && $(this).attr('onclick').includes(target)) {
            $(this).addClass('active');
        }
    });
}

function toSection(target) {
    var $target = $(target);
    
    // If element doesn't exist yet (dynamically loaded content), wait for it
    if ($target.length === 0) {
        var attempts = 0;
        var maxAttempts = 20; // Wait up to 2 seconds (20 * 100ms)
        var waitForElement = setInterval(function() {
            $target = $(target);
            attempts++;
            if ($target.length > 0) {
                clearInterval(waitForElement);
                var offset = $target.offset();
                var scrollto = offset.top - estimatedHeaderHeight;
                $('html, body').animate({scrollTop:scrollto}, 100);
                setActiveNavItem(target);
            } else if (attempts >= maxAttempts) {
                clearInterval(waitForElement);
                console.warn('Element not found: ' + target);
            }
        }, 100);
    } else {
        var offset = $target.offset();
        var scrollto = offset.top - estimatedHeaderHeight;
        $('html, body').animate({scrollTop:scrollto}, 100);
        setActiveNavItem(target);
    }
}

function mobileToSection(target) {
    hideMenu();
    toSection(target);
}

function showChat() {
  var chatBox = document.getElementById("chatBox");
  chatBox.classList.toggle("show");
}

// Scroll spy: highlight nav item based on scroll position
$(document).ready(function() {
    $(window).on('scroll', function() {
        var scrollPos = $(window).scrollTop() + estimatedHeaderHeight + 50; // offset for better UX
        var currentSection = null;
        
        // Find the current section based on scroll position
        for (var i = sectionIds.length - 1; i >= 0; i--) {
            var section = $(sectionIds[i]);
            if (section.length && section.offset().top <= scrollPos) {
                currentSection = sectionIds[i];
                break;
            }
        }
        
        // Default to first section if at top
        if (!currentSection && scrollPos < 200) {
            currentSection = '#title';
        }
        
        if (currentSection) {
            setActiveNavItem(currentSection);
        }
    });
    
    // Set initial active state
    setActiveNavItem('#title');
});
