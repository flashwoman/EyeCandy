function toPage(page) {
    $('.page').removeClass('active')
    $('#page-'+page).addClass('active')
}
function openFile() {
    $('#file').click()
}
$('document').ready(function () {
    $('#file').on('change', function (e) {
        $('#image-viewer-text').hide()
        readURL(this)
    })
})

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#image').attr('src', e.target.result);
            $('#figured-image').attr('src', e.target.result);
        }

        reader.readAsDataURL(input.files[0]);
    }
}


