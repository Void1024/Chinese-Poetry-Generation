
$(document).ready(function(){

    $(".non-selectable").click(
        function(e) {
            generate()
        }
    )
    $('.emotion-nonselect').click(
        function(e) {
            $('#select').css('background-color','#ffffff').css('color','#C12A2B').removeAttr('id')
            $(e.target).attr('id','select').css('background-color','#C12A2B').css('color','#ffffff')
        }
    )
})

function generate() {
    emotion = $('#select').text()
    if(emotion === '') {
        emotion = '无'
    }
    query = $('#user_input').val()
    topic = query + '#' + emotion
    alert(topic)
}

function show(poetry) {
    poetry = poetry.split(',')
    if(poetry.length != 4) {
        return
    }
    for(i = 0;i < 4;i++) {
        id = '#line' + (i + 1)
        char_list = $(id).children('td')
        for(j = 0;j < 7;++j) {
            $(char_list[j]).text(poetry[i][j])
        }
    }
}

function get() {
    $(function(){
        $.ajax({
            url: 'test.txt',
            dataType: 'text',
            success: function(data) {
                alert(data);
            }
        });
    });
}