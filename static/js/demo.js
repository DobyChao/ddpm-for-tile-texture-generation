var show = document.getElementById("show_img");
var input = document.getElementById("input");
var size = document.getElementsByClassName("size")
var strength = document.getElementById("strength")
console.log(input, input.value)
function showImage() {
    var Id = input.value
    console.log('更换图片', Id);
    var img_src;
    if (Id >= 0 && Id < 86) img_src = "static/image/" + Id + ".jpg";
    else img_src = "static/image/default.png";
    console.log(img_src)
    show.src = img_src;
    console.log(show.src)
}

function tempFunc() {
    var formData = new FormData();
    var shape = $("input[name='size']:checked").val()
    console.log(shape)
    formData.append("num", input.value);
    formData.append("size", shape);
    console.log(formData);
    // startProgressBar()
    // 显示
    // $("#loading_text").text("sketch精彩即将展现...").fadeIn(4000);
    // loadIn()
    // console.log("获取图片成功");
    $.ajax({
        type: 'POST',
        url: '/Image_Generate',
        data: formData,
        processData: false,//数据处理
        contentType: false,//内容类型
        cache: false,
        async: true,
        success: function (data) {
            $("#change_img").attr("src", data);
            console.log("转化成功");
            // cancelProgressBar()
        },
        error: function (error) {
            alert(error)
        }
    });

}




