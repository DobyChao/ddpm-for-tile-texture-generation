import base64

from flask import Flask, render_template, request, jsonify
from sample_web import sample


def start_server(host='127.0.0.1', port=8888):
    app = Flask(__name__)

    @app.route('/Image_Generate', methods=['GET', 'POST'])
    def Image_Generate():
        print("Image_Generate, ", request.method)
        if request.method == 'POST':
            dic = dict(request.form)
            opt = dict(
                shape=int(dic["size"]),
                choice=int(dic['num']),
            )
            # img_result = sample(opt)
            # html = "data:image/png;base64,{}"  # html代码
            # htmlstr = html.format(img_result)  # 添加数据
            # return htmlstr
            img_path = sample(opt)
            return img_path
        return render_template("demo.html")

    app.run(host=host, port=port, debug=True)


start_server(port=80)
