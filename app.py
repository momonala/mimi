from os import environ
from flask import Flask
from flask import send_from_directory
from flask import jsonify
from flask import request
import numpy as np
import uuid
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)


def start_x(d, fnt, base, text):
    text_size = d.textsize(text, fnt)[0]
    return base.size[0] / 2.0 - text_size / 2.0


def bottom_y(d, fnt, base, text):
    text_size = d.textsize(text, fnt)[1]
    return base.size[1] - text_size - 10


def get_font_size(txt_layer, text):
    from PIL import ImageDraw, ImageFont
    font_size = 40
    for _font_size in range(40, 10, -1):
        fnt = ImageFont.truetype('impact.ttf', _font_size)
        d = ImageDraw.Draw(txt_layer)
        length = d.textsize(text, fnt)[0]
        font_size = _font_size
        if length < 240:
            break
    return font_size


def select_random_meme():
    from glob import glob
    img_lib = glob('meme/*')
    return img_lib[np.random.randint(0, len(img_lib))]


def generate_image(eye_d, top_text, bottom_text):
    top_uppercase = top_text.upper()
    bottom_uppercased = bottom_text.upper()
    base = Image.open(select_random_meme()).convert("RGBA")
    txt = Image.new('RGBA', base.size, (255, 255, 255, 0))
    top_fnt = ImageFont.truetype('impact.ttf', get_font_size(txt, top_uppercase))
    bottom_fnt = ImageFont.truetype('impact.ttf', get_font_size(txt, bottom_uppercased))
    d = ImageDraw.Draw(txt)
    draw_text(d, start_x(d, top_fnt, base, top_uppercase), 5, top_uppercase, get_font_size(txt, top_uppercase))
    draw_text(d, start_x(d, bottom_fnt, base, bottom_uppercased), bottom_y(d, bottom_fnt, base, bottom_uppercased), bottom_uppercased, get_font_size(txt, bottom_uppercased))
    # d.multiline_text((start_x(d, top_fnt, base, top_uppercase), 5), top_uppercase, font=top_fnt,
    #                  fill='white', align="center")
    # d.multiline_text((start_x(d, bottom_fnt, base, bottom_uppercased),
    #                   bottom_y(d, bottom_fnt, base, bottom_uppercased)),
    #                  bottom_uppercased, font=bottom_fnt, fill='white', align="center")
    out = Image.alpha_composite(base, txt)
    out.save('meme/' + eye_d + '.png', 'PNG')


@app.route('/')
def health():
    return 'healthy'


@app.route('/meme/<string:image_file_name>')
def meme_image(image_file_name):
    return send_from_directory('meme', image_file_name)


from predict_text import load_model, generate_text

loaded_model = load_model()


def generate_bottom_text(text):
    return generate_text(text, loaded_model)


def draw_text(draw, x, y, text, pointsize):
    fillcolor = "white"
    shadowcolor = "black"
    font = 'impact.ttf'
    font = ImageFont.truetype(font, pointsize)
    # thicker border
    draw.text((x - 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y - 1), text, font=font, fill=shadowcolor)
    draw.text((x - 1, y + 1), text, font=font, fill=shadowcolor)
    draw.text((x + 1, y + 1), text, font=font, fill=shadowcolor)
    # now draw the text over it
    draw.text((x, y), text, font=font, fill=fillcolor)
    return draw


@app.route('/stub', methods=['POST'])
def stub():
    json = request.get_json()
    eye_d = str(uuid.uuid4())
    message_ = json['item']['message']['message'][6:]
    generate_image(eye_d, message_, generate_bottom_text(message_))
    return jsonify({
        "message": ".",
        "notify": False,
        "card": {
            "style": "image",
            "id": eye_d,
            "title": message_,
            "thumbnail": {
                "url": "http://35.227.204.93/meme/" + eye_d + ".png",
                "width": 250,
                "height": 250},
            "url": "http://35.227.204.93/meme/" + eye_d + ".png"},
        "message_format": "html"})


if __name__ == '__main__':
    port = int(environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
