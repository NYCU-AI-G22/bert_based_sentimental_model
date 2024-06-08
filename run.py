import numpy as np
from text_generation import generate_emotional_text

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/text_generation', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    text = insertValues['input_text']

    result = generate_emotional_text(text)

    return jsonify({'return': str(result)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)