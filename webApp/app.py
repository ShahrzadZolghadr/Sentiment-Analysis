from flask import Flask, request, jsonify
import utils_prepro
import pre_bilstm
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def convert_to_tokens(data):
	return pad_sequences(pre_bilstm.tokenizer.texts_to_sequences([data]), pre_bilstm.maxlen)
 
app = Flask(__name__)

@app.route('/predict',methods=['POST'])
def analyse():
  
	pred = None
	if not request.json or not 'rawtext' in request.json:
		abort(400)
	rawtext = request.json['rawtext']

	inp = convert_to_tokens(utils_prepro.apply_cleaning(rawtext))
	model_SA_ = keras.models.load_model('model.02-0.35.h5')
	pred = model_SA_.predict(inp)
	response = {
		"output" : str(pred), 
	}

	return jsonify(response), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 5000, debug=True)