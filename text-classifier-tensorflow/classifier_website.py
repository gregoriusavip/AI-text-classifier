from flask import Flask, request, render_template
from classifier_tensorflow_model import text_model
from classifier_rfc import rfc_model

tensorflow_model = text_model()
random_forest_model = rfc_model()
app = Flask(__name__)

def my_function(user_input):
   output = []
   output.append("using tensorflow RNN model based of IMDB dataset: " + tensorflow_model.predict_text(user_input))
   output.append("using Random Forest model based of twitter comment dataset: " + random_forest_model.predict(user_input))
   
   return output

@app.route('/')
def home():
   return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
   user_input = "The text: "
   user_input += "<b>" + request.form['user_input'] + "</b>"
   output = my_function(user_input)
   return render_template('output.html', output=output, user_input=user_input)

if __name__ == '__main__':
   app.run(debug=True)