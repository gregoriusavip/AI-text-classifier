from flask import Flask, request, render_template
from classifier_tensorflow_model import text_model
from classifier_rfc import rfc_model

tensorflow_model = text_model()
random_forest_model = rfc_model()
app = Flask(__name__)

def sentiment_analysis():
   return None

def my_function(user_input):
   # Your Python script logic here
   output = "Inputted text is analyzed as"
   output += "<br>&emsp;using tensorflow RNN model based of IMDB dataset: " + tensorflow_model.predict_text(user_input)
   output += "<br>&emsp;using Random Forest model based of twitter comment dataset: " + random_forest_model.predict(user_input)
   return output

@app.route('/')
def home():
   return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
   user_input = request.form['user_input']
   output = my_function(user_input)
   return render_template('output.html', output=output)

if __name__ == '__main__':
   app.run(debug=True)