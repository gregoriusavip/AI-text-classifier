from flask import Flask, request, render_template
from model_loader.classifier_rfc_loader import rfc_model
from model_loader.classifier_mnb_loader import mnb_model
import nltk

try:
   nltk.data.find('tokenizers/punkt')
except LookupError:
   nltk.download('punkt')

random_forest_model = rfc_model()
multinomial_model = mnb_model()
app = Flask(__name__)

def my_function(user_input):
   output = []
   output.append("using Random Forest model: " + random_forest_model.predict(user_input))
   output.append("using MultinomialNB model: " + multinomial_model.predict(user_input))
   
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