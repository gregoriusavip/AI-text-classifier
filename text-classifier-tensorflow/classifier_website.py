from flask import Flask, request, render_template
from classifier_model import text_model

model = text_model()
app = Flask(__name__)

def sentiment_analysis():
   return None

def my_function(user_input):
   # Your Python script logic here
   output = "Inputted text is analyzed as: " + str(model.predict_text(user_input))
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