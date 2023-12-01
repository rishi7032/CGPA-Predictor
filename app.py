
import pandas as pd
from flask import Flask, render_template, request
import os
import csv

app = Flask(__name__)

DATA_CSV_PATH = r"Student details.csv"

csv_file_path = r"Student details.csv"
def encrypt_grade(grade):
    encryption_mapping = {'O': 10, 'A+': 9, 'A': 8, 'B+': 7, 'B': 6, 'C': 5}
    return encryption_mapping.get(grade, 0)
file_path = r"MYDATASET.csv"
df = pd.read_csv(file_path)
input = df.drop(["4Predictive Analytics","4Web and Social Media Analytics","4Professional Elective - 4","4Professional Elective - 5","4Open Elective - 2","4Seminar","4Web and Social Media Analytics Lab","4Mini Project ","4Project Stage - 1","4Organizational Behaviour","4Professional Elective - 6","4Open Elective - 3","4Project Stage - 2","No","CGPA"], axis=1)
target = df[["CGPA"]]
grades_enc = {'O':10, 'A+':9, 'A':8, 'B+':7, 'B':6, 'C':5}
type(input)
for item in input:
      input[item] = input[item].replace(grades_enc)
df = pd.concat([input, target], axis=1)
model1_columns = []
model2_columns = []
model3_columns = []
for item in [input]:
 for i in item:
   if i[0]=='1':
    model1_columns.append(i)
   elif i[0]=='2':
    model2_columns.append(i)
   else:
    model3_columns.append(i)

model2_columns = model1_columns + model2_columns
model3_columns = model2_columns + model3_columns
# Input data
labels = df['CGPA'].values
features = df[list(model1_columns)].values
# Split data

from sklearn.model_selection import train_test_split
random_seed=42

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30,random_state=random_seed)
# Build a Model

from sklearn import linear_model

# Bulid a new Model
lr_1 = linear_model.LinearRegression()

# Train the Model
lr_1.fit(X_train, y_train)
# Predict


# Predict for training set
y_train_predict = lr_1.predict(X_train)

# Predict for training set
y_test_predict = lr_1.predict(X_test)
# Input data
labels = df['CGPA'].values
features = df[list(model2_columns)].values
# Split data

from sklearn.model_selection import train_test_split
random_seed=42

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30,random_state=random_seed)
# Build a Model

from sklearn import linear_model

# Bulid a new Model
lr_2 = linear_model.LinearRegression()

# Train the Model
lr_2.fit(X_train, y_train)
# Predict


# Predict for training set
y_train_predict = lr_2.predict(X_train)

# Predict for training set
y_test_predict = lr_2.predict(X_test)
# Input data
labels = df['CGPA'].values
features = df[list(model3_columns)].values
# Split data

from sklearn.model_selection import train_test_split
random_seed=42

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30,random_state=random_seed)
# Build a Model

from sklearn import linear_model

# Bulid a new Model
lr_3 = linear_model.LinearRegression()

# Train the Model
lr_3.fit(X_train, y_train)
# Predict


# Predict for training set
y_train_predict = lr_3.predict(X_train)

# Predict for training set
y_test_predict = lr_3.predict(X_test)

@app.route('/', methods=['GET', 'POST'])
def home():
    selected_model = None
    grades_info = None 
    if request.method == 'POST':
        name = request.form['name']
        roll_number = request.form['roll_number']
        model = request.form['model']  # This is a string, not an int
        
        if model in ['1', '2', '3']:  # Use strings in the list for comparison
            # Find the row corresponding to the entered name and roll number
            selected_row = None
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row["Name"] == name and row["Roll Number"] == roll_number:
                        selected_row = row
                        break

            if selected_row is not None:
                if model == '1':
                    columns = model1_columns
                    lr_model = lr_1
                    selected_model = 'Model 1'
                elif model == '2':
                    columns = model2_columns
                    lr_model = lr_2
                    selected_model = 'Model 2'
                else:
                    columns = model3_columns
                    lr_model = lr_3
                    selected_model = 'Model 3'

                print(f"\n{name}'s Grades for Model {model}:")
                grades_info = {column: selected_row.get(column, "N/A") for column in columns}

                # Perform prediction using the selected linear regression model
                grades = [selected_row.get(column, 0) for column in columns]
                encrypted_grades = [encrypt_grade(grade) for grade in grades]
                input_lr = [float(grade) for grade in encrypted_grades]
                prediction = lr_model.predict([input_lr])

                return render_template('index.html', name=name, prediction=prediction[0], selected_model=selected_model, grades_info=grades_info)
            else:
                return render_template('index.html', name="Details not found", prediction="N/A", selected_model="N/A", grades_info=grades_info)
        else:
            return render_template('index.html', name="Invalid model", prediction="N/A", selected_model="N/A", grades_info=grades_info)

    return render_template('index.html')

@app.route('/get_details', methods=['POST'])
def get_details():
    name = request.form['name']
    roll_number = request.form['roll_number']
    model = request.form['model']

    selected_row = None
    with open(csv_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row["Name"] == name and row["Roll Number"] == roll_number:
                selected_row = row
                break

    if selected_row is not None:
        if model == '1':
            columns = model1_columns
            selected_model = 'Model 1'
        elif model == '2':
            columns = model2_columns
            selected_model = 'Model 2'
        else:
            columns = model3_columns
            selected_model = 'Model 3'

        grades_info = {column: selected_row.get(column, "N/A") for column in columns}
        details_html = "<h3>Grades Information:</h3><ul>"
        details_html += "".join([f"<li>{column}: {grade}</li>" for column, grade in grades_info.items()])
        details_html += "</ul>"

        return details_html
    else:
        return "Details not found"

@app.route('/predict', methods=['POST'])
def predict():
    selected_model = None
    grades_info = None

    if request.method == 'POST':
        name = request.form['name']
        roll_number = request.form['roll_number']
        model = request.form['model']

        if model in ['1', '2', '3']:
            # Find the row corresponding to the entered name and roll number
            selected_row = None
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row["Name"] == name and row["Roll Number"] == roll_number:
                        selected_row = row
                        break

            if selected_row is not None:
                if model == '1':
                    columns = model1_columns
                    lr_model = lr_1
                    selected_model = 'Model 1'
                elif model == '2':
                    columns = model2_columns
                    lr_model = lr_2
                    selected_model = 'Model 2'
                else:
                    columns = model3_columns
                    lr_model = lr_3
                    selected_model = 'Model 3'

                print(f"\n{name}'s Grades for Model {model}:")
                grades_info = {column: selected_row.get(column, "N/A") for column in columns}

                # Perform prediction using the selected linear regression model
                grades = [selected_row.get(column, 0) for column in columns]
                encrypted_grades = [encrypt_grade(grade) for grade in grades]
                input_lr = [float(grade) for grade in encrypted_grades]
                prediction = lr_model.predict([input_lr])

                return render_template('result.html', name=name, prediction=prediction[0], selected_model=selected_model, grades_info=grades_info)
            else:
                return render_template('result.html', name="Details not found", prediction="N/A", selected_model="N/A", grades_info=grades_info)
        else:
            return render_template('result.html', name="Invalid model", prediction="N/A", selected_model="N/A", grades_info=grades_info)


@app.route('/enterdetails')
def enterdetails():
    columns = get_csv_columns(DATA_CSV_PATH)
    return render_template('enterdetails.html', columns=columns)

@app.route('/submit', methods=['POST'])
def submit():
    data = {column: request.form[column] for column in request.form}
    save_to_csv(data)
    submission_successful = True
    columns = get_csv_columns(DATA_CSV_PATH)
    return render_template('enterdetails.html', columns=columns, submission_successful=submission_successful)

def get_csv_columns(file_path):
    df = pd.read_csv(file_path)
    return df.columns.tolist()

def save_to_csv(data):
    df = pd.DataFrame([data])
    
    df.to_csv(DATA_CSV_PATH, mode='a', index=False, header=not os.path.isfile(DATA_CSV_PATH))


if __name__ == '__main__':
    app.run(debug=True)

