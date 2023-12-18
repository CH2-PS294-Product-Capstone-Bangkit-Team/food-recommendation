from flask import Flask, render_template, jsonify, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = os.environ.get('FLASK_DEBUG')

# Load the trained model
model = load_model('sysrec.h5')

# Load ratings data
RATING_PATH = 'csv_dataset/Exported Rating 07-06-2021 16-35-04.csv'
rating = pd.read_csv(RATING_PATH)

# Encode user and food IDs
user_enc = LabelEncoder()
rating['User_ID'] = user_enc.fit_transform(rating['user_id'])

food_enc = LabelEncoder()
rating['Food_ID'] = food_enc.fit_transform(rating['food_code'])

# Create dictionaries for user and food mappings
user_dict = dict(zip(rating['user_id'], rating['User_ID']))
food_dict = dict(zip(rating['food_code'], rating['Food_ID']))

# Assuming zeros_df is the user-item matrix
zeros_df = pd.DataFrame(0, index=sorted(rating['User_ID'].unique()), columns=rating['Food_ID'].unique())

# Load food data for additional information
RATED_FOOD_PATH = 'csv_dataset/Exported Rated Food 07-06-2021 16-35-04.csv'
food_df = pd.read_csv(RATED_FOOD_PATH)

# Load food data for additional information
MENU_PATH = 'csv_dataset/Exported Food 07-06-2021 16-35-04.csv'
menu_df = pd.read_csv(MENU_PATH)


@app.route('/recommendation', methods=['POST'])
def predict():
    try:
        user_id = int(request.form['user_id'])
        USER_ID = user_dict[user_id]

        food_code_arr = []
        name_arr = []
        prediction_arr = []
        category_arr = []
        type_arr = []
        calories_arr = []
        protein_arr = []
        carbs_arr = []
        fat_arr = []
        fiber_arr = []
        sugar_arr = []
        vitamin_a_arr = []
        vitamin_b6_arr = []
        vitamin_b12_arr = []
        vitamin_c_arr = []
        vitamin_d_arr = []
        vitamin_e_arr = []

        not_rated = menu_df.loc[(~menu_df['food_code'].isin(zeros_df.columns)), 'food_code'].tolist()

        for rate in not_rated[:100]:  # Hanya menggunakan 10 pertama
            try:
                prediction = float(model.predict([np.array([USER_ID]), np.array([food_dict[rate]])])[0][0])

                # Dapatkan data makanan dari menu_df berdasarkan food_code
                food_data = menu_df.loc[menu_df['food_code'] == rate].iloc[0]

                food_code_arr.append(int(rate))
                name_arr.append(food_data['name'])
                prediction_arr.append(prediction)
                category_arr.append(food_data['category'])
                type_arr.append(food_data['type'])
                calories_arr.append(food_data['calories'])
                protein_arr.append(food_data['protein'])
                carbs_arr.append(food_data['carbs'])
                fat_arr.append(food_data['fat'])
                fiber_arr.append(food_data['fiber'])
                sugar_arr.append(food_data['sugar'])
                vitamin_a_arr.append(food_data['vitamin_a'])
                vitamin_b6_arr.append(food_data['vitamin_b6'])
                vitamin_b12_arr.append(food_data['vitamin_b12'])
                vitamin_c_arr.append(food_data['vitamin_c'])
                vitamin_d_arr.append(food_data['vitamin_d'])
                vitamin_e_arr.append(food_data['vitamin_e'])

            except:
                continue

        if not food_code_arr:
            return jsonify({
                'message': "Data tidak ditemukan",
                'category': "error",
                'status': 404,
                'description': "",
                'recommendation': ""
            }), 404  # Respon Not Found
        else:
            # Urutkan berdasarkan predicted_rating dari yang tertinggi ke terendah
            sorted_indices = sorted(range(len(prediction_arr)), key=lambda k: prediction_arr[k], reverse=True)
            food_code_arr = [food_code_arr[i] for i in sorted_indices][:10]
            name_arr = [name_arr[i] for i in sorted_indices][:10]
            prediction_arr = [prediction_arr[i] for i in sorted_indices][:10]
            category_arr = [category_arr[i] for i in sorted_indices][:10]
            type_arr = [type_arr[i] for i in sorted_indices][:10]
            calories_arr = [calories_arr[i] for i in sorted_indices][:10]
            protein_arr = [protein_arr[i] for i in sorted_indices][:10]
            carbs_arr = [carbs_arr[i] for i in sorted_indices][:10]
            fat_arr = [fat_arr[i] for i in sorted_indices][:10]
            fiber_arr = [fiber_arr[i] for i in sorted_indices][:10]
            sugar_arr = [sugar_arr[i] for i in sorted_indices][:10]
            vitamin_a_arr = [vitamin_a_arr[i] for i in sorted_indices][:10]
            vitamin_b6_arr = [vitamin_b6_arr[i] for i in sorted_indices][:10]
            vitamin_b12_arr = [vitamin_b12_arr[i] for i in sorted_indices][:10]
            vitamin_c_arr = [vitamin_c_arr[i] for i in sorted_indices][:10]
            vitamin_d_arr = [vitamin_d_arr[i] for i in sorted_indices][:10]
            vitamin_e_arr = [vitamin_e_arr[i] for i in sorted_indices][:10]

            response = {
                'food_code': food_code_arr,
                'name': name_arr,
                'predicted_rating': prediction_arr,
                'category': category_arr,
                'type': type_arr,
                'calories': calories_arr,
                'protein': protein_arr,
                'carbs': carbs_arr,
                'fat': fat_arr,
                'fiber': fiber_arr,
                'sugar': sugar_arr,
                'vitamin_a': vitamin_a_arr,
                'vitamin_b6': vitamin_b6_arr,
                'vitamin_b12': vitamin_b12_arr,
                'vitamin_c': vitamin_c_arr,
                'vitamin_d': vitamin_d_arr,
                'vitamin_e': vitamin_e_arr
            }
            return jsonify(response), 200  # Respon OK

    except ValueError as ve:
        return jsonify({
            'message': "Bad Request",
            'category': "error",
            'status': 400,
            'description': str(ve),
            'recommendation': ""
        }), 400  # Respon Bad Request
    except KeyError as ke:
        return jsonify({
            'message': "Data tidak ditemukan",
            'category': "error",
            'status': 404,
            'description': str(ke),
            'recommendation': ""
        }), 404  # Respon Not Found
    except Exception as e:
        return jsonify({
            'message': "Server Error",
            'category': "error",
            'status': 500,
            'description': str(e),
            'recommendation': ""
        }), 500  # Respon Server Error


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommendation', methods=['POST'])
def predict_route():
    return predict()


if __name__ == '__main__':
    app.run()
