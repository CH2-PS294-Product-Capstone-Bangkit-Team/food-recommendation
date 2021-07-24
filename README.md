# Menu-Suggestion

In this repository, some Jupyter notebook files are used to create a recommendation system on AyoHealthy App. Ayohealthy recommendation system implementing the content-based recommendation and collaborative filtering using matrix factorization method.


<h3> 1. Collaborative Filtering Recommendation with Tensorflow using Matrix Factorization Method </h3>
We choose the matrix Deep Learning Matrix Factorization method because it's good to handle sparsity on the data. After all, not every user will try every food/beverage on the system. So we use matrix factorization to predict the rating of the user to every food, then we sort the predicted value then give some highest predicted rating food to the user.
This first recommendation is created by using Tensorflow Python Library, step to step to creating this recommendation model is:

1. Import the rating and food dataset in the csv_dataset folder to Jupyter notebook.
2. With labelencoder() from the scikit_learn library we encode user_id and rating from the imported dataset.
3. After that we set up TensorFlow model for creating this recommendation, some layer that used on this model is:
    1. User Embedding Layer : After we encode the user_id data we need to feed it to the user embedding layer. The embedding layer is one of the methods that are good to handle sparsity
    2. Food Rating Embedding Layer : Same as the food we also need to feed the encoded food rating to the embedding layer.
    3. Merge Layer : In this layer, we merge both User Embedding and Food Rating Output by using concatenate function, before we merge it we need to flatten the embedding output first.
    3. Dense Layer : Then we add a dense layer, in this notebook we use 2 Dense Layer with every 128 and 64 nodes.
    4. Output : This is the output of user prediction rating to some food/beverages.
4. Split the rating data into 80% training data and 20% validation data.
5. Set up some callbacks to save the best epochs result, and stop the training when there is no significant improvement.
6. We feed the training and validation data to the model.
7. Evaluate the model.
8. Make a data frame rating prediction of every user to every food using the model.
9. Save the rating prediction data frame.

    
    
<h3> 2. Content-Based Recommendation using KNN </h3>
Because sometimes recommendations based on other user activity like collaborative filtering it's not good enough, we add some content based recommendation to find some similar food or beverages. Like if a user looking at the detail of Tuna Salad our system will recommend some food that closes / similar nutrition with Tuna Salad.

This type of recommendation we create it using the KNN model to find the n neighbors of the item. Here is some step to create this recommendation

1. Import the food dataset in the csv_dataset folder to Jupyter notebook.
2. Drop some unused columns like food name, category, type, and keep only the nutrition data.
3. By using MinMaxScaller we rescale the number data of nutrition, this is good to improve the result of the KNN model.
4. Then we feed train the data using the KNN model from the scikit_learn library. We use 10 nearest neighbors on this model
5. Try to predict the 10 nearest items for every food and beverage data.
6. Then save the trained KNN model to pickle format.


After done creating the model, we serve the recommendation the the frontend/mobile apps using API. 
[Link to the API Repository](https://github.com/imamseptian/Flask-Menu-Recommendation)
    

