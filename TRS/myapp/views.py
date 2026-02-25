import joblib
import numpy as np
from django.shortcuts import render
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load saved files
destination_data = joblib.load("destination_data.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Rebuild scaler (since you didn't save it)
scaler = MinMaxScaler()
scaler.fit(destination_data[["Cost", "Time"]])


def recommend_by_features(difficulty, cost, time, top_n=5):

    # Transform dataset features
    difficulty_matrix = vectorizer.transform(destination_data["Difficulty"])
    numeric_matrix = scaler.transform(destination_data[["Cost", "Time"]])

    dataset_matrix = np.hstack((difficulty_matrix.toarray(), numeric_matrix))

    # Transform user input
    user_difficulty = vectorizer.transform([difficulty])
    user_numeric = scaler.transform([[cost, time]])

    user_vector = np.hstack((user_difficulty.toarray(), user_numeric))

    # Compute similarity
    similarity_scores = cosine_similarity(user_vector, dataset_matrix)[0]

    # Get top matches
    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    return destination_data.iloc[top_indices]


def home(request):

    if request.method == "POST":
        difficulty = request.POST.get("difficulty")
        cost = float(request.POST.get("cost"))
        time = float(request.POST.get("time"))

        recommendations = recommend_by_features(difficulty, cost, time)

        return render(request, "recommend.html", {
    "recommendations": recommendations.to_dict(orient="records")
    })

    return render(request, "recommend.html")
    