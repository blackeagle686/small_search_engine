from flask import Flask, render_template, request
from ai_part import search_by_city, preprocess_query, df, tfidf_matrix, tfidf_vectorizer
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Global caches
results_cache = []
query_cache = ""
city_name_cache = ""
num_results_all = 0
country_name_cache = ""

COUNTRIES = [
    'australia', 'brazil', 'india', 'indonesia', 'south africa',
    'kenya', 'uae', 'england', 'usa', 'spain', 'france',
    'malaysia', 'japan'
]

CITIES = [
    'Melbourne', 'Rio de Janeiro', 'Mumbai', 'Jakarta', 'Johannesburg',
    'Nairobi', 'Dubai', 'London', 'New York', 'Barcelona', 'Paris',
    'Kuala Lumpur', 'Tokyo', 'Miami'
]

def get_flag_path(country_name):
    if not country_name:
        return None
    safe_name = country_name.lower().replace(" ", "_")
    return f"./static/flags/{safe_name}.jpg"

def evaluate_results(relevant_indices, total_results):
    relevant_set = set(relevant_indices)
    truth_set = set(range(total_results))
    true_positives = len(relevant_set & truth_set)

    precision = true_positives / len(relevant_set) if relevant_set else 0
    recall = true_positives / len(truth_set) if truth_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    reciprocal_ranks = [
        1 / (i + 1) for i in range(total_results) if i in relevant_indices and i in truth_set
    ]
    mrr = reciprocal_ranks[0] if reciprocal_ranks else 0

    return round(precision, 3), round(recall, 3), round(f1, 3), round(mrr, 3)

@app.route("/", methods=["GET", "POST"])
def index():
    global results_cache, query_cache, city_name_cache, num_results_all, country_name_cache

    metrics = {}
    results = results_cache
    query = query_cache
    city_name = city_name_cache

    if "query" in request.form:
        query = request.form.get("query")
        num_results = int(request.form.get("num_results", 5))
        processed_query = preprocess_query(query)
    
        # Search by city name or  only Part
    

        # results = search_by_city(query)
    
        # # Compute avg and sort results by avg descending
        # for res in results:
        #     res["avg"] = (res["Score"] + res["Rating"]) / 2
    
        # # Sort by avg descending
        # results = sorted(results, key=lambda x: x["avg"], reverse=True)
        # results = results[:num_results]
        
        # ------------------------------------------------------------
        
        # Cosine Similarity Part
        query_vec = tfidf_vectorizer.transform([processed_query])
        cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        ranked_indices = cosine_similarities.argsort()[::-1]
        top_indices = ranked_indices[:num_results]
        results = df.iloc[top_indices].to_dict(orient="records")
    
        # Cache values
        results_cache = results[:num_results]
        query_cache = query
        city_name_cache = results[0]["City"] if results else ""
        country_name_cache = results[0]["Country"] if results else ""
        num_results_all = num_results


    elif "relevant" in request.form:
        relevant = request.form.getlist("relevant")
        relevant_indices = [int(i) for i in relevant]
        if relevant_indices:
            precision, recall, f1, mrr = evaluate_results(relevant_indices, num_results_all)
            metrics = {
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "MRR": mrr
            }

    flag_path = get_flag_path(country_name_cache)

    return render_template(
        "index.html",
        results=results,
        query=query,
        city_name=city_name,
        country_name=country_name_cache,
        flag_path=flag_path,
        metrics=metrics
    )


if __name__ == "__main__":
    app.run(port=5001, debug=True)
