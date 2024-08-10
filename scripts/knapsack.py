import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from ortools.algorithms.python import knapsack_solver

nltk.download('punkt')

def main():
    # Read the dataset from the file
    with open('/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/trainDir/datasets/babylm_100M24/simple_wiki.train', 'r', encoding='utf-8') as file:
        data = file.read()

    def parse_dataset(data):
        titles = []
        contexts = []

        segments = re.split(r'= = = (.*?) = = =', data)
        for i in range(1, len(segments), 2):
            title = segments[i].strip()
            context = segments[i + 1].strip().replace('\n', ' ')
            titles.append(title)
            contexts.append(context)

        return titles, contexts

    titles, contexts = parse_dataset(data)

    def compute_cosine_similarities(titles):
        vectorizer = TfidfVectorizer(max_features=5000).fit_transform(titles)
        vectors = vectorizer.toarray()
        cosine_matrix = cosine_similarity(vectors)

        # Compute the sum of similarities for each title (excluding self similarity)
        value_vector = cosine_matrix.sum(axis=1) - 1  # subtract 1 to exclude self similarity
        return value_vector

    value_vector = compute_cosine_similarities(titles).tolist()  # convert to list

    # Convert the value_vector to a list of integers
    value_vector = [int(v * 1000) for v in value_vector]  # multiplying by 1000 to handle small values

    def count_tokens(contexts):
        token_counts = [len(word_tokenize(context)) for context in contexts]
        return token_counts

    token_counts = count_tokens(contexts)

    # Create the solver.
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )

    values = value_vector
    weights = [token_counts]
    capacities = [500]

    solver.init(values, weights, capacities)
    computed_value = solver.solve()

    packed_items = []
    packed_weights = []
    total_weight = 0
    for i in range(len(values)):
        if solver.best_solution_contains(i):
            packed_items.append(i)
            packed_weights.append(weights[0][i])
            total_weight += weights[0][i]

    # Write the solution to a new text file
    with open('simple_wiki-5M.txt', 'w', encoding='utf-8') as file:
        file.write("".format(packed_items))
        for i in packed_items:
            file.write("{}\n".format(titles[i]))
            file.write("{}\n\n".format(contexts[i]))

if __name__ == "__main__":
    main()
