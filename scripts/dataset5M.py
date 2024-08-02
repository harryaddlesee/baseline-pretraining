import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk

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
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(titles).toarray()
        norms = np.linalg.norm(tfidf_matrix, axis=1, keepdims=True)
        dot_product = np.dot(tfidf_matrix, tfidf_matrix.T)
        cosine_similarity_matrix = dot_product / (norms @ norms.T)
        
        # Compute the sum of similarities for each title (excluding self similarity)
        np.fill_diagonal(cosine_similarity_matrix, 0)
        value_vector = cosine_similarity_matrix.sum(axis=1)
        return value_vector

    value_vector = compute_cosine_similarities(titles).tolist()  # convert to list

    # Convert the value_vector to a list of integers
    value_vector = [int(v * 1000) for v in value_vector]  # multiplying by 1000 to handle small values

    def count_tokens(contexts):
        token_counts = [len(word_tokenize(context)) for context in contexts]
        return token_counts

    token_counts = count_tokens(contexts)

    # Define the capacity
    capacity = 5000000

    # Initialize the dynamic programming table with rolling arrays
    n = len(value_vector)
    dp = [(float('inf'), 0)] * (capacity + 1)
    dp[0] = (0, 0)  # (min_value, max_weight)

    for i in range(1, n + 1):
        new_dp = dp[:]
        for w in range(capacity + 1):
            if w >= token_counts[i - 1]:
                without_item = dp[w]
                with_item = (dp[w - token_counts[i - 1]][0] + value_vector[i - 1],
                             dp[w - token_counts[i - 1]][1] + token_counts[i - 1])

                if with_item[0] < without_item[0]:
                    new_dp[w] = with_item
                elif with_item[0] == without_item[0]:
                    new_dp[w] = max(with_item, without_item, key=lambda x: x[1])
                else:
                    new_dp[w] = without_item
        dp = new_dp

    # Find the minimum value that maximizes the weight within the capacity
    min_value, max_weight = min(dp)

    # Backtrack to find the selected items
    packed_items = []
    w = capacity
    for i in range(n, 0, -1):
        if w >= token_counts[i - 1] and dp[w] != dp[w - token_counts[i - 1]]:
            packed_items.append(i - 1)
            w -= token_counts[i - 1]

    packed_items.reverse()  # optional, to maintain the original order
    packed_weights = [token_counts[i] for i in packed_items]
    total_weight = sum(packed_weights)

    # Write the solution to a new text file
    with open('simple_wiki-5M.txt', 'w', encoding='utf-8') as file:
        file.write("Packed items = {}\n\n".format(packed_items))
        for i in packed_items:
            file.write("{}\n".format(titles[i]))
            file.write("{}\n\n".format(contexts[i]))

    print("Total value =", min_value)
    print("Total weight:", total_weight)
    print("Packed items:", packed_items)
    print("Packed weights:", packed_weights)

if __name__ == "__main__":
    main()

