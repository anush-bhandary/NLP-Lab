from collections import defaultdict
import numpy as np

# Given movie reviews
documents = [
    ("fun, couple, love, love", "comedy"),
    ("fast, furious, shoot", "action"),
    ("couple, fly, fast, fun, fun", "comedy"),
    ("furious, shoot, shoot, fun", "action"),
    ("fly, fast, shoot, love", "action")
]

# New document
new_doc = "fast, couple, shoot, fly"

# Preprocess the data
def preprocess(doc):
    return doc.lower().split(", ")

# Initialize variables
word_counts = defaultdict(lambda: defaultdict(int))
class_counts = defaultdict(int)
vocab = set()

# Count words and classes
for doc, label in documents:
    words = preprocess(doc)
    class_counts[label] += 1
    for word in words:
        word_counts[label][word] += 1
        vocab.add(word)

# Total number of documents
total_docs = sum(class_counts.values())

# Prior probabilities
priors = {label: count / total_docs for label, count in class_counts.items()}

# Total word counts for each class and vocabulary size
total_words = {label: sum(word_counts[label].values()) for label in word_counts}
vocab_size = len(vocab)

# Likelihoods with add-1 smoothing
def likelihood(word, label):
    return (word_counts[label][word] + 1) / (total_words[label] + vocab_size)

# Compute posteriors for the new document
def compute_posterior(doc, label):
    words = preprocess(doc)
    posterior = np.log(priors[label])
    for word in words:
        posterior += np.log(likelihood(word, label))
    return posterior

# Classify the new document
posteriors = {label: compute_posterior(new_doc, label) for label in priors}
predicted_class = max(posteriors, key=posteriors.get)

print(f"The most likely class for the new document - '{new_doc}' is: {predicted_class}")
