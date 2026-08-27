import matplotlib.pyplot as plt

def visualize_verifier_entropies(results: list[dict]):
    """
    results: the list your verifiers return (each entry has 'binary_entropy' and 'topk_entropy_at_label')
    """
    bin_H  = [r["binary_entropy"] for r in results if r.get("binary_entropy") is not None]
    topk_H = [r["topk_entropy_at_label"] for r in results if r.get("topk_entropy_at_label") is not None]

    if bin_H:
        plt.figure()
        plt.hist(bin_H, bins=30, edgecolor="black", alpha=0.8)
        plt.xlabel("Binary entropy H({YES,NO})")
        plt.ylabel("Count")
        plt.title("Verifier decision entropy")
        plt.tight_layout()

    if topk_H:
        plt.figure()
        plt.hist(topk_H, bins=30, edgecolor="black", alpha=0.8)
        plt.xlabel("Top-k entropy at label token (approx)")
        plt.ylabel("Count")
        plt.title("Token entropy (top-k approx) at verdict position")
        plt.tight_layout()

    if not bin_H and not topk_H:
        print("No entropy data found in results.")

# def plot_entropy_distributions(all_responses):
#     """
#     all_responses: list of dicts (entries from verifiers with 'logprobs')
#     """
#     entropies = []
#     for r in all_responses:
#         if r.get("logprobs"):
#             ent = token_entropy(r["logprobs"])
#             entropies.append(ent)

#     plt.hist(entropies, bins=30, alpha=0.7, edgecolor="black")
#     plt.xlabel("Token Entropy")
#     plt.ylabel("Frequency")
#     plt.title("Entropy Distribution of Verifier Tokens")
#     plt.show()