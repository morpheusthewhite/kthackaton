from recommandation.tree import build_tree
import json

if __name__ == "__main__":
    name = input("Enter the name of a film: ")
    tree = build_tree(name, 3)

    with open('tree.js', 'w') as f:
        suggestions_json = json.encoder.JSONEncoder().encode(tree)
        f.write("var tree = ")
        f.write(suggestions_json)
