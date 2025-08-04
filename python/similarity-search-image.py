import sys 
import json 
from embedder import testing_similarity



if len(sys.argv) != 3:
        print("Usage: python useimagecaptioningmodel.py <image_path>")
        sys.exit(1)

query = sys.argv[1]
k = sys.argv[2]


results=testing_similarity(query,k)

print(json.dumps(results))