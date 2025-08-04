from useimagecaptioningmodel import generate_caption_preprocessed
from embedder import make_embedd
import sys 
import os

if len(sys.argv) != 2:
        print("Usage: python useimagecaptioningmodel.py <image_path>")
        sys.exit(1)

image_path = sys.argv[1]



try:
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)

        
        caption=generate_caption_preprocessed(image_path)
        make_embedd(image_path,caption)
        print("Caption:",caption)


except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)