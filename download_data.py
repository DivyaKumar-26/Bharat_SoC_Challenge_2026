# Downloads only MVTec AD "bottle" (train/good + test) from the Hugging Face mirror foersben/mvtec-ad.
# Licence: CC BY-NC-SA 4.0 (non-commercial). Do NOT commit the images to GitHub.
from huggingface_hub import snapshot_download
snapshot_download("foersben/mvtec-ad", repo_type="dataset",
                  allow_patterns=["bottle/train/good/*", "bottle/test/*"], local_dir="mvtec")
print("done -> mvtec/bottle")
