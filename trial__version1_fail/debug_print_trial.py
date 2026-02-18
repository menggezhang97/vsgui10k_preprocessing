import gzip, json

PATH = "trials_with_elements.jsonl.gz"

PID = "015f21"
IMG = "0ff0be.png"
MEDIA_ID = 63

def main():
    with gzip.open(PATH, "rt", encoding="utf-8") as f:
        for line in f:
            tr = json.loads(line)
            if (
                tr["key"]["pid"] == PID
                and tr["key"]["img_name"] == IMG
                and tr["meta"]["media_id"] == MEDIA_ID
            ):
                print("\n===== TRIAL FOUND =====\n")
                print("trial_id:", tr["trial_id"])
                print("\n--- TASK ---")
                print(tr["task"])
                print("\n--- TARGET (seg-space bbox) ---")
                print(tr["target"]["bbox_seg"])
                print("\n--- SUMMARY ---")
                print(tr["summary"])
                print("\n--- META ---")
                print(tr["meta"])
                return

    print("Trial not found!")

if __name__ == "__main__":
    main()
