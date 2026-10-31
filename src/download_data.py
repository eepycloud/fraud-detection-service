"""Download the pinned public simulated dataset on a cloud runner."""
from pathlib import Path
import shutil
import kagglehub

ROOT=Path(__file__).resolve().parents[1]
def main():
    source=Path(kagglehub.dataset_download("kartik2112/fraud-detection/versions/1"))
    target=ROOT/"data"
    target.mkdir(exist_ok=True)
    for name in ("fraudTrain.csv","fraudTest.csv"):
        matches=list(source.rglob(name))
        if len(matches)!=1:raise RuntimeError(f"Expected one {name} in public dataset archive")
        shutil.copyfile(matches[0],target/name)
        print(f"Ready: {name} ({(target/name).stat().st_size:,} bytes)")
if __name__=="__main__": main()
