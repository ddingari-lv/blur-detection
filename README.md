# blur-detection
Python script to calculate blur levels in an image.

## How to Setup
Steps:
1. Create a Python Virtual Environment:
```
python3.13.15 -m venv .venv
```
3. Activate Virtual Environment:
```
.venv/bin/activate
```
5. Import required Python Libraries:
```
pip install -r requirements.txt
```
<b>The script should now be ready to use.</b>

## How to Use
### Flags:
* --folder: Define the path to the directory the images you wish to process (REQUIRED)
* --out: Define an output path for the CSV file to export data (default="sharpness_table.csv")
* --resize_max: Resizes the longest side of the image to this size (default=1200, set to 0 to disable)
* --center_crop: If set to a value between 0-1, the images will be center-cropped by that fraction before processing
* --keep_low_pct: FFT low-frequency fraction to keep (default=0.05)
* --no-progress: Disables progress bar

### How to run:
After setting up the project, run this command:
```
python blur_detection.py --folder /path/to/dir/
```
To add more flags, add the flag identifiers to the end separated by spaces.
> Note: Other than `--folder /path/to/dir/`, all flags are optional and this script can be run without adding them.

```
python blur_detection.py --folder /path/to/dir/ --out /path/to/out/dir/ --resize_max 800 --center_crop 0.5 --keep_low_pct 0.02 --no-progress
```
