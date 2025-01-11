# heichole dataset processing

## dataset access

the heichole dataset is available at: [heichole on synapse](https://www.synapse.org/Synapse:syn18824884/wiki/592580)

dataset access steps:
1. register for a synapse account
2. request access to the dataset
3. navigate to "files" → "videos" → "full"
4. download the desired video files

## dependencies

- python 3.7+
- opencv (cv2)
- tqdm

## usage

```bash
python process_heichole.py /path/to/video /path/to/output
```
## advanced options
```bash
python process_heichole.py \
    /path/to/video \
    /path/to/output \
    --fps 1.0 \
    --size 512 512 \
    --format png
```
## parameters
```bash
- `--fps`: target frame rate for extraction (default: 1.0)
- `--size`: target frame size as width height (default: 512 512)
- `--format`: output image format (png or jpg, default: png)
```

## storage informations
storage requirements for processing:
- example for one video (807.78 mb mp4):
  - 54,930 frames at full frame rate
  - ~34.92 gb extracted frames
  - ratio: 1 mb mp4 ≈ 68 frames ≈ 43.23 mb extracted frames

the implemented solution reduces storage requirements through:
- extraction with reduced frame rate (default: 1 fps)
- frame size adjustment (default: 512x512)
- optional jpg compression