# demo_rps.py

Live Rock Paper Scissors demo. Robot picks randomly at GO; event camera classifies the human gesture; screen shows who won.

## Usage

```bash
python demo_rps.py --window_ms 50 [--repr histogram] [--offset_ms 150]
```

`SPACE` start / skip result screen — `ESC` quit

## Arguments

```
--window_ms    classification window in ms   (default: 50)
--repr         histogram | voxel | timesurface (default: histogram)
--offset_ms    delay after GO before window   (default: 150)
```

`--offset_ms` is the reaction time offset — sets τ from GO to window start. Use the τ value you want to demonstrate from RQ2.

## .env

```dotenv
SLIDING_ROOT=/media/lau/T7/thesis/sliding_window_time
```

Model path is derived automatically: `SLIDING_ROOT/<repr>/<window_ms>ms/merged/model_<repr>_<window_ms>ms_best.pth`

## Examples

**histogram, 50ms window at τ=30ms**

```bash
python demo_rps.py --window_ms 50 --repr histogram --offset_ms 30
```

**histogram, 20ms window at τ=40ms**

```bash
python demo_rps.py --window_ms 20 --repr histogram --offset_ms 40
```

**voxel, 50ms**

```bash
python demo_rps.py --window_ms 50 --repr voxel
```

**timesurface, 50ms**

```bash
python demo_rps.py --window_ms 50 --repr timesurface
```
